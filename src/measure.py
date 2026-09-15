"""Convert a stitched detection mask into georeferenced operational measurements.

This module is the measurement boundary of the pipeline. It cleans probabilities,
measures connected regions in the raster's real ground units, transforms centroids
to WGS84, finds the nearest Natural Earth coastline, and derives alert and learned-
support classes. It never changes model inference or the screener pass footprint.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS as PyprojCRS
from pyproj import Geod, Transformer
from rasterio.crs import CRS
from rasterio.transform import Affine, array_bounds, xy
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_bounds
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, disk, remove_small_objects

import config

REGION_COLUMNS = (
    "region_id",
    "area_km2",
    "centroid_lon",
    "centroid_lat",
    "distance_to_coast_km",
    "mean_probability",
    "severity",
    "confidence",
    "axis_major_length_px",
    "axis_minor_length_px",
    "orientation_deg",
)

WGS84_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class _MetricCoastline:
    """Nearby coastline geometry and the metre-based CRS holding it."""

    geometry: object
    crs: PyprojCRS
    search_buffer_deg: float


def alert_level(
    area_km2: float,
    dist_coast_km: float | None,
) -> str:
    """Classify the operational impact of one detected region.

    Args:
        area_km2: Region area in square kilometres; must be non-negative.
        dist_coast_km: Nearest-coast distance in kilometres, or ``None`` when
            coastline measurement is unavailable.

    Returns:
        One of ``Low``, ``Moderate``, ``High``, or ``Critical``. Missing or
        non-finite coastline distance deliberately behaves as infinity so an
        ancillary-data failure cannot manufacture a coastal emergency.
    """

    distance = (
        float("inf")
        if dist_coast_km is None or not np.isfinite(dist_coast_km)
        else dist_coast_km
    )
    if area_km2 > 5 and distance < 15:
        return "Critical"
    if area_km2 > 5 and distance < 30:
        return "High"
    if area_km2 > 1 or distance < 10:
        return "Moderate"
    return "Low"


def severity(
    area_km2: float,
    dist_coast_km: float | None,
    mean_prob: float | None = None,
) -> str:
    """Backward-compatible alias for :func:`alert_level`.

    ``mean_prob`` is accepted for callers using the previous signature but is
    intentionally not blended into operational impact reporting.
    """

    return alert_level(area_km2, dist_coast_km)


def detection_confidence(mean_probability: float | None) -> str:
    """Classify the learned segmenter's mean support for one final region.

    Args:
        mean_probability: Mean segmenter output over every pixel in the final
            region, or ``None`` when unavailable. Classical-union pixels add no
            learned-positive support and therefore lower this mean.

    Returns:
        ``Low``, ``Medium``, or ``High`` using the natural-gap cut points stored
        in :mod:`config`. The result is descriptive support, not calibrated odds
        that the region is oil.
    """

    score = (
        float(mean_probability)
        if mean_probability is not None and np.isfinite(mean_probability)
        else float("-inf")
    )
    if score >= config.CONFIDENCE_HIGH_THRESHOLD:
        return "High"
    if score >= config.CONFIDENCE_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def postprocess_probabilities(
    probabilities: np.ndarray,
    cfg: object = config,
) -> np.ndarray:
    """Binarize and clean a probability raster using the configured mask rules.

    Args:
        probabilities: ``(height, width)`` array of pixel probabilities.
        cfg: Configuration object providing threshold, morphology, and
            connectivity values.

    Returns:
        A boolean array with the same shape. Processing order is threshold,
        binary opening, then minimum-component removal; changing that order can
        alter thin slick geometry.

    Raises:
        ValueError: Propagated by morphology if configuration values are invalid.
    """

    probability_array = np.asarray(probabilities, dtype=np.float32)
    binary = probability_array >= float(cfg.MASK_THRESHOLD)
    footprint = disk(int(cfg.MORPH_OPENING_RADIUS))
    opened = binary_opening(binary, footprint=footprint)
    cleaned = remove_small_objects(
        opened,
        min_size=int(cfg.MIN_BLOB_PX),
        connectivity=int(cfg.BLOB_CONNECTIVITY),
    )
    return cleaned.astype(bool, copy=False)


def _require_crs(crs: CRS | str | None) -> CRS:
    """Return a parsed CRS or fail rather than guessing measurement units."""

    if crs is None:
        raise ValueError(
            "Scene CRS is required for area and distance measurement; "
            "the input GeoTIFF does not define one."
        )
    try:
        return CRS.from_user_input(crs)
    except Exception as exc:
        raise ValueError(
            f"Scene CRS is invalid and cannot be measured: {crs!r}"
        ) from exc


def pixel_area_m2(transform: Affine, crs: CRS | str | None) -> float:
    """Return the constant affine pixel area for a projected scene.

    Args:
        transform: Raster affine transform.
        crs: Projected raster CRS.

    Returns:
        Constant area of one pixel in square metres.

    Raises:
        ValueError: If the CRS is missing, invalid, or geographic.

    This deliberately preserves the established projected-CRS calculation.
    Geographic pixels do not have a constant ground area and must use
    :func:`pixel_area_rows_m2` instead.
    """

    parsed_crs = _require_crs(crs)
    if parsed_crs.is_geographic:
        raise ValueError(
            "Geographic-CRS pixels require row-wise geodesic areas; "
            "use pixel_area_rows_m2()."
        )
    return abs(float(transform.a * transform.e))


def pixel_area_rows_m2(
    transform: Affine,
    crs: CRS | str | None,
    height: int,
    width: int,
) -> np.ndarray:
    """Return one WGS84 ground-area value per raster row.

    Args:
        transform: Raster affine transform.
        crs: Projected or geographic raster CRS.
        height: Raster height in pixels.
        width: Raster width in pixels; validated for consistency even though a
            north-up row shares one pixel area across all columns.

    Returns:
        ``float64[height]`` square-metre area values, one per raster row.

    Raises:
        ValueError: If dimensions or CRS are invalid, or a geographic raster is
            rotated and cannot be represented by one area per row.

    For projected scenes the returned array repeats the existing affine pixel
    area. For geographic scenes this is the accepted Part III method: measure
    one pixel's east-west and north-south dimensions geodesically at each row
    and multiply them. North-up grids are required because a rotated geographic
    grid does not have one area shared by every pixel in a row.
    """

    parsed_crs = _require_crs(crs)
    if height <= 0 or width <= 0:
        raise ValueError(
            "Raster height and width must be positive for area measurement."
        )
    if parsed_crs.is_projected:
        return np.full(height, pixel_area_m2(transform, parsed_crs), dtype=np.float64)
    if not parsed_crs.is_geographic:
        raise ValueError(f"Unsupported CRS type for area measurement: {parsed_crs}")
    if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
        raise ValueError(
            "Rotated geographic rasters are not supported by row-wise geodesic "
            "area measurement; reproject to a north-up grid."
        )

    rows = np.arange(height, dtype=np.float64)
    row_top = transform.f + rows * transform.e
    row_bottom = row_top + transform.e
    row_center = (row_top + row_bottom) / 2.0
    pixel_left = np.full(height, transform.c, dtype=np.float64)
    pixel_right = np.full(height, transform.c + transform.a, dtype=np.float64)
    pixel_center = (pixel_left + pixel_right) / 2.0

    to_wgs84 = Transformer.from_crs(parsed_crs, "EPSG:4326", always_xy=True)
    left_lon, left_lat = to_wgs84.transform(pixel_left, row_center)
    right_lon, right_lat = to_wgs84.transform(pixel_right, row_center)
    top_lon, top_lat = to_wgs84.transform(pixel_center, row_top)
    bottom_lon, bottom_lat = to_wgs84.transform(pixel_center, row_bottom)
    _, _, width_m = WGS84_GEOD.inv(left_lon, left_lat, right_lon, right_lat)
    _, _, height_m = WGS84_GEOD.inv(top_lon, top_lat, bottom_lon, bottom_lat)
    row_areas = np.abs(
        np.asarray(width_m, dtype=np.float64) * np.asarray(height_m, dtype=np.float64)
    )
    if not np.isfinite(row_areas).all() or np.any(row_areas <= 0):
        raise ValueError("Geodesic pixel-area calculation produced invalid values.")
    return row_areas


def mask_area_m2(
    mask: np.ndarray,
    transform: Affine,
    crs: CRS | str | None,
) -> float:
    """Measure a binary raster in square metres for either CRS family.

    Args:
        mask: Two-dimensional array; nonzero values are measured.
        transform: Raster affine transform.
        crs: Projected or geographic raster CRS.

    Returns:
        Total true-pixel ground area in square metres.

    Raises:
        ValueError: If the mask is not 2D or the CRS cannot be measured.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("Area measurement expects a two-dimensional mask.")
    parsed_crs = _require_crs(crs)
    if parsed_crs.is_projected:
        return float(np.count_nonzero(binary) * pixel_area_m2(transform, parsed_crs))
    row_areas = pixel_area_rows_m2(
        transform,
        parsed_crs,
        binary.shape[0],
        binary.shape[1],
    )
    pixels_by_row = binary.sum(axis=1, dtype=np.int64)
    return float((pixels_by_row * row_areas).sum())


def bounds_lonlat(
    width: int,
    height: int,
    transform: Affine,
    crs: CRS | str,
) -> tuple[float, float, float, float]:
    """Return densified scene bounds as west, south, east, north in EPSG:4326.

    Densifying the edges before reprojection avoids underestimating bounds for
    projected rasters whose edges curve in longitude/latitude space.

    Raises:
        CRSError: If ``crs`` cannot be transformed to WGS84.
    """

    west, south, east, north = array_bounds(height, width, transform)
    return tuple(
        float(value)
        for value in transform_bounds(
            crs,
            "EPSG:4326",
            west,
            south,
            east,
            north,
            densify_pts=21,
        )
    )


def _load_clipped_coastline(
    coastline_path: str | Path | None,
    scene_bounds_lonlat: tuple[float, float, float, float],
    scene_crs: CRS,
    search_buffer_deg: float = config.COASTLINE_SEARCH_BUFFER_DEG,
) -> _MetricCoastline | None:
    """Load real coastlines in a buffered window and project them to metres.

    The configured window is tried first. If it contains no geometry, one
    wider 5-degree search is attempted before distance-to-coast is declared
    unavailable. The scene's projected CRS is retained when it uses metres;
    otherwise a local WGS84 azimuthal-equidistant CRS is used.
    """

    if coastline_path is None:
        return None
    path = Path(coastline_path)
    if not path.exists():
        warnings.warn(
            f"Coastline file is missing at {path}; distances will be blank. "
            "Restore the Natural Earth 1:10m coastline shapefile and its "
            ".dbf/.shx/.prj/.cpg siblings under data/natural_earth/; see "
            "https://www.naturalearthdata.com/downloads/10m-physical-vectors/.",
            UserWarning,
            stacklevel=2,
        )
        return None

    initial_buffer = float(search_buffer_deg)
    if not np.isfinite(initial_buffer) or initial_buffer <= 0:
        raise ValueError("coastline search buffer must be a positive finite value")

    scene_metric_crs = PyprojCRS.from_user_input(scene_crs)
    axis_units_are_metres = bool(scene_metric_crs.axis_info) and all(
        axis.unit_conversion_factor is not None
        and np.isclose(float(axis.unit_conversion_factor), 1.0)
        for axis in scene_metric_crs.axis_info
    )
    if not scene_metric_crs.is_projected or not axis_units_are_metres:
        west, south, east, north = scene_bounds_lonlat
        centre_lon = (west + east) / 2.0
        centre_lat = (south + north) / 2.0
        scene_metric_crs = PyprojCRS.from_proj4(
            "+proj=aeqd "
            f"+lat_0={centre_lat:.12f} +lon_0={centre_lon:.12f} "
            "+datum=WGS84 +units=m +no_defs"
        )

    search_buffers = [initial_buffer]
    if initial_buffer < 5.0 and not np.isclose(initial_buffer, 5.0):
        search_buffers.append(5.0)

    try:
        import geopandas as gpd
        from shapely.geometry import box

        west, south, east, north = scene_bounds_lonlat
        for margin_deg in search_buffers:
            search_bounds = (
                max(-180.0, west - margin_deg),
                max(-90.0, south - margin_deg),
                min(180.0, east + margin_deg),
                min(90.0, north + margin_deg),
            )
            coastline = gpd.read_file(path, bbox=search_bounds)
            if coastline.empty:
                continue
            if coastline.crs is None:
                warnings.warn(
                    "Coastline file has no CRS; distance-to-coast is unavailable. "
                    "Restore the complete Natural Earth shapefile, including its "
                    ".prj sibling, under data/natural_earth/.",
                    UserWarning,
                    stacklevel=2,
                )
                return None
            search_polygon = (
                gpd.GeoSeries([box(*search_bounds)], crs="EPSG:4326")
                .to_crs(coastline.crs)
                .iloc[0]
            )
            coastline = coastline.clip(search_polygon)
            coastline = coastline.loc[
                coastline.geometry.notna() & ~coastline.geometry.is_empty
            ]
            if coastline.empty:
                continue
            projected = coastline.to_crs(scene_metric_crs)
            geometry = projected.geometry.union_all()
            if geometry is not None and not geometry.is_empty:
                return _MetricCoastline(
                    geometry=geometry,
                    crs=scene_metric_crs,
                    search_buffer_deg=float(margin_deg),
                )
        warnings.warn(
            "No coastline geometry was found within the configured search "
            f"window or the 5-degree fallback around bounds {scene_bounds_lonlat}; "
            "distances will be blank.",
            UserWarning,
            stacklevel=2,
        )
        return None
    except Exception as exc:  # Ancillary data should never abort spill detection.
        warnings.warn(
            "Could not load coastline data; distances will be blank. Confirm the "
            "Natural Earth .shp/.dbf/.shx/.prj/.cpg files are together under "
            "data/natural_earth/ and install requirements-lock.txt. "
            f"Underlying error: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None


def measure_regions(
    probabilities: np.ndarray,
    binary: np.ndarray,
    transform: Affine,
    crs: CRS | str | None,
    coastline_path: str | Path | None = config.COASTLINE_PATH,
    coastline_search_buffer_deg: float = config.COASTLINE_SEARCH_BUFFER_DEG,
) -> pd.DataFrame:
    """Measure connected detections and attach geospatial classifications.

    Args:
        probabilities: ``float32[height,width]`` learned probability mosaic.
        binary: Boolean final mask aligned with ``probabilities``.
        transform: Source raster affine transform.
        crs: Source raster CRS; missing CRS is rejected rather than guessed.
        coastline_path: Natural Earth shapefile or ``None`` to omit distances.
        coastline_search_buffer_deg: Positive initial WGS84 search margin.

    Returns:
        A dataframe with one row per eight-connected region and the stable
        :data:`REGION_COLUMNS` schema.

    Raises:
        ValueError: If arrays do not align or geospatial metadata is invalid.

    Geographic rasters use WGS84 geodesic area row by row because a degree of
    longitude shrinks with latitude. Projected rasters retain the original
    constant affine-pixel calculation to preserve reference UTM results.
    """

    probability_array = np.asarray(probabilities, dtype=np.float32)
    binary_array = np.asarray(binary, dtype=bool)
    if probability_array.shape != binary_array.shape or probability_array.ndim != 2:
        raise ValueError("probabilities and binary must be same-shaped 2D arrays.")

    parsed_crs = _require_crs(crs)
    geographic_row_areas = (
        pixel_area_rows_m2(
            transform,
            parsed_crs,
            binary_array.shape[0],
            binary_array.shape[1],
        )
        if parsed_crs.is_geographic
        else None
    )
    # Keep the established scalar operation exactly unchanged for projected
    # scenes so the reference UTM region tables remain byte-identical.
    area_per_pixel = (
        None
        if geographic_row_areas is not None
        else pixel_area_m2(transform, parsed_crs)
    )
    scene_bounds = bounds_lonlat(
        probability_array.shape[1],
        probability_array.shape[0],
        transform,
        parsed_crs,
    )
    metric_coastline = _load_clipped_coastline(
        coastline_path,
        scene_bounds,
        parsed_crs,
        search_buffer_deg=coastline_search_buffer_deg,
    )
    to_distance_crs = None
    if metric_coastline is not None:
        to_distance_crs = Transformer.from_crs(
            PyprojCRS.from_user_input(parsed_crs),
            metric_coastline.crs,
            always_xy=True,
        )

    labelled = label(binary_array, connectivity=config.BLOB_CONNECTIVITY)
    rows: list[dict[str, object]] = []
    for region in regionprops(labelled, intensity_image=probability_array):
        centroid_row, centroid_col = region.centroid
        easting, northing = xy(
            transform,
            centroid_row,
            centroid_col,
            offset="center",
        )
        longitudes, latitudes = warp_transform(
            parsed_crs,
            "EPSG:4326",
            [easting],
            [northing],
        )
        distance_km = None
        if metric_coastline is not None and to_distance_crs is not None:
            from shapely.geometry import Point

            distance_x, distance_y = to_distance_crs.transform(easting, northing)
            distance_km = float(
                Point(float(distance_x), float(distance_y)).distance(
                    metric_coastline.geometry
                )
                / 1_000.0
            )

        if geographic_row_areas is not None:
            min_row, _, max_row, _ = region.bbox
            pixels_by_row = region.image.sum(axis=1, dtype=np.int64)
            area_m2 = float(
                (pixels_by_row * geographic_row_areas[min_row:max_row]).sum()
            )
        else:
            area_m2 = float(region.area * area_per_pixel)
        area_km2 = area_m2 / 1_000_000.0
        mean_probability = float(region.mean_intensity)
        rows.append(
            {
                "region_id": int(region.label),
                "area_km2": area_km2,
                "centroid_lon": float(longitudes[0]),
                "centroid_lat": float(latitudes[0]),
                "distance_to_coast_km": distance_km,
                "mean_probability": mean_probability,
                "severity": severity(area_km2, distance_km, mean_probability),
                "confidence": detection_confidence(mean_probability),
                "axis_major_length_px": float(region.axis_major_length),
                "axis_minor_length_px": float(region.axis_minor_length),
                "orientation_deg": float(np.degrees(region.orientation)),
            }
        )

    return pd.DataFrame(rows, columns=REGION_COLUMNS)


def overall_severity(regions: pd.DataFrame, cfg: object = config) -> str:
    """Return the scene's highest alert while preserving the legacy API name.

    Args:
        regions: Region dataframe returned by :func:`measure_regions`.
        cfg: Configuration carrying the ordered alert labels.

    Returns:
        Highest regional alert, or ``Low`` for an empty scene.
    """

    return overall_alert_and_confidence(regions, cfg)[0]


def overall_alert_and_confidence(
    regions: pd.DataFrame,
    cfg: object = config,
) -> tuple[str, str]:
    """Return independently aggregated scene alert and confidence classes.

    Args:
        regions: Region dataframe containing ``severity`` and ``confidence``.
        cfg: Configuration carrying both orderings.

    Returns:
        ``(alert, confidence)``. Alert is the highest impact class across all
        regions; confidence is the highest learned-support class across all
        regions. Empty scenes return ``("Low", "Low")``.

    Alert is the highest impact class across all regions. Confidence is the
    highest learned-support class across all regions and may therefore come
    from a different detection than the scene alert.
    """

    if regions.empty:
        return "Low", "Low"
    alert_ranking = {name: index for index, name in enumerate(cfg.SEVERITY_ORDER)}
    confidence_ranking = {
        name: index for index, name in enumerate(cfg.CONFIDENCE_ORDER)
    }
    scene_alert = max(
        (str(value) for value in regions["severity"]),
        key=lambda value: alert_ranking.get(value, -1),
    )
    scene_confidence = max(
        (str(value) for value in regions["confidence"]),
        key=lambda value: confidence_ranking.get(value, -1),
    )
    return scene_alert, scene_confidence


def overall_confidence(regions: pd.DataFrame, cfg: object = config) -> str:
    """Return the highest regional learned-support class in the scene.

    The class may come from a different region than the scene alert. Empty
    scenes return ``Low``.
    """

    return overall_alert_and_confidence(regions, cfg)[1]

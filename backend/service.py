"""Own analysis state, cache expensive results, and build API-ready records.

The service is the boundary between HTTP concerns and scientific processing. It
resolves packaged scenes or content-addressed uploads, calls the production
pipeline once per scene/configuration pair, computes the post-hoc diagnostic,
and retains a bounded thread-safe least-recently-used cache for renders/exports.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile

import config
from backend.diagnostics import UndetectedDarkResult, compute_undetected_dark
from src.pipeline import PipelineResult, run_pipeline

DEMO_SCENE_NAMES = (
    "jeddah_oct_2019_vv_db_10m.tif",
    "jeddah_oct25_2019_vv_db_10m.tif",
    "gulf_open_water_oct2019_vv_db_10m.tif",
    "kuwait_aug_2017_vv_db_10m.tif",
)

DISPLAY_NAMES = {
    "jeddah_oct_2019_vv_db_10m.tif": "Jeddah — 13 Oct 2019",
    "jeddah_oct25_2019_vv_db_10m.tif": "Jeddah — 25 Oct 2019",
    "gulf_open_water_oct2019_vv_db_10m.tif": "Arabian Gulf — open-water control",
    "kuwait_aug_2017_vv_db_10m.tif": "Kuwait — Aug 2017",
}

OPERATING_STATUS = (
    f"Screener {config.SCREENER_THRESHOLD:.2f} / "
    f"Segmenter {config.MASK_THRESHOLD:.2f} / "
    f"Classical union {'on' if config.CLASSICAL_UNION_ENABLED else 'off'}"
)


def _config_hash() -> str:
    """Hash every setting and file identity that can alter analysis."""

    fields = (
        "RANDOM_SEED",
        "TILE_SIZE",
        "STRIDE",
        "SCREENER_THRESHOLD",
        "MASK_THRESHOLD",
        "BASELINE_THRESHOLD",
        "MAX_INVALID_FRACTION",
        "INFILL_INVALID_PIXELS",
        "NORM_SCOPE",
        "NORM_LO_PCT",
        "NORM_HI_PCT",
        "FIXED_DB_LO",
        "FIXED_DB_HI",
        "SCREENER_PASS_MAP_CLOSING",
        "SCREENER_PASS_MAP_HOLE_FILLING",
        "CLASSICAL_UNION_ENABLED",
        "CLASSICAL_UNION_RESTRICT_TO_PASS_FOOTPRINT",
        "CHAIN_BINARY_FILL_HOLES",
        "MORPH_OPENING_RADIUS",
        "MIN_BLOB_PX",
        "BLOB_CONNECTIVITY",
        "COASTLINE_SEARCH_BUFFER_DEG",
        "CONFIDENCE_MEDIUM_THRESHOLD",
        "CONFIDENCE_HIGH_THRESHOLD",
        "CONFIDENCE_ORDER",
        "SEVERITY_ORDER",
    )
    payload: dict[str, Any] = {name: getattr(config, name) for name in fields}
    paths = (
        Path(config.OPERATING_SCREENER_CHECKPOINT),
        Path(config.OPERATING_SEGMENTER_CHECKPOINT),
        Path(config.COASTLINE_PATH),
    )
    coastline_base = Path(config.COASTLINE_PATH).with_suffix("")
    paths += tuple(
        coastline_base.with_suffix(suffix)
        for suffix in (".dbf", ".shx", ".prj", ".cpg")
    )
    payload["files"] = []
    for path in paths:
        stat = path.stat() if path.exists() else None
        payload["files"].append(
            {
                "path": str(path.resolve()),
                "size": None if stat is None else stat.st_size,
                "mtime_ns": None if stat is None else stat.st_mtime_ns,
            }
        )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _region_status(
    bounds: tuple[float, float, float, float],
) -> tuple[str, float, float, bool]:
    west, south, east, north = bounds
    lon = (west + east) / 2.0
    lat = (south + north) / 2.0
    region_name = "Outside Saudi focus regions"
    for name, limits in config.REGION_BOUNDS.items():
        if (
            limits["lon"][0] <= lon <= limits["lon"][1]
            and limits["lat"][0] <= lat <= limits["lat"][1]
        ):
            region_name = name
            break
    validated = config.VALIDATED_REGION_BOUNDS.get(region_name)
    inside_validated = bool(
        validated
        and validated["lon"][0] <= lon <= validated["lon"][1]
        and validated["lat"][0] <= lat <= validated["lat"][1]
    )
    return region_name, lat, lon, inside_validated


def _pixel_size(result: PipelineResult) -> dict[str, float | str]:
    transform = result.transform
    x_size = float(np.hypot(transform.a, transform.b))
    y_size = float(np.hypot(transform.d, transform.e))
    if result.crs.is_projected:
        factor = result.crs.linear_units_factor
        multiplier = float(factor[1]) if isinstance(factor, tuple) else float(factor)
        return {"x": x_size * multiplier, "y": y_size * multiplier, "unit": "m"}
    return {"x": x_size, "y": y_size, "unit": "degrees"}


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _region_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = frame.sort_values("area_km2", ascending=False).reset_index(drop=True)
    return [
        {str(column): _json_value(value) for column, value in row.items()}
        for row in ordered.to_dict(orient="records")
    ]


@dataclass(frozen=True)
class AnalysisRecord:
    """Immutable source data and outputs retained for one analyzed scene.

    Full-resolution arrays stay on the source ``(height, width)`` grid. Keeping
    them with the pipeline result lets layer rendering and exports reuse analysis
    without invoking either neural network again.
    """

    scene_id: str
    source_name: str
    display_name: str
    result: PipelineResult
    undetected_dark: UndetectedDarkResult
    source_image: np.ndarray
    source_nodata: float | None
    width: int
    height: int
    config_hash: str

    def response(self, *, cached: bool) -> dict[str, Any]:
        """Serialize this record into the stable analysis-response schema.

        Args:
            cached: Whether the current request reused this stored record.

        Returns:
            JSON-compatible metrics, regions, diagnostics, and metadata. NaN and
            infinity are converted to ``null`` instead of invalid JSON numbers.
        """

        regions = self.result.regions_df
        coast = regions["distance_to_coast_km"].dropna()
        largest = float(regions["area_km2"].max()) if not regions.empty else 0.0
        invalid_excluded = sum(
            tile.get("reason") == "invalid_pixel_fraction_above_threshold"
            for tile in self.result.rejected_tiles
        )
        screened_out = len(self.result.rejected_tiles) - invalid_excluded
        region_name, centre_lat, centre_lon, validated = _region_status(
            self.result.scene_bounds_lonlat
        )
        return {
            "scene_id": self.scene_id,
            "source_name": self.source_name,
            "display_name": self.display_name,
            "cached": cached,
            "config_hash": self.config_hash,
            "total_area_km2": float(self.result.total_area_km2),
            "region_count": int(self.result.n_regions),
            "largest_region_km2": largest,
            "nearest_coast_km": None if coast.empty else float(coast.min()),
            "alert_level": self.result.overall_severity,
            "confidence": self.result.overall_confidence,
            "severity": self.result.overall_severity,
            "regions": _region_records(regions),
            "undetected_dark": self.undetected_dark.response(),
            "metadata": {
                "pixel_size": _pixel_size(self.result),
                "width_px": self.width,
                "height_px": self.height,
                "invalid_tiles_excluded": int(invalid_excluded),
                "tiles_screened_out": int(screened_out),
                "region_label": region_name,
                "normalization_scope": self.result.norm_scope_used,
                "operating_configuration": OPERATING_STATUS,
                "scene_bounds_lonlat": [
                    float(value) for value in self.result.scene_bounds_lonlat
                ],
                "centre_lat": centre_lat,
                "centre_lon": centre_lon,
                "inside_validated_bounds": validated,
            },
        }


class AnalysisService:
    """Thread-safe in-memory LRU around the released production pipeline.

    Records are keyed by ``(scene_id, config_hash)``. Demo IDs are filenames;
    uploads use the first 20 hexadecimal characters of their SHA-256 digest.
    Least-recently-used records are evicted when ``max_records`` is exceeded.
    The cache is process-local and intentionally disappears on restart.
    """

    def __init__(self, max_records: int = 6) -> None:
        self._max_records = max_records
        self._records: OrderedDict[tuple[str, str], AnalysisRecord] = OrderedDict()
        self._lock = threading.RLock()
        self.inference_runs = 0

    @property
    def config_hash(self) -> str:
        """Return the active settings and asset identity used in cache keys."""

        return _config_hash()

    def list_scenes(self) -> list[dict[str, Any]]:
        """Describe packaged demo scenes that currently exist on disk.

        Returns:
            Scene identifiers, display names, regional labels, and validated-
            bounds flags.

        Raises:
            RasterioError: If an existing demo cannot be opened or transformed.
        """

        scenes: list[dict[str, Any]] = []
        for name in DEMO_SCENE_NAMES:
            path = Path(config.DEMO_SCENES_DIR) / name
            if not path.exists():
                continue
            with rasterio.open(path) as src:
                from src.measure import bounds_lonlat

                bounds = bounds_lonlat(src.width, src.height, src.transform, src.crs)
            region, _, _, validated = _region_status(bounds)
            scenes.append(
                {
                    "scene_id": name,
                    "name": name,
                    "display_name": DISPLAY_NAMES[name],
                    "region_label": region,
                    "inside_validated_bounds": validated,
                }
            )
        return scenes

    def analyze_demo(self, name: str) -> tuple[AnalysisRecord, bool]:
        """Return a cached or newly analyzed packaged demo scene.

        Args:
            name: Exact filename from :data:`DEMO_SCENE_NAMES`.

        Returns:
            ``(record, cache_hit)``.

        Raises:
            KeyError: If ``name`` is not a designated demo.
            FileNotFoundError: If the designated GeoTIFF is absent.
            RuntimeError: If CUDA or model requirements are unavailable.
        """

        if name not in DEMO_SCENE_NAMES:
            raise KeyError(f"Unknown demo scene: {name}")
        path = Path(config.DEMO_SCENES_DIR) / name
        if not path.exists():
            raise FileNotFoundError(
                f"Demo scene is missing: {path}. Download "
                "saudi-oil-spill-assets.zip from "
                "https://github.com/YOURNAME/saudi-oil-spill-system/releases/latest "
                "and extract it from the repository root to install the scenes "
                "in data/demo_scenes/."
            )
        config_hash = self.config_hash
        key = (name, config_hash)
        with self._lock:
            cached = self._records.get(key)
            if cached is not None:
                self._records.move_to_end(key)
                return cached, True
            result = run_pipeline(str(path), config)
            with rasterio.open(path) as src:
                image = src.read(1).astype(np.float32, copy=False)
                nodata = src.nodata
                width, height = src.width, src.height
            undetected_dark = compute_undetected_dark(image, nodata, result, config)
            record = AnalysisRecord(
                scene_id=name,
                source_name=name,
                display_name=DISPLAY_NAMES[name],
                result=result,
                undetected_dark=undetected_dark,
                source_image=image,
                source_nodata=nodata,
                width=width,
                height=height,
                config_hash=config_hash,
            )
            self.inference_runs += 1
            self._store(key, record)
            return record, False

    def analyze_upload(self, data: bytes, filename: str) -> tuple[AnalysisRecord, bool]:
        """Analyze an in-memory GeoTIFF using a content-addressed cache key.

        Args:
            data: Complete uploaded GeoTIFF bytes.
            filename: Original client filename, used only for display.

        Returns:
            ``(record, cache_hit)``. Identical bytes share an analysis even when
            uploaded under different names.

        Raises:
            ValueError: If the upload is empty or fails pipeline validation.
        """

        if not data:
            raise ValueError("Uploaded GeoTIFF is empty.")
        content_hash = hashlib.sha256(data).hexdigest()
        scene_id = f"upload-{content_hash[:20]}"
        config_hash = self.config_hash
        key = (scene_id, config_hash)
        with self._lock:
            cached = self._records.get(key)
            if cached is not None:
                self._records.move_to_end(key)
                return cached, True
            with MemoryFile(data) as memory_file:
                result = run_pipeline(memory_file, config)
                with memory_file.open() as src:
                    image = src.read(1).astype(np.float32, copy=False)
                    nodata = src.nodata
                    width, height = src.width, src.height
            undetected_dark = compute_undetected_dark(image, nodata, result, config)
            record = AnalysisRecord(
                scene_id=scene_id,
                source_name=filename,
                display_name=Path(filename).stem.replace("_", " "),
                result=result,
                undetected_dark=undetected_dark,
                source_image=image,
                source_nodata=nodata,
                width=width,
                height=height,
                config_hash=config_hash,
            )
            self.inference_runs += 1
            self._store(key, record)
            return record, False

    def get(self, scene_id: str) -> AnalysisRecord:
        """Retrieve an analysis, lazily processing a known demo on cache miss.

        Args:
            scene_id: Demo filename or content-addressed upload ID.

        Returns:
            Cached analysis record.

        Raises:
            KeyError: If an upload was evicted or the ID is unknown.
        """

        config_hash = self.config_hash
        key = (scene_id, config_hash)
        with self._lock:
            record = self._records.get(key)
            if record is not None:
                self._records.move_to_end(key)
                return record
        if scene_id in DEMO_SCENE_NAMES:
            record, _ = self.analyze_demo(scene_id)
            return record
        raise KeyError(f"No cached analysis exists for scene: {scene_id}")

    def _store(self, key: tuple[str, str], record: AnalysisRecord) -> None:
        self._records[key] = record
        self._records.move_to_end(key)
        while len(self._records) > self._max_records:
            self._records.popitem(last=False)


analysis_service = AnalysisService()

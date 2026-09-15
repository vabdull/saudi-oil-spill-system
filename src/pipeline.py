"""Orchestrate GeoTIFF validation, inference, measurement, and mask export.

This is the production entry point used by the API. It reads exactly one SAR
band, preserves source georeferencing, delegates model work to
:mod:`src.operating`, zeros invalid probabilities, and attaches connected-region
measurements. Rendering and post-hoc diagnostics happen later in the backend.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.io import DatasetReader, MemoryFile
from rasterio.transform import Affine

import config
from src import models
from src.measure import (
    bounds_lonlat,
    measure_regions,
    overall_alert_and_confidence,
    postprocess_probabilities,
)
from src.tiling import run_tiled_inference, valid_pixel_mask


@dataclass(frozen=True)
class PipelineResult:
    """Immutable inference and measurement outputs for one source scene.

    ``full_mask`` is the learned ``float32[height,width]`` probability mosaic;
    ``binary`` is the final boolean detection mask. Both share ``crs`` and
    ``transform`` with the input GeoTIFF. ``regions_df`` uses the stable schema
    defined by :data:`src.measure.REGION_COLUMNS`.
    """

    full_mask: np.ndarray
    binary: np.ndarray
    regions_df: pd.DataFrame
    rejected_tiles: list[dict[str, object]]
    total_area_km2: float
    n_regions: int
    overall_severity: str
    overall_confidence: str
    crs: CRS
    transform: Affine
    scene_bounds_lonlat: tuple[float, float, float, float]
    norm_scope_used: str


@contextmanager
def _open_source(
    source: str | PathLike[str] | MemoryFile | DatasetReader,
) -> Iterator[DatasetReader]:
    """Open paths/MemoryFiles while leaving caller-owned datasets open."""

    if isinstance(source, (str, PathLike)):
        with rasterio.open(source) as dataset:
            yield dataset
        return
    if isinstance(source, MemoryFile):
        with source.open() as dataset:
            yield dataset
        return
    if hasattr(source, "read") and hasattr(source, "transform"):
        yield source
        return
    raise TypeError("Expected a GeoTIFF path, rasterio MemoryFile, or open dataset.")


def run_pipeline(
    src_path_or_memfile: str | PathLike[str] | MemoryFile | DatasetReader,
    cfg: object = config,
) -> PipelineResult:
    """Run the complete released analysis for one single-band SAR GeoTIFF.

    Args:
        src_path_or_memfile: Filesystem path, rasterio ``MemoryFile``, or open
            dataset. Caller-owned datasets are not closed.
        cfg: Configuration object; production uses :mod:`config`.

    Returns:
        :class:`PipelineResult` on the exact source grid.

    Raises:
        TypeError: If the source is not a supported raster handle.
        ValueError: If the raster is not single-band, has no CRS, or contains
            invalid geometry/normalization inputs.
        FileNotFoundError: If a configured checkpoint is missing.
        RuntimeError: If the required CUDA-bf16 configuration is unavailable.
    """

    with _open_source(src_path_or_memfile) as src:
        if src.count != 1:
            raise ValueError(
                "Input GeoTIFF must contain exactly one SAR band; "
                f"found {src.count}."
            )
        if src.crs is None:
            raise ValueError("Input GeoTIFF must define a CRS.")
        image = src.read(1).astype(np.float32, copy=False)
        nodata = src.nodata
        crs = src.crs
        affine = src.transform
        width = src.width
        height = src.height

    if bool(getattr(cfg, "USE_TRAINED_OPERATING_MODELS", False)):
        from src.operating import run_operating_inference

        operating = run_operating_inference(image, nodata, cfg)
        full_mask = operating.probabilities.copy()
        binary = operating.binary.copy()
        rejected_tiles = operating.rejected_tiles
    else:
        tiled = run_tiled_inference(
            image,
            screen_fn=models.screen,
            segment_fn=models.segment,
            norm_scope=str(cfg.NORM_SCOPE),
            nodata=nodata,
            tile_size=int(cfg.TILE_SIZE),
            stride=int(cfg.STRIDE),
            screener_threshold=float(cfg.SCREENER_THRESHOLD),
            norm_lo_pct=float(cfg.NORM_LO_PCT),
            norm_hi_pct=float(cfg.NORM_HI_PCT),
            fixed_db_lo=float(cfg.FIXED_DB_LO),
            fixed_db_hi=float(cfg.FIXED_DB_HI),
            max_invalid_fraction=float(cfg.MAX_INVALID_FRACTION),
            infill_invalid=bool(cfg.INFILL_INVALID_PIXELS),
            exclude_boundary_adjacent=bool(cfg.EXCLUDE_BOUNDARY_ADJACENT_TILES),
        )
        full_mask = tiled.probabilities.copy()
        binary = postprocess_probabilities(full_mask, cfg)
        rejected_tiles = tiled.rejected_tiles
    invalid = ~valid_pixel_mask(image, nodata)
    full_mask[invalid] = 0.0
    regions = measure_regions(
        full_mask,
        binary,
        affine,
        crs,
        coastline_path=cfg.COASTLINE_PATH,
        coastline_search_buffer_deg=float(
            getattr(cfg, "COASTLINE_SEARCH_BUFFER_DEG", 2.0)
        ),
    )
    total_area = float(regions["area_km2"].sum()) if not regions.empty else 0.0
    scene_alert, scene_confidence = overall_alert_and_confidence(regions, cfg)
    return PipelineResult(
        full_mask=full_mask,
        binary=binary,
        regions_df=regions,
        rejected_tiles=rejected_tiles,
        total_area_km2=total_area,
        n_regions=len(regions),
        overall_severity=scene_alert,
        overall_confidence=scene_confidence,
        crs=crs,
        transform=affine,
        scene_bounds_lonlat=bounds_lonlat(width, height, affine, crs),
        norm_scope_used=str(cfg.NORM_SCOPE),
    )


def save_mask_geotiff(result: PipelineResult, path: str | PathLike[str]) -> None:
    """Write a QGIS-ready binary mask with source grid and CRS unchanged.

    Args:
        result: Completed pipeline output.
        path: Destination path; parent directories are created as needed.

    Returns:
        ``None``. The GeoTIFF contains one uint8 band where 1 is detected and 0
        is background, with Deflate compression and normalization metadata.

    Raises:
        RasterioError: If GDAL cannot create or write the destination.
        OSError: If the destination directory cannot be created.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        destination,
        "w",
        driver="GTiff",
        height=result.binary.shape[0],
        width=result.binary.shape[1],
        count=1,
        dtype="uint8",
        crs=result.crs,
        transform=result.transform,
        compress="deflate",
    ) as dst:
        dst.write(result.binary.astype(np.uint8), 1)
        dst.update_tags(
            description="Binary oil-spill mask",
            normalization_scope=result.norm_scope_used,
        )

"""Compute the post-hoc undetected-dark diagnostic used by the web UI.

The diagnostic runs only after :func:`src.pipeline.run_pipeline` has finalized
the production mask. It finds cleaned dark pixels outside that mask and reports
the subset rejected by Stage 1. Its label raster is presentation-only and is
kept separate so requesting a diagnostic layer cannot alter detection outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.measure import label
from skimage.morphology import binary_opening, disk, remove_small_objects

import config
from src.measure import mask_area_m2
from src.normalize import apply_norm, fixed_db_norm, scene_stats
from src.tiling import valid_pixel_mask

REJECTED_BY_STAGE1 = 1


@dataclass(frozen=True)
class UndetectedDarkResult:
    """Immutable diagnostic label raster and its presentation-ready summary.

    ``labels`` is a ``uint8[height,width]`` array where zero means unlabelled and
    :data:`REJECTED_BY_STAGE1` identifies retained dark pixels outside the final
    screener footprint.
    """

    labels: np.ndarray
    total_area_km2: float
    region_count: int
    categories: dict[str, dict[str, Any]]

    def category_mask(self, code: int) -> np.ndarray:
        """Return a boolean view selecting one diagnostic label code.

        Args:
            code: Integer label value to select.

        Returns:
            ``bool[height,width]`` mask aligned with the source scene.
        """

        return self.labels == int(code)

    def response(self) -> dict[str, Any]:
        """Serialize summary values for the public analysis API.

        Returns:
            JSON-compatible totals, category measurements, and the cleanup
            constants needed to interpret the diagnostic.
        """

        return {
            "total_area_km2": self.total_area_km2,
            "region_count": self.region_count,
            "categories": self.categories,
            "baseline_threshold": float(config.BASELINE_THRESHOLD),
            "opening_radius": int(config.MORPH_OPENING_RADIUS),
            "minimum_blob_px": int(config.MIN_BLOB_PX),
        }


def _normalize_scene(
    image: np.ndarray,
    nodata: float | None,
    cfg: object,
) -> np.ndarray:
    scope = str(cfg.NORM_SCOPE)
    if scope == "scene":
        lo, hi = scene_stats(
            image,
            lo_pct=float(cfg.NORM_LO_PCT),
            hi_pct=float(cfg.NORM_HI_PCT),
            nodata=nodata,
        )
        return apply_norm(image, lo, hi)
    if scope == "fixed_db":
        return fixed_db_norm(
            image,
            lo_db=float(cfg.FIXED_DB_LO),
            hi_db=float(cfg.FIXED_DB_HI),
        )
    raise ValueError(
        "Whole-scene undetected-dark analysis requires scene or fixed_db "
        f"normalization; received {scope!r}."
    )


def _coverage_for_reason(
    shape: tuple[int, int],
    rejected_tiles: list[dict[str, object]],
    reason: str,
    tile_size: int,
) -> np.ndarray:
    height, width = shape
    coverage = np.zeros(shape, dtype=bool)
    for tile in rejected_tiles:
        if tile.get("reason") != reason:
            continue
        x = int(tile["x"])
        y = int(tile["y"])
        coverage[y : min(y + tile_size, height), x : min(x + tile_size, width)] = True
    return coverage


def _validate_attribution(
    detected: np.ndarray,
    undetected: np.ndarray,
    rejected_by_stage1: np.ndarray,
    residual: np.ndarray,
) -> None:
    """Fail unless the retained subset and residual exhaust the cleaned layer."""

    for category in (rejected_by_stage1, residual):
        if category.shape != undetected.shape:
            raise RuntimeError("Undetected-dark accounting masks do not align.")
    overlap_pixels = int(np.count_nonzero(rejected_by_stage1 & residual))
    if overlap_pixels:
        raise RuntimeError(
            "Undetected-dark Stage 1 subset and residual overlap at "
            f"{overlap_pixels} pixels."
        )
    accounted_dark = rejected_by_stage1 | residual
    if not np.array_equal(accounted_dark, undetected):
        missing = int(np.count_nonzero(undetected & ~accounted_dark))
        extra = int(np.count_nonzero(accounted_dark & ~undetected))
        raise RuntimeError(
            "Undetected-dark accounting does not exhaust the cleaned layer: "
            f"{missing} missing and {extra} extra pixels."
        )
    if np.any(np.asarray(detected, dtype=bool) & accounted_dark):
        raise RuntimeError("Detected and undetected-dark pixels overlap.")
    # Opening/blob removal intentionally excludes raw threshold speckle. Within
    # the post-cleanup accounting universe, detection plus all retained dark
    # pixels must account for every pixel exactly.
    accounted = np.asarray(detected, dtype=bool) | accounted_dark
    accounting_universe = np.asarray(detected, dtype=bool) | undetected
    if not np.array_equal(accounted, accounting_universe):
        raise RuntimeError(
            "Detection and undetected-dark pixels do not cover the complete "
            "post-cleanup dark accounting universe."
        )


def compute_undetected_dark(
    image: np.ndarray,
    nodata: float | None,
    pipeline_result: object,
    cfg: object = config,
) -> UndetectedDarkResult:
    """Compute and attribute cleaned dark pixels after detection is complete.

    Args:
        image: Raw single-band ``float32[height,width]`` SAR backscatter.
        nodata: Declared nodata value, or ``None``.
        pipeline_result: Completed :class:`src.pipeline.PipelineResult` aligned
            with ``image``.
        cfg: Configuration object holding normalization and cleanup settings.

    Returns:
        Immutable label raster plus total and Stage-1-rejected measurements.

    Raises:
        ValueError: If source and result masks are not aligned or whole-scene
            normalization is unavailable.
        RuntimeError: If accounting masks overlap or fail to exhaust the
            cleaned diagnostic layer.

    The function treats every pipeline output as read-only. Its compact uint8
    label map is separate from the production binary mask, probability raster,
    measurements, alert level, or detection confidence.
    """

    scene = np.asarray(image, dtype=np.float32)
    detected = np.asarray(pipeline_result.binary, dtype=bool)
    if scene.ndim != 2 or scene.shape != detected.shape:
        raise ValueError("Source image and production detection mask must align.")

    valid = valid_pixel_mask(scene, nodata)
    normalized = _normalize_scene(scene, nodata, cfg)
    undetected = (normalized < float(cfg.BASELINE_THRESHOLD)) & valid & ~detected
    undetected = binary_opening(
        undetected,
        footprint=disk(int(cfg.MORPH_OPENING_RADIUS)),
    )
    undetected = remove_small_objects(
        undetected,
        min_size=int(cfg.MIN_BLOB_PX),
        connectivity=int(cfg.BLOB_CONNECTIVITY),
    ).astype(bool, copy=False)

    rejected_tiles = list(pipeline_result.rejected_tiles)
    stage1_coverage = _coverage_for_reason(
        scene.shape,
        rejected_tiles,
        "oil_probability_below_threshold",
        int(cfg.TILE_SIZE),
    )
    # Positive stitched probabilities are exactly the final pass footprint:
    # only final-passed segmenter tiles contribute to the stitched raster.
    pass_footprint = (np.asarray(pipeline_result.full_mask) > 0.0) & valid

    # A pixel is Stage 1-rejected only if it lies outside the final footprint.
    # This correctly treats closing-added tiles as passed and handles overlap
    # between adjacent 256-pixel tiles without contradictory categories.
    rejected_by_stage1 = undetected & stage1_coverage & ~pass_footprint
    residual = undetected & ~rejected_by_stage1
    _validate_attribution(detected, undetected, rejected_by_stage1, residual)

    labels = np.zeros(scene.shape, dtype=np.uint8)
    labels[rejected_by_stage1] = REJECTED_BY_STAGE1

    total_pixels = int(np.count_nonzero(undetected))
    rejected_pixels = int(np.count_nonzero(rejected_by_stage1))
    categories: dict[str, dict[str, Any]] = {
        "rejected_by_stage1": {
            "label": "Rejected by Stage 1",
            "area_km2": float(
                mask_area_m2(
                    rejected_by_stage1,
                    pipeline_result.transform,
                    pipeline_result.crs,
                )
                / 1_000_000.0
            ),
            "percentage": (
                float(100.0 * rejected_pixels / total_pixels) if total_pixels else 0.0
            ),
            "region_count": int(
                label(
                    rejected_by_stage1,
                    connectivity=int(cfg.BLOB_CONNECTIVITY),
                ).max()
            ),
        }
    }

    labels.setflags(write=False)
    return UndetectedDarkResult(
        labels=labels,
        total_area_km2=float(
            mask_area_m2(
                undetected,
                pipeline_result.transform,
                pipeline_result.crs,
            )
            / 1_000_000.0
        ),
        region_count=int(
            label(undetected, connectivity=int(cfg.BLOB_CONNECTIVITY)).max()
        ),
        categories=categories,
    )

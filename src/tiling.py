"""Generate deterministic overlapping tiles and support stitched inference.

The released learned path reuses this module's grid, invalid-pixel, and infill
helpers. :func:`run_tiled_inference` remains as a deterministic model-interface
fallback. Row-major ordering and the forced far-edge tile are part of numerical
reproducibility and must remain stable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import config
from src import models
from src.normalize import normalize_tile, scene_stats

ScreenFn = Callable[[np.ndarray], tuple[int, np.ndarray]]
SegmentFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TilingResult:
    """Stitched fallback probabilities and tile rejection diagnostics.

    ``probabilities`` and ``counts`` are two-dimensional arrays on the original
    scene grid. ``rejected_tiles`` records pixel origins and an actionable reason
    for every tile omitted from stitching.
    """

    probabilities: np.ndarray
    counts: np.ndarray
    rejected_tiles: list[dict[str, object]]


def tile_starts(
    length: int,
    tile_size: int = config.TILE_SIZE,
    stride: int = config.STRIDE,
) -> list[int]:
    """Return starts that always include the far edge.

    Args:
        length: Scene dimension in pixels.
        tile_size: Model tile edge in pixels.
        stride: Distance in pixels between regular starts.

    Returns:
        Sorted unique zero-based starts including ``length - tile_size``.

    Raises:
        ValueError: If dimensions are non-positive or stride exceeds tile size.

    A plain ``range`` silently drops the final strip when the dimension is not an
    exact tile/stride fit. Appending ``length - tile_size`` closes that gap; set
    deduplication handles exact fits without double-counting the last tile.
    """

    if length <= 0:
        raise ValueError("Scene dimensions must be positive.")
    if tile_size <= 0 or stride <= 0 or stride > tile_size:
        raise ValueError("Require tile_size > 0 and 0 < stride <= tile_size.")
    if length <= tile_size:
        return [0]

    final_start = length - tile_size
    starts = list(range(0, final_start + 1, stride))
    starts.append(final_start)
    return sorted(set(starts))


def pad_to_tile(img: np.ndarray, tile_size: int) -> np.ndarray:
    """Edge-pad only undersized dimensions to one fixed-size model tile.

    Args:
        img: Two-dimensional scene array.
        tile_size: Required minimum height and width in pixels.

    Returns:
        Original array when already large enough; otherwise an edge-padded array.
    """

    pad_h = max(0, tile_size - img.shape[0])
    pad_w = max(0, tile_size - img.shape[1])
    if pad_h == 0 and pad_w == 0:
        return img
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")


def invalid_pixel_fraction(tile: np.ndarray, nodata: float | None) -> float:
    """Return the non-finite or declared-nodata fraction of a raw tile.

    The result lies in ``[0,1]``. Production rejects values strictly greater
    than 0.20, so exactly 20% remains eligible as it did in training.
    """

    return 1.0 - float(valid_pixel_mask(tile, nodata).mean())


def valid_pixel_mask(array: np.ndarray, nodata: float | None) -> np.ndarray:
    """Identify finite pixels that differ from an optional nodata sentinel.

    Args:
        array: Numeric array of any shape.
        nodata: Optional declared nodata value. A non-finite nodata declaration
            needs no equality comparison because non-finite pixels are already
            invalid.

    Returns:
        Boolean array with the same shape.
    """

    valid = np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        valid &= array != nodata
    return valid


def scene_valid_median(scene: np.ndarray, nodata: float | None) -> float:
    """Return the exact median over finite, non-nodata scene pixels.

    Raises:
        ValueError: If the scene contains no valid pixel to use for infill.
    """

    valid = valid_pixel_mask(scene, nodata)
    if not valid.any():
        raise ValueError("Cannot infill a scene with no valid pixels.")
    return float(np.median(scene[valid]))


def infill_invalid_pixels(
    tile: np.ndarray, nodata: float | None, fill_value: float
) -> np.ndarray:
    """Replace invalid pixels with a finite scene value without mutating input.

    Returning the original array when no replacement is needed avoids a copy.
    Median infill keeps eligible nodata from becoming normalized zero, which the
    trained screener reads as maximally dark oil-like backscatter.
    """

    invalid = ~valid_pixel_mask(tile, nodata)
    if not invalid.any():
        return tile
    filled = tile.copy()
    filled[invalid] = fill_value
    return filled


def _invalid_neighbourhood(scene: np.ndarray, nodata: float | None) -> np.ndarray:
    """Return invalid pixels plus their one-pixel, eight-connected dilation."""

    invalid = ~valid_pixel_mask(scene, nodata)
    neighbourhood = invalid.copy()
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0:
                continue
            source_rows = slice(
                max(0, -row_offset), scene.shape[0] - max(0, row_offset)
            )
            source_cols = slice(
                max(0, -col_offset), scene.shape[1] - max(0, col_offset)
            )
            target_rows = slice(
                max(0, row_offset), scene.shape[0] - max(0, -row_offset)
            )
            target_cols = slice(
                max(0, col_offset), scene.shape[1] - max(0, -col_offset)
            )
            neighbourhood[target_rows, target_cols] |= invalid[source_rows, source_cols]
    return neighbourhood


def run_tiled_inference(
    img: np.ndarray,
    *,
    screen_fn: ScreenFn = models.screen,
    segment_fn: SegmentFn = models.segment,
    norm_scope: str = config.NORM_SCOPE,
    nodata: float | None = None,
    tile_size: int = config.TILE_SIZE,
    stride: int = config.STRIDE,
    screener_threshold: float = config.SCREENER_THRESHOLD,
    norm_lo_pct: float = config.NORM_LO_PCT,
    norm_hi_pct: float = config.NORM_HI_PCT,
    fixed_db_lo: float = config.FIXED_DB_LO,
    fixed_db_hi: float = config.FIXED_DB_HI,
    max_invalid_fraction: float = config.MAX_INVALID_FRACTION,
    infill_invalid: bool = config.INFILL_INVALID_PIXELS,
    exclude_boundary_adjacent: bool = config.EXCLUDE_BOUNDARY_ADJACENT_TILES,
) -> TilingResult:
    """Run the fixed model interface and stitch with per-pixel overlap averaging.

    Args:
        img: Two-dimensional raw SAR array.
        screen_fn: Callable returning ``(class_index, float32[3])``.
        segment_fn: Callable returning ``float32[tile_size,tile_size]``.
        norm_scope: ``scene``, ``tile``, or ``fixed_db``.
        nodata: Optional declared source nodata value.
        tile_size: Square model input edge in pixels.
        stride: Sliding-window stride in pixels.
        screener_threshold: Oil probability required to call ``segment_fn``.
        norm_lo_pct: Lower percentile for percentile normalization.
        norm_hi_pct: Upper percentile for percentile normalization.
        fixed_db_lo: Lower fixed-dB normalization bound.
        fixed_db_hi: Upper fixed-dB normalization bound.
        max_invalid_fraction: Inclusive eligibility ceiling in ``[0,1]``.
        infill_invalid: Replace invalid values in eligible tiles with scene median.
        exclude_boundary_adjacent: Enable the legacy diagnostic ring exclusion.

    Returns:
        :class:`TilingResult` with overlap averages and rejection records.

    Raises:
        ValueError: If scene geometry, configuration, or model outputs are invalid.

    Tiles exceeding the same invalid-pixel fraction used to build the training
    store are rejected before normalization or model screening. This prevents
    nodata values from being clipped to maximally dark normalized inputs.
    Otherwise-eligible tiles can instead replace invalid pixels with the exact
    valid-scene median before normalization. The older quantified boundary rule
    remains available as an opt-in diagnostic mode.
    """

    scene = np.asarray(img)
    if scene.ndim != 2:
        raise ValueError(f"Expected a two-dimensional scene, received {scene.shape}.")
    if scene.shape[0] == 0 or scene.shape[1] == 0:
        raise ValueError("Scene dimensions must be positive.")
    if not 0.0 <= max_invalid_fraction <= 1.0:
        raise ValueError("max_invalid_fraction must lie in [0, 1].")

    original_shape = scene.shape
    padded = pad_to_tile(scene, tile_size)
    invalid_neighbourhood = (
        _invalid_neighbourhood(scene, nodata) if exclude_boundary_adjacent else None
    )
    scene_range = None
    if norm_scope == "scene":
        scene_range = scene_stats(
            scene,
            lo_pct=norm_lo_pct,
            hi_pct=norm_hi_pct,
            nodata=nodata,
        )
    scene_median = scene_valid_median(scene, nodata) if infill_invalid else None

    full = np.zeros(padded.shape, dtype=np.float32)
    counts = np.zeros(padded.shape, dtype=np.float32)
    rejected: list[dict[str, object]] = []

    row_starts = tile_starts(padded.shape[0], tile_size, stride)
    col_starts = tile_starts(padded.shape[1], tile_size, stride)
    for y in row_starts:
        for x in col_starts:
            raw_tile = padded[y : y + tile_size, x : x + tile_size]
            invalid_fraction = invalid_pixel_fraction(raw_tile, nodata)
            if invalid_fraction > max_invalid_fraction:
                rejected.append(
                    {
                        "x": x,
                        "y": y,
                        "reason": "invalid_pixel_fraction_above_threshold",
                        "invalid_pixel_fraction": invalid_fraction,
                        "max_invalid_fraction": max_invalid_fraction,
                    }
                )
                continue
            touches_scene_boundary = (
                x == 0
                or y == 0
                or x + tile_size >= original_shape[1]
                or y + tile_size >= original_shape[0]
            )
            adjacent_to_nodata = (
                bool(
                    invalid_neighbourhood[
                        y : min(y + tile_size, original_shape[0]),
                        x : min(x + tile_size, original_shape[1]),
                    ].any()
                )
                if invalid_neighbourhood is not None
                else False
            )
            if exclude_boundary_adjacent and (
                touches_scene_boundary or adjacent_to_nodata
            ):
                rejected.append(
                    {
                        "x": x,
                        "y": y,
                        "reason": "boundary_or_nodata_adjacent",
                        "touches_scene_boundary": touches_scene_boundary,
                        "adjacent_to_nodata": adjacent_to_nodata,
                    }
                )
                continue
            model_tile = (
                infill_invalid_pixels(raw_tile, nodata, scene_median)
                if scene_median is not None
                else raw_tile
            )
            normalized = normalize_tile(
                model_tile,
                norm_scope,
                scene_range=scene_range,
                nodata=nodata,
                lo_pct=norm_lo_pct,
                hi_pct=norm_hi_pct,
                fixed_lo_db=fixed_db_lo,
                fixed_hi_db=fixed_db_hi,
            )
            cls, probabilities = screen_fn(normalized)
            probabilities = np.asarray(probabilities, dtype=np.float32)
            if probabilities.shape != (3,):
                raise ValueError("screen() probabilities must have shape (3,).")
            if float(probabilities[0]) < screener_threshold:
                rejected.append(
                    {
                        "x": x,
                        "y": y,
                        "reason": "oil_probability_below_threshold",
                        "oil_probability": float(probabilities[0]),
                        "class_index": int(cls),
                    }
                )
                continue

            prediction = np.asarray(segment_fn(normalized), dtype=np.float32)
            if prediction.shape != (tile_size, tile_size):
                raise ValueError(
                    "segment() must return the same fixed spatial shape as its input."
                )
            if not np.all(np.isfinite(prediction)):
                raise ValueError("segment() returned non-finite probabilities.")
            if np.any((prediction < 0.0) | (prediction > 1.0)):
                raise ValueError("segment() probabilities must lie in [0, 1].")

            full[y : y + tile_size, x : x + tile_size] += prediction
            counts[y : y + tile_size, x : x + tile_size] += 1.0

    stitched = full / np.maximum(counts, 1.0)
    height, width = original_shape
    return TilingResult(
        probabilities=stitched[:height, :width].astype(np.float32, copy=False),
        counts=counts[:height, :width],
        rejected_tiles=rejected,
    )

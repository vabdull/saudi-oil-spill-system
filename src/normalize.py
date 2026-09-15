"""Map raw single-band SAR backscatter to the ``[0,1]`` model input range.

The operating path uses one 2nd–98th percentile range for the entire scene.
Per-tile stretching is retained only for diagnostics: it can make a uniformly
dark slick interior span the full range and therefore resemble clean water.
"""

from __future__ import annotations

import numpy as np

import config


def _valid_values(img: np.ndarray, nodata: float | None = None) -> np.ndarray:
    """Return finite, non-nodata values without changing the input array."""

    values = np.asarray(img).reshape(-1)
    valid = np.isfinite(values)
    if nodata is not None and np.isfinite(nodata):
        valid &= values != nodata
    return values[valid]


def scene_stats(
    img: np.ndarray,
    lo_pct: float = config.NORM_LO_PCT,
    hi_pct: float = config.NORM_HI_PCT,
    nodata: float | None = None,
) -> tuple[float, float]:
    """Compute robust scene percentiles, ignoring NaN and nodata values.

    Args:
        img: Numeric scene array of any shape.
        lo_pct: Lower percentile in ``[0,100)``.
        hi_pct: Upper percentile above ``lo_pct`` and at most 100.
        nodata: Optional finite value to exclude.

    Returns:
        ``(lower, upper)`` as Python floats.

    Raises:
        ValueError: If percentiles are invalid or no valid pixel remains.

    A deterministic stride sample bounds memory and percentile cost for very
    large rasters while retaining coverage across the entire flattened scene.
    """

    if not 0.0 <= lo_pct < hi_pct <= 100.0:
        raise ValueError("Expected percentiles satisfying 0 <= lo < hi <= 100.")

    flattened = np.asarray(img).reshape(-1)
    if flattened.size > config.NORM_MAX_SAMPLES:
        step = int(np.ceil(flattened.size / config.NORM_MAX_SAMPLES))
        flattened = flattened[::step]
    values = _valid_values(flattened, nodata)
    if values.size == 0:
        raise ValueError("Cannot normalize a scene containing no valid pixels.")

    lo, hi = np.percentile(values, (lo_pct, hi_pct))
    return float(lo), float(hi)


def apply_norm(tile: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linearly map supplied bounds to a clipped float32 tile.

    Invalid input values become zero. A non-finite or collapsed range returns
    all zeros rather than dividing by zero, preserving shape and dtype.
    """

    tile_array = np.asarray(tile, dtype=np.float32)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(tile_array.shape, dtype=np.float32)
    normalized = np.clip((tile_array - lo) / (hi - lo), 0.0, 1.0)
    normalized[~np.isfinite(tile_array)] = 0.0
    return normalized.astype(np.float32, copy=False)


def fixed_db_norm(
    tile: np.ndarray,
    lo_db: float = config.FIXED_DB_LO,
    hi_db: float = config.FIXED_DB_HI,
) -> np.ndarray:
    """Normalize calibrated dB imagery with fixed physical bounds.

    Args:
        tile: Numeric SAR tile.
        lo_db: Backscatter mapped to zero, in decibels.
        hi_db: Backscatter mapped to one, in decibels; must exceed ``lo_db``.

    Returns:
        Clipped ``float32`` array with the input shape.
    """

    return apply_norm(tile, lo_db, hi_db)


def normalize_tile(
    tile: np.ndarray,
    scope: str,
    *,
    scene_range: tuple[float, float] | None = None,
    nodata: float | None = None,
    lo_pct: float = config.NORM_LO_PCT,
    hi_pct: float = config.NORM_HI_PCT,
    fixed_lo_db: float = config.FIXED_DB_LO,
    fixed_hi_db: float = config.FIXED_DB_HI,
) -> np.ndarray:
    """Normalize one tile using an explicit scene, tile, or fixed-dB strategy.

    Args:
        tile: Raw numeric tile.
        scope: ``scene``, ``tile``, or ``fixed_db``.
        scene_range: Required ``(lo,hi)`` pair for scene scope.
        nodata: Optional value ignored when deriving tile percentiles.
        lo_pct: Tile-scope lower percentile.
        hi_pct: Tile-scope upper percentile.
        fixed_lo_db: Fixed-dB lower bound.
        fixed_hi_db: Fixed-dB upper bound.

    Returns:
        Normalized ``float32`` array with the input shape.

    Raises:
        ValueError: If the scope is unknown, a scene range is missing, or tile
            percentiles cannot be derived.
    """

    if scope == "scene":
        if scene_range is None:
            raise ValueError("scene_range is required when scope='scene'.")
        return apply_norm(tile, *scene_range)
    if scope == "tile":
        lo, hi = scene_stats(tile, lo_pct=lo_pct, hi_pct=hi_pct, nodata=nodata)
        return apply_norm(tile, lo, hi)
    if scope == "fixed_db":
        return fixed_db_norm(tile, fixed_lo_db, fixed_hi_db)
    raise ValueError("NORM_SCOPE must be one of: 'scene', 'tile', 'fixed_db'.")

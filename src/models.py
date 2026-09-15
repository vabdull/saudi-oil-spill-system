"""Define deterministic tile-level baselines shared with production inference.

The learned networks live in :mod:`src.operating`. The operating chain calls
:func:`segment` for its classical dark-pixel union, while :func:`screen` and
:func:`segment_passthrough` provide deterministic fallbacks for isolated tiling
and stitching checks.
"""

from __future__ import annotations

import numpy as np

import config


def _validate_tile(tile: np.ndarray) -> np.ndarray:
    """Validate the fixed Week 3 interface before a model sees a tile."""

    array = np.asarray(tile)
    expected = (config.TILE_SIZE, config.TILE_SIZE)
    if array.shape != expected:
        raise ValueError(f"Expected tile shape {expected}, received {array.shape}.")
    if array.dtype != np.float32:
        raise TypeError(f"Expected float32 tile, received {array.dtype}.")
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("Model tiles must contain finite values in [0, 1].")
    return array


def screen(tile: np.ndarray) -> tuple[int, np.ndarray]:
    """Pass every normalized tile through using the fixed three-class interface.

    Args:
        tile: Normalized ``float32[256,256]`` array in ``[0,1]``.

    Returns:
        Oil class index ``0`` and ``float32[3]`` probabilities ``[1,0,0]``.

    Raises:
        TypeError: If ``tile`` is not ``float32``.
        ValueError: If shape, finiteness, or range is invalid.

    Class 0 is oil, class 1 is look-alike, and class 2 is clean. The placeholder
    intentionally accepts everything so the full downstream pipeline is exercised.
    """

    _validate_tile(tile)
    return 0, np.array([1.0, 0.0, 0.0], dtype=np.float32)


def segment(tile: np.ndarray) -> np.ndarray:
    """Return the classical normalized-intensity mask used by the final union.

    Args:
        tile: Normalized ``float32[256,256]`` array in ``[0,1]``.

    Returns:
        ``float32[256,256]`` values of zero or one, with one below the configured
        baseline threshold. Production intersects this mask with the final
        screener footprint because the classical rule alone accepts dark
        look-alikes; Stage 1 is its containment boundary.
    """

    array = _validate_tile(tile)
    return (array < config.BASELINE_THRESHOLD).astype(np.float32)


def segment_passthrough(tile: np.ndarray) -> np.ndarray:
    """Return a validated tile copy for deterministic stitching checks.

    Args:
        tile: Normalized ``float32[256,256]`` array in ``[0,1]``.

    Returns:
        Independent copy with identical values and shape.
    """

    return _validate_tile(tile).copy()

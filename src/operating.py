"""Execute the released two-pass screener/segmenter operating configuration.

Pass 1 screens every tile that satisfies the invalid-pixel rule and expands the
binary pass map with closing and hole filling. Pass 2 segments only that final
set, preserves historical CUDA batch shapes, overlap-averages predictions, and
adds the classical dark-pixel mask only inside the pass footprint. This module
is the only production code that runs neural-network inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import segmentation_models_pytorch as smp
import timm
import torch
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.morphology import binary_opening, disk, remove_small_objects

import config
from src import models
from src.normalize import normalize_tile, scene_stats
from src.tiling import (
    infill_invalid_pixels,
    invalid_pixel_fraction,
    pad_to_tile,
    scene_valid_median,
    tile_starts,
    valid_pixel_mask,
)


@dataclass(frozen=True)
class OperatingInferenceResult:
    """Arrays and tile-accounting diagnostics returned by learned inference.

    ``probabilities`` is ``float32[height,width]`` and ``binary`` and
    ``pass_footprint`` are boolean arrays on the input scene grid. ``counts`` is
    ``uint16[height,width]`` and records how many final-pass segmenter tiles
    contributed to each stitched pixel.
    """

    probabilities: np.ndarray
    binary: np.ndarray
    counts: np.ndarray
    pass_footprint: np.ndarray
    rejected_tiles: list[dict[str, object]]
    base_passed_tiles: int
    closing_added_tiles: int
    final_passed_tiles: int


def _seed() -> None:
    seed = int(config.RANDOM_SEED)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(cfg: object) -> torch.device:
    require_bf16 = bool(getattr(cfg, "OPERATING_REQUIRE_CUDA_BF16", True))
    if require_bf16 and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "The released operating path requires a CUDA device with bf16 support. "
            "Install the CUDA requirements and run on a compatible NVIDIA GPU; "
            "CPU fp32 is diagnostic-only and does not reproduce published values."
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=4)
def _load_models(
    screener_checkpoint_path: str,
    segmenter_checkpoint_path: str,
    device_name: str,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load and cache the trained model pair once per device and checkpoint path.

    The screener is timm ResNet-18 with one input channel and three logits. The
    segmenter is a segmentation-models-pytorch U-Net with a ResNet-34 encoder,
    one input channel, and one output logit. Epoch guards prevent silently using
    a similarly named but non-operating checkpoint.
    """

    device = torch.device(device_name)
    screener_checkpoint = torch.load(
        screener_checkpoint_path, map_location="cpu", weights_only=False
    )
    segmenter_checkpoint = torch.load(
        segmenter_checkpoint_path, map_location="cpu", weights_only=False
    )
    if int(screener_checkpoint.get("epoch", -1)) != 17:
        raise RuntimeError("Operating screener checkpoint must be epoch 17.")
    if int(segmenter_checkpoint.get("epoch", -1)) != 14:
        raise RuntimeError("Operating segmenter checkpoint must be epoch 14.")
    screener = timm.create_model(
        "resnet18", pretrained=False, in_chans=1, num_classes=3
    )
    screener.load_state_dict(screener_checkpoint["model_state"])
    segmenter = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )
    segmenter.load_state_dict(segmenter_checkpoint["model_state"])
    return screener.to(device).eval(), segmenter.to(device).eval()


def _normalized_batch(
    image: np.ndarray,
    locations: list[tuple[int, int]],
    tile_size: int,
    nodata: float | None,
    scene_range: tuple[float, float] | None,
    scene_median: float | None,
    cfg: object,
) -> np.ndarray:
    arrays = []
    for x, y in locations:
        raw = image[y : y + tile_size, x : x + tile_size]
        model_tile = (
            infill_invalid_pixels(raw, nodata, scene_median)
            if scene_median is not None
            else raw
        )
        arrays.append(
            normalize_tile(
                model_tile,
                str(cfg.NORM_SCOPE),
                scene_range=scene_range,
                nodata=nodata,
                lo_pct=float(cfg.NORM_LO_PCT),
                hi_pct=float(cfg.NORM_HI_PCT),
                fixed_lo_db=float(cfg.FIXED_DB_LO),
                fixed_hi_db=float(cfg.FIXED_DB_HI),
            )
        )
    return np.stack(arrays)


def _partition_tile_locations(
    image: np.ndarray,
    locations: list[tuple[int, int]],
    tile_size: int,
    nodata: float | None,
    max_invalid_fraction: float,
) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    """Split production tiles into eligible and F1-rejected locations."""

    eligible: list[tuple[int, int]] = []
    rejected: list[dict[str, object]] = []
    for x, y in locations:
        raw = image[y : y + tile_size, x : x + tile_size]
        fraction = invalid_pixel_fraction(raw, nodata)
        if fraction > max_invalid_fraction:
            rejected.append(
                {
                    "x": x,
                    "y": y,
                    "reason": "invalid_pixel_fraction_above_threshold",
                    "invalid_pixel_fraction": fraction,
                    "max_invalid_fraction": max_invalid_fraction,
                }
            )
        else:
            eligible.append((x, y))
    return eligible, rejected


def _expand_pass_grid(
    passed_grid: np.ndarray,
    eligible_grid: np.ndarray,
    *,
    closing: bool,
    fill_holes: bool,
) -> np.ndarray:
    """Apply the configured tile-map morphology without admitting ineligible tiles."""

    passed = np.asarray(passed_grid, dtype=bool)
    eligible = np.asarray(eligible_grid, dtype=bool)
    if passed.shape != eligible.shape:
        raise ValueError("passed_grid and eligible_grid must have the same shape.")
    expanded = passed.copy()
    if closing:
        expanded = binary_closing(
            expanded,
            structure=np.ones((3, 3), dtype=bool),
            iterations=1,
            border_value=0,
        )
    if fill_holes:
        expanded = binary_fill_holes(expanded)
    return (passed | expanded) & eligible


def _combine_candidate_masks(
    stitched: np.ndarray,
    classical_union: np.ndarray,
    pass_footprint: np.ndarray,
    valid: np.ndarray,
    cfg: object,
) -> np.ndarray:
    """Combine learned and classical candidates under the configured footprint rule."""

    segmenter_binary = np.asarray(stitched) >= float(cfg.MASK_THRESHOLD)
    classical = np.asarray(classical_union, dtype=bool).copy()
    footprint = np.asarray(pass_footprint, dtype=bool)
    valid_mask = np.asarray(valid, dtype=bool)
    if not (
        segmenter_binary.shape == classical.shape == footprint.shape == valid_mask.shape
    ):
        raise ValueError("Candidate masks, footprint, and valid mask must align.")
    combined = segmenter_binary.copy()
    if bool(cfg.CLASSICAL_UNION_ENABLED):
        if bool(cfg.CLASSICAL_UNION_RESTRICT_TO_PASS_FOOTPRINT):
            classical &= footprint
        combined |= classical
    return combined & valid_mask


def _validate_configuration(cfg: object) -> None:
    expected: dict[str, Any] = {
        "RANDOM_SEED": 42,
        "TILE_SIZE": 256,
        "STRIDE": 224,
        "SCREENER_THRESHOLD": 0.45,
        "MASK_THRESHOLD": 0.10,
        "BASELINE_THRESHOLD": 0.30,
        "MAX_INVALID_FRACTION": 0.20,
        "INFILL_INVALID_PIXELS": True,
        "EXCLUDE_BOUNDARY_ADJACENT_TILES": False,
        "SCREENER_PASS_MAP_CLOSING": True,
        "SCREENER_PASS_MAP_HOLE_FILLING": True,
        "CLASSICAL_UNION_ENABLED": True,
        "CLASSICAL_UNION_RESTRICT_TO_PASS_FOOTPRINT": True,
        "CHAIN_BINARY_FILL_HOLES": False,
        "MORPH_OPENING_RADIUS": 2,
        "MIN_BLOB_PX": 1000,
    }
    mismatches = {
        name: (getattr(cfg, name, None), value)
        for name, value in expected.items()
        if getattr(cfg, name, None) != value
    }
    if mismatches:
        raise RuntimeError(f"Operating configuration mismatch: {mismatches}")
    for name in ("OPERATING_SCREENER_CHECKPOINT", "OPERATING_SEGMENTER_CHECKPOINT"):
        path = Path(getattr(cfg, name))
        if not path.exists():
            raise FileNotFoundError(
                f"Missing operating checkpoint: {path}. Download "
                "saudi-oil-spill-assets.zip from "
                "https://github.com/YOURNAME/saudi-oil-spill-system/releases/latest "
                "and extract it from the repository root to install the "
                "checkpoints in models/."
            )


def run_operating_inference(
    image: np.ndarray,
    nodata: float | None,
    cfg: object = config,
) -> OperatingInferenceResult:
    """Apply the released two-stage pipeline to one raw SAR scene.

    Args:
        image: Non-empty ``(height, width)`` SAR array. It is converted to
            ``float32`` without changing its spatial grid.
        nodata: Declared source nodata value, or ``None``.
        cfg: Configuration object. A guard requires every operating value to
            equal the published configuration before model execution begins.

    Returns:
        :class:`OperatingInferenceResult` containing stitched learned
        probabilities, the final post-processed mask, pass footprint, and
        rejection/tile counts.

    Raises:
        ValueError: If the scene is empty or not two-dimensional.
        FileNotFoundError: If either checkpoint is absent.
        RuntimeError: If settings, checkpoint epochs, CUDA, or bf16 support do
            not match the released operating requirements.

    The second pass re-slices normalized tiles from the padded scene instead of
    retaining every first-pass batch, which keeps peak memory lower. Incomplete
    CUDA groups are padded by repeating the final passing tile solely to retain
    the historical convolution batch shape; repeated outputs are discarded.
    This is necessary because bf16 values close to the 0.10 threshold can shift
    with batch shape even when weights and inputs are otherwise identical.
    """

    _validate_configuration(cfg)
    scene = np.asarray(image, dtype=np.float32)
    if scene.ndim != 2 or not scene.size:
        raise ValueError("Operating inference expects a non-empty 2D scene.")
    _seed()
    device = _device(cfg)
    screener, segmenter = _load_models(
        str(Path(cfg.OPERATING_SCREENER_CHECKPOINT).resolve()),
        str(Path(cfg.OPERATING_SEGMENTER_CHECKPOINT).resolve()),
        str(device),
    )
    tile_size = int(cfg.TILE_SIZE)
    stride = int(cfg.STRIDE)
    batch_size = int(cfg.OPERATING_BATCH_SIZE)
    padded = pad_to_tile(scene, tile_size)
    row_starts = tile_starts(padded.shape[0], tile_size, stride)
    col_starts = tile_starts(padded.shape[1], tile_size, stride)
    locations = [(x, y) for y in row_starts for x in col_starts]
    scene_range = (
        scene_stats(
            scene,
            lo_pct=float(cfg.NORM_LO_PCT),
            hi_pct=float(cfg.NORM_HI_PCT),
            nodata=nodata,
        )
        if str(cfg.NORM_SCOPE) == "scene"
        else None
    )
    scene_median = (
        scene_valid_median(scene, nodata) if bool(cfg.INFILL_INVALID_PIXELS) else None
    )

    eligible, f1_rejected = _partition_tile_locations(
        padded,
        locations,
        tile_size,
        nodata,
        float(cfg.MAX_INVALID_FRACTION),
    )

    probability_by_location: dict[tuple[int, int], float] = {}
    autocast_enabled = device.type == "cuda"
    with torch.no_grad():
        for start in range(0, len(eligible), batch_size):
            batch_locations = eligible[start : start + batch_size]
            arrays = _normalized_batch(
                padded,
                batch_locations,
                tile_size,
                nodata,
                scene_range,
                scene_median,
                cfg,
            )
            tensor = torch.from_numpy(arrays).unsqueeze(1).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                probabilities = torch.softmax(screener(tensor).float(), dim=1)[:, 0]
            for location, probability in zip(
                batch_locations, probabilities.cpu().numpy(), strict=True
            ):
                probability_by_location[location] = float(probability)

    x_index = {value: index for index, value in enumerate(col_starts)}
    y_index = {value: index for index, value in enumerate(row_starts)}
    eligible_grid = np.zeros((len(row_starts), len(col_starts)), dtype=bool)
    passed_grid = np.zeros_like(eligible_grid)
    for x, y in eligible:
        index = (y_index[y], x_index[x])
        eligible_grid[index] = True
        passed_grid[index] = probability_by_location[(x, y)] >= float(
            cfg.SCREENER_THRESHOLD
        )
    final_grid = _expand_pass_grid(
        passed_grid,
        eligible_grid,
        closing=bool(cfg.SCREENER_PASS_MAP_CLOSING),
        fill_holes=bool(cfg.SCREENER_PASS_MAP_HOLE_FILLING),
    )
    final_passed = {
        (col_starts[col], row_starts[row]) for row, col in np.argwhere(final_grid)
    }

    probability_sum = np.zeros(padded.shape, dtype=np.float32)
    contributing_count = np.zeros(padded.shape, dtype=np.uint16)
    classical_union = np.zeros(padded.shape, dtype=bool)
    # Preserve the original row-major eligible-tile order while restricting the
    # expensive second stage to the morphology-expanded pass set. Re-slicing
    # from ``padded`` keeps peak memory below caching every normalized tile from
    # the screening pass.
    # CUDA bf16 inference is sensitive to convolution batch shape at pixels very
    # close to MASK_THRESHOLD. Keep the old full-batch shape for locations that
    # originally belonged to full eligible batches, and keep the old tail shape
    # for locations from the original tail batch. Padding repeats only final-pass
    # locations; no screened-out tile reaches the segmenter, and padded outputs
    # are ignored. This preserves reference outputs while retaining the gating gain.
    full_batch_boundary = (len(eligible) // batch_size) * batch_size
    segmenter_groups = [
        (
            [
                location
                for location in eligible[:full_batch_boundary]
                if location in final_passed
            ],
            batch_size,
        )
    ]
    tail_locations = [
        location
        for location in eligible[full_batch_boundary:]
        if location in final_passed
    ]
    if tail_locations:
        segmenter_groups.append((tail_locations, len(eligible) - full_batch_boundary))
    with torch.no_grad():
        for group_locations, model_batch_size in segmenter_groups:
            for start in range(0, len(group_locations), model_batch_size):
                batch_locations = group_locations[start : start + model_batch_size]
                model_locations = list(batch_locations)
                model_locations.extend(
                    [batch_locations[-1]] * (model_batch_size - len(batch_locations))
                )
                arrays = _normalized_batch(
                    padded,
                    model_locations,
                    tile_size,
                    nodata,
                    scene_range,
                    scene_median,
                    cfg,
                )
                tensor = torch.from_numpy(arrays).unsqueeze(1).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    segmenter_probabilities = torch.sigmoid(segmenter(tensor).float())[
                        :, 0
                    ]
                probabilities = (
                    segmenter_probabilities[: len(batch_locations)].cpu().numpy()
                )
                for index, ((x, y), probability) in enumerate(
                    zip(batch_locations, probabilities, strict=True)
                ):
                    probability_sum[y : y + tile_size, x : x + tile_size] += probability
                    contributing_count[y : y + tile_size, x : x + tile_size] += 1
                    if bool(cfg.CLASSICAL_UNION_ENABLED):
                        classical = models.segment(arrays[index]).astype(bool)
                        classical_union[
                            y : y + tile_size, x : x + tile_size
                        ] |= classical

    stitched = probability_sum / np.maximum(contributing_count, 1)
    height, width = scene.shape
    stitched = stitched[:height, :width].astype(np.float32, copy=False)
    counts = contributing_count[:height, :width]
    classical_union = classical_union[:height, :width]
    valid = valid_pixel_mask(scene, nodata)
    stitched[~valid] = 0.0
    counts[~valid] = 0
    pass_footprint = counts > 0
    combined = _combine_candidate_masks(
        stitched,
        classical_union,
        pass_footprint,
        valid,
        cfg,
    )
    if bool(cfg.CHAIN_BINARY_FILL_HOLES):
        combined = binary_fill_holes(combined) & valid
    opened = binary_opening(combined, footprint=disk(int(cfg.MORPH_OPENING_RADIUS)))
    binary = remove_small_objects(
        opened,
        min_size=int(cfg.MIN_BLOB_PX),
        connectivity=int(cfg.BLOB_CONNECTIVITY),
    ).astype(bool, copy=False)

    rejected = list(f1_rejected)
    for x, y in eligible:
        if (x, y) not in final_passed:
            rejected.append(
                {
                    "x": x,
                    "y": y,
                    "reason": "oil_probability_below_threshold",
                    "oil_probability": probability_by_location[(x, y)],
                    "screener_threshold": float(cfg.SCREENER_THRESHOLD),
                }
            )
    return OperatingInferenceResult(
        probabilities=stitched,
        binary=binary,
        counts=counts,
        pass_footprint=pass_footprint,
        rejected_tiles=rejected,
        base_passed_tiles=int(passed_grid.sum()),
        closing_added_tiles=int((final_grid & ~passed_grid).sum()),
        final_passed_tiles=int(final_grid.sum()),
    )

"""Render native-resolution SAR and diagnostic layers for the analyst UI.

Rendering is deliberately downstream of analysis: it reads an immutable
:class:`backend.service.AnalysisRecord`, composites presentation overlays, and
returns PNG bytes. No function in this module can alter the cached detection,
measurements, or model state.
"""

from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from rasterio.warp import transform as transform_coordinates
from skimage.measure import find_contours, label

import config
from backend.diagnostics import REJECTED_BY_STAGE1
from backend.service import AnalysisRecord
from src.normalize import apply_norm, fixed_db_norm, scene_stats
from src.tiling import valid_pixel_mask

MASK_COLOUR = "#d64040"
FOOTPRINT_COLOUR = "#27b6a5"
REJECTED_COLOUR = "#e5c453"
REGION_LABEL_COLOUR = "#f6f8fa"
REGION_HIGHLIGHT_COLOUR = "#70ddff"
UNDETECTED_STAGE1_COLOUR = "#e5c453"
UNDETECTED_ALPHA = 0.62
PNG_COMPRESSION_LEVEL = 3
FOOTPRINT_ALPHA = 0.30
REFERENCE_DISPLAY_EDGE_PX = 1_050.0
REJECTED_OUTLINE_WIDTH_SCALE = 0.55
REJECTED_OUTLINE_ALPHA = 0.72
HIGHLIGHT_WIDTH_SCALE = 1.8
HIGHLIGHT_ALPHA = 0.98


def _hex_rgb(colour: str) -> tuple[int, int, int]:
    value = colour.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _display_uint8(image: np.ndarray, nodata: float | None) -> np.ndarray:
    if config.NORM_SCOPE == "fixed_db":
        normalized = fixed_db_norm(image, config.FIXED_DB_LO, config.FIXED_DB_HI)
    else:
        lo, hi = scene_stats(
            image,
            config.NORM_LO_PCT,
            config.NORM_HI_PCT,
            nodata=nodata,
        )
        normalized = apply_norm(image, lo, hi)
    invalid = ~valid_pixel_mask(image, nodata)
    values = np.ma.filled(normalized, 0.0)
    values = np.where(invalid, 0.0, values)
    return np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)


def _blend_colour(
    rgb: np.ndarray,
    mask: np.ndarray,
    colour: str,
    alpha: float,
) -> None:
    if not np.any(mask):
        return
    target = _hex_rgb(colour)
    keep = 1.0 - float(alpha)
    for channel, value in enumerate(target):
        source = rgb[..., channel]
        source[mask] = np.rint(source[mask] * keep + value * alpha).astype(np.uint8)


def _rejected_coverage(record: AnalysisRecord) -> np.ndarray:
    height, width = record.result.binary.shape
    coverage = np.zeros((height, width), dtype=bool)
    for tile in record.result.rejected_tiles:
        x = int(tile["x"])
        y = int(tile["y"])
        coverage[
            y : min(y + config.TILE_SIZE, height),
            x : min(x + config.TILE_SIZE, width),
        ] = True
    return coverage


def _visible_contour_segments(
    contour: np.ndarray,
    shape: tuple[int, int],
    occlusion_mask: np.ndarray | None,
) -> list[list[tuple[float, float]]]:
    points = [(float(column - 1), float(row - 1)) for row, column in contour]
    if occlusion_mask is None:
        return [points]
    height, width = shape
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for x, y in points:
        column = min(max(int(round(x)), 0), width - 1)
        row = min(max(int(round(y)), 0), height - 1)
        if occlusion_mask[row, column]:
            if len(current) > 1:
                segments.append(current)
            current = []
        else:
            current.append((x, y))
    if len(current) > 1:
        segments.append(current)
    return segments


def _draw_merged_boundary(
    canvas: Image.Image,
    binary: np.ndarray,
    colour: str,
    *,
    width: int,
    alpha: float,
    occlusion_mask: np.ndarray | None = None,
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    padded = np.pad(np.asarray(binary, dtype=np.uint8), 1, mode="constant")
    rgba = (*_hex_rgb(colour), int(round(255 * alpha)))
    for contour in find_contours(padded, level=0.5):
        for segment in _visible_contour_segments(
            contour,
            binary.shape,
            occlusion_mask,
        ):
            draw.line(segment, fill=rgba, width=width, joint="curve")


@lru_cache(maxsize=8)
def _mono_font(size: int) -> ImageFont.FreeTypeFont:
    path = font_manager.findfont("DejaVu Sans Mono")
    return ImageFont.truetype(path, size=size)


def _region_pixel_centres(
    record: AnalysisRecord,
) -> list[tuple[int, float, float]]:
    frame = record.result.regions_df.sort_values("region_id")
    if frame.empty:
        return []
    xs, ys = transform_coordinates(
        "EPSG:4326",
        record.result.crs,
        frame["centroid_lon"].astype(float).tolist(),
        frame["centroid_lat"].astype(float).tolist(),
    )
    inverse = ~record.result.transform
    centres: list[tuple[int, float, float]] = []
    for region_id, x, y in zip(frame["region_id"], xs, ys, strict=True):
        column, row = inverse * (x, y)
        centres.append((int(region_id), float(column), float(row)))
    return centres


def _draw_region_labels(canvas: Image.Image, record: AnalysisRecord) -> None:
    long_edge = max(canvas.size)
    font_size = max(14, int(round(long_edge / 110.0)))
    font = _mono_font(font_size)
    padding = max(3, int(round(font_size * 0.18)))
    stroke = max(1, int(round(font_size * 0.035)))
    draw = ImageDraw.Draw(canvas, "RGBA")
    for region_id, column, row in _region_pixel_centres(record):
        text = str(region_id)
        box = draw.textbbox(
            (column, row),
            text,
            font=font,
            anchor="mm",
            stroke_width=stroke,
        )
        rectangle = (
            box[0] - padding,
            box[1] - padding,
            box[2] + padding,
            box[3] + padding,
        )
        draw.rounded_rectangle(
            rectangle,
            radius=padding,
            fill=(8, 12, 16, 210),
            outline=(*_hex_rgb(REGION_LABEL_COLOUR), 225),
            width=max(1, stroke),
        )
        draw.text(
            (column, row),
            text,
            font=font,
            fill=(*_hex_rgb(REGION_LABEL_COLOUR), 255),
            anchor="mm",
            stroke_width=stroke,
            stroke_fill=(8, 12, 16, 255),
        )


def render_png(
    record: AnalysisRecord,
    layer: str,
    merged_rejected: bool = False,
    highlight_region: int | None = None,
    show_region_labels: bool = True,
) -> bytes:
    """Render one native source pixel to one PNG pixel.

    Args:
        record: Cached analysis and source image on a common pixel grid.
        layer: ``mask``, ``footprint``, ``sar``, or ``undetected``.
        merged_rejected: Add the yellow union of all rejected tile footprints.
        highlight_region: Optional connected-region ID to outline.
        show_region_labels: Draw IDs at measured region centroids except on SAR.

    Returns:
        PNG file bytes at the source raster's native width and height. Keeping
        native pixels makes browser 1:1 zoom meaningful and keeps every overlay
        aligned without client-side reprojection.

    Raises:
        ValueError: If ``layer`` is not supported.
    """

    if layer not in {"mask", "footprint", "sar", "undetected"}:
        raise ValueError(f"Unknown render layer: {layer}")

    grey = _display_uint8(record.source_image, record.source_nodata)
    requires_colour = layer != "sar" or merged_rejected
    if not requires_colour:
        canvas = Image.fromarray(grey)
    else:
        rgb = np.repeat(grey[..., np.newaxis], 3, axis=2)
        oil = np.asarray(record.result.binary, dtype=bool)
        rejected = None
        if merged_rejected and record.result.rejected_tiles:
            rejected = _rejected_coverage(record)
            visible_rejected = (
                rejected & ~oil if layer in {"mask", "undetected"} else rejected
            )
            _blend_colour(
                rgb,
                visible_rejected,
                REJECTED_COLOUR,
                config.REJECTED_OVERLAY_ALPHA,
            )
        if layer == "mask":
            _blend_colour(rgb, oil, MASK_COLOUR, config.OVERLAY_ALPHA)
        elif layer == "footprint":
            footprint = np.asarray(record.result.full_mask > 0.0, dtype=bool)
            _blend_colour(rgb, footprint, FOOTPRINT_COLOUR, FOOTPRINT_ALPHA)
        elif layer == "undetected":
            diagnostic = record.undetected_dark.labels
            _blend_colour(
                rgb,
                diagnostic == REJECTED_BY_STAGE1,
                UNDETECTED_STAGE1_COLOUR,
                UNDETECTED_ALPHA,
            )
            _blend_colour(rgb, oil, MASK_COLOUR, config.OVERLAY_ALPHA)
        canvas = Image.fromarray(rgb)

        visual_scale = max(record.source_image.shape) / REFERENCE_DISPLAY_EDGE_PX
        if rejected is not None:
            _draw_merged_boundary(
                canvas,
                rejected,
                REJECTED_COLOUR,
                width=max(1, int(round(REJECTED_OUTLINE_WIDTH_SCALE * visual_scale))),
                alpha=REJECTED_OUTLINE_ALPHA,
                occlusion_mask=oil if layer in {"mask", "undetected"} else None,
            )
        if layer != "sar" and highlight_region is not None and highlight_region > 0:
            components = label(
                oil,
                connectivity=int(config.BLOB_CONNECTIVITY),
            )
            highlighted = components == int(highlight_region)
            if np.any(highlighted):
                _draw_merged_boundary(
                    canvas,
                    highlighted,
                    REGION_HIGHLIGHT_COLOUR,
                    width=max(1, int(round(HIGHLIGHT_WIDTH_SCALE * visual_scale))),
                    alpha=HIGHLIGHT_ALPHA,
                )
        if layer != "sar" and show_region_labels:
            _draw_region_labels(canvas, record)

    buffer = io.BytesIO()
    canvas.save(
        buffer,
        format="PNG",
        compress_level=PNG_COMPRESSION_LEVEL,
        optimize=False,
    )
    return buffer.getvalue()

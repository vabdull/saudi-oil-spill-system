"""Expose the released analysis service through FastAPI and serve the web client.

HTTP handlers validate scene selection or GeoTIFF upload, delegate expensive work
to :class:`backend.service.AnalysisService`, and serialize cached renders and
exports. Model execution is deliberately absent from this module so requests,
caching, inference, measurement, and rendering remain independently testable.
"""

from __future__ import annotations

import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from rasterio.errors import RasterioError
from starlette.datastructures import UploadFile

from backend.rendering import render_png
from backend.service import analysis_service
from src.pipeline import save_mask_geotiff

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_RENDER_CACHE_ITEMS = 8
MAX_RENDER_CACHE_BYTES = 384 * 1024 * 1024
_render_cache: OrderedDict[tuple[str, str, str, bool, int | None, bool], bytes] = (
    OrderedDict()
)
_render_lock = threading.RLock()

app = FastAPI(
    title="Saudi Oil Spill Detection API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)
app.mount("/assets", StaticFiles(directory=FRONTEND_ROOT), name="assets")


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc).strip("'"))
    if isinstance(exc, (ValueError, FileNotFoundError, RasterioError)):
        return HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}")
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/", include_in_schema=False)
def frontend_index() -> FileResponse:
    """Serve the framework-free dashboard shell.

    Returns:
        ``frontend/index.html`` as a file response.
    """

    return FileResponse(FRONTEND_ROOT / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Report process health and the active analysis-configuration identity.

    Returns:
        JSON object containing ``status`` and the current deterministic
        ``config_hash``. This endpoint does not load a scene or run inference.
    """

    return {"status": "ok", "config_hash": analysis_service.config_hash}


@app.get("/api/scenes")
def scenes() -> dict[str, Any]:
    """List packaged demo scenes that are present and geospatially readable.

    Returns:
        JSON object with a ``scenes`` list of IDs, labels, and validated-region
        flags.

    Raises:
        HTTPException: With an actionable API detail if scene metadata fails.
    """

    try:
        return {"scenes": analysis_service.list_scenes()}
    except Exception as exc:
        raise _http_error(exc) from exc


async def _analysis_input(
    request: Request,
) -> tuple[str | None, bytes | None, str | None]:
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        payload = await request.json()
        return payload.get("scene"), None, None
    if (
        "multipart/form-data" in content_type
        or "application/x-www-form-urlencoded" in content_type
    ):
        form = await request.form()
        scene = form.get("scene")
        upload = form.get("file")
        if isinstance(upload, UploadFile):
            data = await upload.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise ValueError("Uploaded GeoTIFF exceeds the 500 MB limit.")
            return (
                str(scene) if scene else None,
                data,
                upload.filename or "uploaded_scene.tif",
            )
        return str(scene) if scene else None, None, None
    raise ValueError("POST /api/analyze requires JSON or multipart form data.")


@app.post("/api/analyze")
async def analyze(request: Request) -> dict[str, Any]:
    """Analyze exactly one named demo scene or uploaded single-band GeoTIFF.

    Args:
        request: JSON ``{"scene": name}`` request or multipart request carrying
            one ``file`` field. Uploads are limited to 500 MiB.

    Returns:
        JSON-compatible analysis record including metrics, regions, metadata,
        and diagnostic summaries.

    Raises:
        HTTPException: ``400`` for invalid input/geospatial data, ``404`` for an
            unknown scene, or ``500`` for an unexpected processing failure.
    """

    try:
        scene, upload_data, upload_name = await _analysis_input(request)
        if bool(scene) == bool(upload_data):
            raise ValueError("Provide exactly one demo scene name or uploaded GeoTIFF.")
        if upload_data is not None:
            record, cached = analysis_service.analyze_upload(
                upload_data, upload_name or "uploaded_scene.tif"
            )
        else:
            record, cached = analysis_service.analyze_demo(scene or "")
        return record.response(cached=cached)
    except HTTPException:
        raise
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/render/{scene_id}/{layer}")
def render(
    scene_id: str,
    layer: str,
    merged_rejected: bool = False,
    highlight_region: int | None = None,
    show_region_labels: bool = True,
) -> Response:
    """Render a native-resolution PNG for one cached analysis layer.

    Args:
        scene_id: Demo filename or content-addressed upload identifier.
        layer: One of ``sar``, ``mask``, ``footprint``, or ``undetected``.
        merged_rejected: Overlay merged Stage-1-rejected tile coverage.
        highlight_region: Optional connected-region ID to outline.
        show_region_labels: Draw region IDs on layers that support them.

    Returns:
        ``image/png`` response. The bounded LRU cache is keyed by every visual
        option plus scene and configuration identity, so display interaction
        never reruns inference.

    Raises:
        HTTPException: If the scene is unavailable or a render option is invalid.
    """

    try:
        record = analysis_service.get(scene_id)
        cache_key = (
            scene_id,
            record.config_hash,
            layer,
            merged_rejected,
            highlight_region,
            show_region_labels,
        )
        with _render_lock:
            png = _render_cache.get(cache_key)
            if png is None:
                png = render_png(
                    record,
                    layer,
                    merged_rejected=merged_rejected,
                    highlight_region=highlight_region,
                    show_region_labels=show_region_labels,
                )
                _render_cache[cache_key] = png
                _render_cache.move_to_end(cache_key)
                while (
                    len(_render_cache) > MAX_RENDER_CACHE_ITEMS
                    or sum(len(value) for value in _render_cache.values())
                    > MAX_RENDER_CACHE_BYTES
                ):
                    _render_cache.popitem(last=False)
            else:
                _render_cache.move_to_end(cache_key)
        return Response(
            png,
            media_type="image/png",
            headers={
                "Content-Disposition": f'inline; filename="{scene_id}_{layer}.png"'
            },
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/export/regions/{scene_id}")
def export_regions(scene_id: str) -> Response:
    """Download the measured region table for a cached scene as UTF-8 CSV.

    Args:
        scene_id: Demo or upload analysis identifier.

    Returns:
        ``text/csv`` response sorted by region area descending.
    """

    try:
        record = analysis_service.get(scene_id)
        ordered = record.result.regions_df.sort_values("area_km2", ascending=False)
        return Response(
            ordered.to_csv(index=False).encode("utf-8"),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{scene_id}_regions.csv"'
            },
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/export/mask/{scene_id}")
def export_mask(scene_id: str) -> Response:
    """Download the byte-valued final mask on the source GeoTIFF grid.

    Args:
        scene_id: Demo or upload analysis identifier.

    Returns:
        Deflate-compressed single-band GeoTIFF response with source CRS and
        affine transform preserved.
    """

    try:
        record = analysis_service.get(scene_id)
        with tempfile.TemporaryDirectory(prefix="oil-mask-api-") as directory:
            path = Path(directory) / f"{scene_id}_mask.tif"
            save_mask_geotiff(record.result, path)
            data = path.read_bytes()
        return Response(
            data,
            media_type="image/tiff",
            headers={
                "Content-Disposition": f'attachment; filename="{scene_id}_mask.tif"'
            },
        )
    except Exception as exc:
        raise _http_error(exc) from exc


def run() -> None:
    """Start the local Uvicorn server on ``127.0.0.1:8000``.

    This is the public ``python -m backend.main`` entry point. Importing the
    module does not start a server.
    """

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    run()

/**
 * Framework-free client for the Saudi Oil Spill Detection API.
 *
 * Browser state lives in `state`; DOM references live in `elements`. Scene
 * selection or upload calls `POST /api/analyze` once, while layer, label,
 * rejected-coverage, zoom, and region-highlight interactions request only
 * cached PNG renders. No client operation can mutate pipeline configuration.
 */

const state = {
  analysis: null,
  layer: "mask",
  mergedRejected: false,
  showRegionLabels: true,
  highlightRegion: null,
  hoverTimer: null,
  timer: null,
  startedAt: 0,
  resetViewOnImageLoad: true,
  view: {
    scale: 1,
    fitScale: 1,
    x: 0,
    y: 0,
    naturalWidth: 0,
    naturalHeight: 0,
    frameWidth: 0,
    frameHeight: 0,
    dragging: false,
    pointerId: null,
    lastPointerX: 0,
    lastPointerY: 0,
  },
};

const byId = (id) => document.getElementById(id);

const elements = {
  sceneSelect: byId("scene-select"),
  sceneUpload: byId("scene-upload"),
  uploadName: byId("upload-name"),
  rejectedToggle: byId("rejected-toggle"),
  regionLabelsToggle: byId("region-labels-toggle"),
  rejectedLegend: byId("rejected-legend"),
  undetectedLegend: byId("undetected-legend"),
  undetectedPanel: byId("undetected-panel"),
  imageFrame: document.querySelector(".image-frame"),
  sceneImage: byId("scene-image"),
  zoomOut: byId("zoom-out"),
  zoomIn: byId("zoom-in"),
  zoomReset: byId("zoom-reset"),
  zoomReadout: byId("zoom-readout"),
  loading: byId("loading-state"),
  elapsed: byId("loading-elapsed"),
  errorPanel: byId("error-panel"),
  errorMessage: byId("error-message"),
};

const formats = {
  area: new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 4 }),
  one: new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 }),
  four: new Intl.NumberFormat("en-US", { minimumFractionDigits: 4, maximumFractionDigits: 4 }),
  score: new Intl.NumberFormat("en-US", { minimumFractionDigits: 4, maximumFractionDigits: 4 }),
  percent: new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 }),
};

const severityClasses = [
  "severity-low",
  "severity-moderate",
  "severity-high",
  "severity-critical",
];

const confidenceClasses = [
  "confidence-low",
  "confidence-medium",
  "confidence-high",
];

function setText(id, value) {
  byId(id).textContent = value;
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function format(value, formatter, fallback = "—") {
  return finiteNumber(value) ? formatter.format(value) : fallback;
}

function clampView() {
  // Keeping at least one image edge in each direction prevents a user from
  // losing a large native-resolution scene outside the viewport.
  const view = state.view;
  if (!view.naturalWidth || !view.naturalHeight) return;
  const frameWidth = elements.imageFrame.clientWidth;
  const frameHeight = elements.imageFrame.clientHeight;
  const scaledWidth = view.naturalWidth * view.scale;
  const scaledHeight = view.naturalHeight * view.scale;
  view.x = scaledWidth <= frameWidth
    ? (frameWidth - scaledWidth) / 2
    : Math.min(0, Math.max(frameWidth - scaledWidth, view.x));
  view.y = scaledHeight <= frameHeight
    ? (frameHeight - scaledHeight) / 2
    : Math.min(0, Math.max(frameHeight - scaledHeight, view.y));
  view.frameWidth = frameWidth;
  view.frameHeight = frameHeight;
}

function updateZoomControls() {
  const view = state.view;
  const atFit = Math.abs(view.scale - view.fitScale) < 0.0001;
  const atNative = Math.abs(view.scale - 1) < 0.0001;
  elements.zoomReadout.textContent = `${Math.round(view.scale * 100)}%`;
  elements.zoomReadout.title = atFit ? "Fit to panel" : "Current image scale";
  elements.zoomOut.disabled = atFit;
  elements.zoomIn.disabled = atNative;
}

function applyView() {
  const view = state.view;
  clampView();
  elements.sceneImage.style.transform =
    `translate3d(${view.x}px, ${view.y}px, 0) scale(${view.scale})`;
  updateZoomControls();
}

function resetView() {
  const view = state.view;
  if (!view.naturalWidth || !view.naturalHeight) return;
  const frameWidth = elements.imageFrame.clientWidth;
  const frameHeight = elements.imageFrame.clientHeight;
  view.fitScale = Math.min(
    frameWidth / view.naturalWidth,
    frameHeight / view.naturalHeight,
    1,
  );
  view.scale = view.fitScale;
  view.x = (frameWidth - view.naturalWidth * view.scale) / 2;
  view.y = (frameHeight - view.naturalHeight * view.scale) / 2;
  applyView();
}

function zoomAt(factor, clientX, clientY) {
  // Preserve the source pixel under the cursor while changing scale; otherwise
  // fine SAR features jump away from the analyst during wheel zoom.
  const view = state.view;
  if (!view.naturalWidth || !view.naturalHeight) return;
  const bounds = elements.imageFrame.getBoundingClientRect();
  const anchorX = clientX - bounds.left;
  const anchorY = clientY - bounds.top;
  const imageX = (anchorX - view.x) / view.scale;
  const imageY = (anchorY - view.y) / view.scale;
  const nextScale = Math.min(1, Math.max(view.fitScale, view.scale * factor));
  view.x = anchorX - imageX * nextScale;
  view.y = anchorY - imageY * nextScale;
  view.scale = nextScale;
  applyView();
}

function zoomFromCentre(factor) {
  const bounds = elements.imageFrame.getBoundingClientRect();
  zoomAt(
    factor,
    bounds.left + bounds.width / 2,
    bounds.top + bounds.height / 2,
  );
}

function prepareNativeImage(reset) {
  const view = state.view;
  view.naturalWidth = elements.sceneImage.naturalWidth;
  view.naturalHeight = elements.sceneImage.naturalHeight;
  elements.sceneImage.style.width = `${view.naturalWidth}px`;
  elements.sceneImage.style.height = `${view.naturalHeight}px`;
  const nextFit = Math.min(
    elements.imageFrame.clientWidth / view.naturalWidth,
    elements.imageFrame.clientHeight / view.naturalHeight,
    1,
  );
  if (reset || view.scale < nextFit || view.scale > 1) {
    resetView();
  } else {
    view.fitScale = nextFit;
    applyView();
  }
}

function applySeverityClass(element, severity) {
  element.classList.remove(...severityClasses);
  const normalized = String(severity ?? "").trim().toLowerCase();
  const className = `severity-${normalized}`;
  if (severityClasses.includes(className)) element.classList.add(className);
}

function applyConfidenceClass(element, confidence) {
  element.classList.remove(...confidenceClasses);
  const normalized = String(confidence ?? "").trim().toLowerCase();
  const className = `confidence-${normalized}`;
  if (confidenceClasses.includes(className)) element.classList.add(className);
}

function showError(error) {
  const message = error instanceof Error ? error.message : String(error);
  elements.errorMessage.textContent = `${message}\n\nVerify that .venv is active and all dependencies are installed.`;
  elements.errorPanel.hidden = false;
  setRailStatus("error", "Analysis error");
}

function clearError() {
  elements.errorPanel.hidden = true;
  elements.errorMessage.textContent = "";
}

function setRailStatus(mode, text) {
  byId("rail-status").dataset.state = mode;
  setText("rail-status-text", text);
}

function startLoading(label = "Running analysis") {
  clearError();
  elements.loading.querySelector("strong").textContent = label;
  elements.loading.hidden = false;
  state.startedAt = performance.now();
  elements.elapsed.textContent = "0.0 s";
  clearInterval(state.timer);
  state.timer = window.setInterval(() => {
    const elapsed = (performance.now() - state.startedAt) / 1000;
    elements.elapsed.textContent = `${elapsed.toFixed(1)} s`;
  }, 100);
  setRailStatus("busy", label);
}

function stopLoading(cached = false) {
  clearInterval(state.timer);
  state.timer = null;
  elements.loading.hidden = true;
  setRailStatus("ready", cached ? "Cached analysis ready" : "Analysis ready");
}

async function requestJson(url, options = {}) {
  /** Fetch JSON and surface the backend's actionable `detail` message. */
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    throw new Error(payload?.detail || `${response.status} ${response.statusText}`);
  }
  return payload;
}

function renderUrl(layer = state.layer, merged = state.mergedRejected) {
  /** Build a cache-complete render URL without triggering analysis. */
  if (!state.analysis) return "";
  const scene = encodeURIComponent(state.analysis.scene_id);
  const query = new URLSearchParams({
    merged_rejected: String(merged),
    config: state.analysis.config_hash,
    show_region_labels: String(state.showRegionLabels),
  });
  if (layer !== "sar" && state.highlightRegion !== null) {
    query.set("highlight_region", String(state.highlightRegion));
  }
  return `/api/render/${scene}/${layer}?${query}`;
}

function updateImage({ showLoading = false, resetViewOnLoad = false } = {}) {
  /** Swap the rendered layer while preserving pan/zoom unless explicitly reset. */
  if (!state.analysis) return;
  if (resetViewOnLoad) state.resetViewOnImageLoad = true;
  if (showLoading) startLoading("Rendering cached scene");
  const expectedUrl = renderUrl();
  elements.sceneImage.onload = () => {
    prepareNativeImage(state.resetViewOnImageLoad);
    state.resetViewOnImageLoad = false;
    stopLoading(Boolean(state.analysis.cached));
    elements.sceneImage.onload = null;
    elements.sceneImage.onerror = null;
  };
  elements.sceneImage.onerror = () => {
    stopLoading(false);
    showError(new Error(`Could not render ${state.layer} layer for ${state.analysis.display_name}.`));
  };
  elements.sceneImage.src = expectedUrl;
  elements.sceneImage.alt = `${state.analysis.display_name}: ${state.layer} layer`;
  const layerName = state.layer === "undetected" ? "UNDETECTED DARK AREA" : state.layer.toUpperCase();
  setText("layer-readout", `${layerName}${state.mergedRejected ? " + REJECTED" : ""}`);
  byId("export-view").href = expectedUrl;
  byId("export-view").download = `${state.analysis.scene_id}_${state.layer}.png`;
  elements.rejectedLegend.hidden = !state.mergedRejected;
  elements.undetectedLegend.hidden = state.layer !== "undetected";
  elements.undetectedPanel.hidden = state.layer !== "undetected";
}

function updateUndetectedDark(diagnostic) {
  const categories = diagnostic.categories;
  const rejected = categories.rejected_by_stage1;
  setText("dark-total-area", format(diagnostic.total_area_km2, formats.area));
  setText("dark-total-regions", String(diagnostic.region_count));
  setText("dark-rejected-area", format(rejected.area_km2, formats.area));
  setText("dark-rejected-percent", `${format(rejected.percentage, formats.percent)}%`);
  setText("dark-rejected-regions", String(rejected.region_count));
}

function addBarCell(row, value, maximum, formatter) {
  const cell = document.createElement("td");
  cell.className = "numeric bar-cell";
  const track = document.createElement("div");
  track.className = "bar-track";
  const line = document.createElement("span");
  line.className = "bar-line";
  const fill = document.createElement("span");
  fill.className = "bar-fill";
  const width = maximum > 0 && finiteNumber(value) ? Math.max(0, Math.min(100, (value / maximum) * 100)) : 0;
  fill.style.width = `${width}%`;
  line.append(fill);
  const label = document.createElement("span");
  label.textContent = format(value, formatter);
  track.append(line, label);
  cell.append(track);
  row.append(cell);
}

function addTextCell(row, value, className = "") {
  const cell = document.createElement("td");
  cell.className = className;
  cell.textContent = value;
  row.append(cell);
}

function addSeverityCell(row, severity) {
  const cell = document.createElement("td");
  const pill = document.createElement("span");
  pill.className = "severity-pill";
  pill.textContent = severity ?? "—";
  applySeverityClass(pill, severity);
  cell.append(pill);
  row.append(cell);
}

function addConfidenceCell(row, confidence) {
  const cell = document.createElement("td");
  const pill = document.createElement("span");
  pill.className = "confidence-pill";
  pill.textContent = confidence ?? "—";
  pill.title = "Learned segmenter support; not calibrated probability";
  applyConfidenceClass(pill, confidence);
  cell.append(pill);
  row.append(cell);
}

function setRegionHighlight(regionId) {
  const nextRegion = state.layer === "sar" ? null : regionId;
  if (state.highlightRegion === nextRegion) return;
  state.highlightRegion = nextRegion;
  for (const row of document.querySelectorAll(".region-row")) {
    row.classList.toggle(
      "is-highlighted",
      nextRegion !== null && Number(row.dataset.regionId) === nextRegion,
    );
  }
  updateImage();
}

function scheduleRegionHighlight(regionId) {
  window.clearTimeout(state.hoverTimer);
  state.hoverTimer = window.setTimeout(() => setRegionHighlight(regionId), 70);
}

function populateRegionTables(regions) {
  /** Rebuild the measurement and geometry tables in descending area order. */
  const body = byId("region-table-body");
  const geometryBody = byId("geometry-table-body");
  body.replaceChildren();
  geometryBody.replaceChildren();
  window.clearTimeout(state.hoverTimer);
  state.highlightRegion = null;
  const ordered = [...regions].sort((left, right) => right.area_km2 - left.area_km2);
  const largest = ordered.length ? ordered[0].area_km2 : 0;

  for (const region of ordered) {
    const row = document.createElement("tr");
    row.className = "region-row";
    row.dataset.regionId = String(region.region_id);
    row.tabIndex = 0;
    row.addEventListener("mouseenter", () => scheduleRegionHighlight(region.region_id));
    row.addEventListener("mouseleave", () => scheduleRegionHighlight(null));
    row.addEventListener("focus", () => scheduleRegionHighlight(region.region_id));
    row.addEventListener("blur", () => scheduleRegionHighlight(null));
    addTextCell(row, String(region.region_id));
    addBarCell(row, region.area_km2, largest, formats.area);
    addTextCell(row, format(region.centroid_lat, formats.four), "numeric");
    addTextCell(row, format(region.centroid_lon, formats.four), "numeric");
    addTextCell(row, format(region.distance_to_coast_km, formats.one), "numeric");
    addBarCell(row, region.mean_probability, 1, formats.score);
    addSeverityCell(row, region.severity);
    addConfidenceCell(row, region.confidence);
    body.append(row);

    const geometryRow = document.createElement("tr");
    addTextCell(geometryRow, String(region.region_id));
    addTextCell(geometryRow, format(region.axis_major_length_px, formats.one), "numeric");
    addTextCell(geometryRow, format(region.axis_minor_length_px, formats.one), "numeric");
    addTextCell(geometryRow, format(region.orientation_deg, formats.one), "numeric");
    geometryBody.append(geometryRow);
  }

  const empty = ordered.length === 0;
  byId("no-regions").hidden = !empty;
  byId("region-table-wrap").hidden = empty;
  byId("geometry-details").hidden = empty;
  setText("region-count-label", empty ? "NO DETECTIONS" : `${ordered.length} REGIONS`);
}

function updateDashboard(analysis) {
  /** Commit one API analysis response to every dependent dashboard element. */
  state.analysis = analysis;
  const metadata = analysis.metadata;

  setText("chip-scene", analysis.display_name);
  setText("chip-region", metadata.region_label);
  setText("metric-area", format(analysis.total_area_km2, formats.area));
  setText("metric-regions", String(analysis.region_count));
  setText("metric-largest", format(analysis.largest_region_km2, formats.area));
  setText("metric-coast", format(analysis.nearest_coast_km, formats.one));
  byId("metric-coast-unit").hidden = !finiteNumber(analysis.nearest_coast_km);
  const alertLevel = analysis.alert_level ?? analysis.severity;
  setText("metric-alert", alertLevel);
  applySeverityClass(byId("metric-alert"), alertLevel);
  setText("metric-confidence", analysis.confidence);
  applyConfidenceClass(byId("metric-confidence"), analysis.confidence);
  updateUndetectedDark(analysis.undetected_dark);

  const pixel = metadata.pixel_size;
  const pixelText = pixel.unit === "m"
    ? `${format(pixel.x, formats.one)} × ${format(pixel.y, formats.one)} m pixels`
    : `${format(pixel.x, formats.four)} × ${format(pixel.y, formats.four)} degree pixels`;
  setText(
    "image-caption",
    `${pixelText} · ${metadata.width_px.toLocaleString()} × ${metadata.height_px.toLocaleString()} px · ` +
      `${metadata.invalid_tiles_excluded.toLocaleString()} tiles excluded because invalid pixels exceeded 20% · ` +
      `${metadata.tiles_screened_out.toLocaleString()} eligible tiles screened out by Stage 1.`,
  );

  populateRegionTables(analysis.regions);
  const scene = encodeURIComponent(analysis.scene_id);
  byId("export-regions").href = `/api/export/regions/${scene}`;
  byId("export-regions").download = `${analysis.scene_id}_regions.csv`;
  byId("export-mask").href = `/api/export/mask/${scene}`;
  byId("export-mask").download = `${analysis.scene_id}_mask.tif`;
  updateImage({ resetViewOnLoad: true });
}

async function analyzeDemo(scene) {
  /** Request one packaged scene; server-side caching avoids repeat inference. */
  startLoading("Running analysis");
  try {
    const analysis = await requestJson("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene }),
    });
    updateDashboard(analysis);
  } catch (error) {
    stopLoading(false);
    showError(error);
  }
}

async function analyzeUpload(file) {
  /** Upload one single-band GeoTIFF and render its content-addressed analysis. */
  startLoading("Analyzing uploaded GeoTIFF");
  const form = new FormData();
  form.append("file", file, file.name);
  try {
    const analysis = await requestJson("/api/analyze", { method: "POST", body: form });
    updateDashboard(analysis);
  } catch (error) {
    stopLoading(false);
    showError(error);
  }
}

async function initialize() {
  /** Populate available scenes and analyze the first one on initial page load. */
  try {
    const payload = await requestJson("/api/scenes");
    if (!payload.scenes.length) throw new Error("No designated demo scenes were found.");
    for (const scene of payload.scenes) {
      const option = document.createElement("option");
      option.value = scene.scene_id;
      option.textContent = scene.display_name;
      elements.sceneSelect.append(option);
    }
    await analyzeDemo(payload.scenes[0].scene_id);
  } catch (error) {
    showError(error);
  }
}

elements.sceneSelect.addEventListener("change", () => {
  elements.sceneUpload.value = "";
  elements.uploadName.textContent = "No file selected";
  analyzeDemo(elements.sceneSelect.value);
});

elements.sceneUpload.addEventListener("change", () => {
  const file = elements.sceneUpload.files?.[0];
  if (!file) return;
  elements.uploadName.textContent = file.name;
  analyzeUpload(file);
});

for (const button of document.querySelectorAll(".segment")) {
  button.addEventListener("click", () => {
    window.clearTimeout(state.hoverTimer);
    state.highlightRegion = null;
    state.layer = button.dataset.layer;
    for (const candidate of document.querySelectorAll(".segment")) {
      const active = candidate === button;
      candidate.classList.toggle("is-active", active);
      candidate.setAttribute("aria-pressed", String(active));
    }
    for (const row of document.querySelectorAll(".region-row")) {
      row.classList.remove("is-highlighted");
    }
    updateImage({ showLoading: true });
  });
}

elements.rejectedToggle.addEventListener("change", () => {
  state.mergedRejected = elements.rejectedToggle.checked;
  updateImage({ showLoading: true });
});

elements.regionLabelsToggle.addEventListener("change", () => {
  state.showRegionLabels = elements.regionLabelsToggle.checked;
  updateImage({ showLoading: true });
});

elements.imageFrame.addEventListener("wheel", (event) => {
  if (!state.analysis) return;
  event.preventDefault();
  const factor = Math.exp(-event.deltaY * 0.0015);
  zoomAt(factor, event.clientX, event.clientY);
}, { passive: false });

elements.imageFrame.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || event.target.closest(".zoom-controls")) return;
  const view = state.view;
  view.dragging = true;
  view.pointerId = event.pointerId;
  view.lastPointerX = event.clientX;
  view.lastPointerY = event.clientY;
  elements.imageFrame.classList.add("is-dragging");
  elements.imageFrame.setPointerCapture(event.pointerId);
});

elements.imageFrame.addEventListener("pointermove", (event) => {
  const view = state.view;
  if (!view.dragging || event.pointerId !== view.pointerId) return;
  view.x += event.clientX - view.lastPointerX;
  view.y += event.clientY - view.lastPointerY;
  view.lastPointerX = event.clientX;
  view.lastPointerY = event.clientY;
  applyView();
});

function finishPan(event) {
  const view = state.view;
  if (!view.dragging || event.pointerId !== view.pointerId) return;
  view.dragging = false;
  view.pointerId = null;
  elements.imageFrame.classList.remove("is-dragging");
  if (elements.imageFrame.hasPointerCapture(event.pointerId)) {
    elements.imageFrame.releasePointerCapture(event.pointerId);
  }
}

elements.imageFrame.addEventListener("pointerup", finishPan);
elements.imageFrame.addEventListener("pointercancel", finishPan);
elements.imageFrame.addEventListener("dblclick", (event) => {
  if (event.target.closest(".zoom-controls")) return;
  resetView();
});

elements.zoomIn.addEventListener("click", () => zoomFromCentre(1.35));
elements.zoomOut.addEventListener("click", () => zoomFromCentre(1 / 1.35));
elements.zoomReset.addEventListener("click", resetView);

new ResizeObserver(() => {
  const view = state.view;
  if (!view.naturalWidth || !view.naturalHeight) return;
  const wasAtFit = Math.abs(view.scale - view.fitScale) < 0.0001;
  const centreImageX = (view.frameWidth / 2 - view.x) / view.scale;
  const centreImageY = (view.frameHeight / 2 - view.y) / view.scale;
  const frameWidth = elements.imageFrame.clientWidth;
  const frameHeight = elements.imageFrame.clientHeight;
  view.fitScale = Math.min(
    frameWidth / view.naturalWidth,
    frameHeight / view.naturalHeight,
    1,
  );
  if (wasAtFit) {
    resetView();
    return;
  }
  view.scale = Math.min(1, Math.max(view.fitScale, view.scale));
  view.x = frameWidth / 2 - centreImageX * view.scale;
  view.y = frameHeight / 2 - centreImageY * view.scale;
  applyView();
}).observe(elements.imageFrame);

initialize();

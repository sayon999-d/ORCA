const fileInput = document.querySelector("#fileInput");
const depthInput = document.querySelector("#depthInput");
const analyzeButton = document.querySelector("#analyzeButton");
const deepButton = document.querySelector("#deepButton");
const zoomOutButton = document.querySelector("#zoomOutButton");
const zoomResetButton = document.querySelector("#zoomResetButton");
const zoomInButton = document.querySelector("#zoomInButton");
const canvas = document.querySelector("#previewCanvas");
const ctx = canvas.getContext("2d");
const profileCanvas = document.querySelector("#profileCanvas");
const profileCtx = profileCanvas.getContext("2d");
const profileMetaEl = document.querySelector("#profileMeta");
const overviewEl = document.querySelector("#overview");
const findingsEl = document.querySelector("#findings");
const deepEl = document.querySelector("#deep");
const reportEl = document.querySelector("#report");
const reviewsEl = document.querySelector("#reviews");
const sidebarToggle = document.querySelector("#sidebarToggle");
const candidateCountEl = document.querySelector("#candidateCount");
const deepCountEl = document.querySelector("#deepCount");
const reviewCountEl = document.querySelector("#reviewCount");
const modelStateEl = document.querySelector("#modelState");
const imageMetaEl = document.querySelector("#imageMeta");
const viewTitleEl = document.querySelector("#viewTitle");
const viewSubtitleEl = document.querySelector("#viewSubtitle");

let currentImage = null;
let currentFileName = "";
let currentResult = null;
let currentDeepResult = null;
let activeView = "overview";
let zoom = 1;
let panX = 0;
let panY = 0;
let isPanning = false;
let panStart = { x: 0, y: 0, panX: 0, panY: 0 };

const API_BASE = window.ORCA_API_BASE || localStorage.getItem("ORCA_API_BASE") || "";
const LOCAL_MODEL_LABEL = "Browser";

const viewLabels = {
  overview: ["Overview", "Current analysis state"],
  findings: ["Findings", "Candidate evidence"],
  deep: ["Deep Search", "Recursive search tree"],
  report: ["Report", "Generated run summary"],
  reviews: ["Reviews", "Human validation queue"],
};

function drawEmpty() {
  ctx.fillStyle = "#15191f";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#d8dde4";
  ctx.font = "18px system-ui";
  ctx.fillText("Orca inspection canvas", 32, 54);
}

function drawProfile(items = []) {
  profileCtx.clearRect(0, 0, profileCanvas.width, profileCanvas.height);
  profileCtx.fillStyle = "#fff";
  profileCtx.fillRect(0, 0, profileCanvas.width, profileCanvas.height);
  profileCtx.strokeStyle = "#e4e5e7";
  profileCtx.lineWidth = 1;

  for (let i = 0; i < 4; i += 1) {
    const y = 24 + i * 34;
    profileCtx.beginPath();
    profileCtx.moveTo(34, y);
    profileCtx.lineTo(profileCanvas.width - 18, y);
    profileCtx.stroke();
  }

  if (!items.length) {
    profileCtx.fillStyle = "#72757d";
    profileCtx.font = "15px system-ui";
    profileCtx.fillText("Run Analyze or Deep Search to compare pattern strength.", 34, 92);
    profileMetaEl.textContent = "No pattern data yet";
    return;
  }

  const visible = items.slice(0, 8);
  const groupWidth = (profileCanvas.width - 76) / visible.length;
  const barWidth = Math.min(18, groupWidth / 5);
  const baseline = 146;
  const chartHeight = 112;
  const series = [
    ["score", "#050505"],
    ["confidence", "#5b5f68"],
    ["novelty", "#a8a9ad"],
  ];

  visible.forEach((item, index) => {
    const x = 42 + index * groupWidth;
    series.forEach(([key, color], seriesIndex) => {
      const value = Math.max(0, Math.min(1, item[key] ?? 0));
      const height = value * chartHeight;
      profileCtx.fillStyle = color;
      profileCtx.fillRect(x + seriesIndex * (barWidth + 4), baseline - height, barWidth, height);
    });
    profileCtx.fillStyle = "#72757d";
    profileCtx.font = "12px system-ui";
    profileCtx.fillText(String(index + 1), x + 6, 166);
  });

  profileCtx.fillStyle = "#111317";
  profileCtx.font = "12px system-ui";
  profileCtx.fillText("score", profileCanvas.width - 170, 22);
  profileCtx.fillStyle = "#5b5f68";
  profileCtx.fillText("confidence", profileCanvas.width - 122, 22);
  profileCtx.fillStyle = "#a8a9ad";
  profileCtx.fillText("novelty", profileCanvas.width - 40, 22);
  profileMetaEl.textContent = `${visible.length} pattern${visible.length === 1 ? "" : "s"} compared`;
}

function imageFrame() {
  if (!currentImage) return null;
  const baseScale = Math.min(canvas.width / currentImage.width, canvas.height / currentImage.height);
  const scale = baseScale * zoom;
  const width = currentImage.width * scale;
  const height = currentImage.height * scale;
  const x = (canvas.width - width) / 2 + panX;
  const y = (canvas.height - height) / 2 + panY;
  return { scale, width, height, x, y };
}

function drawBox(frame, box, label, color) {
  const x = frame.x + box.x_min * frame.scale;
  const y = frame.y + box.y_min * frame.scale;
  const width = (box.x_max - box.x_min) * frame.scale;
  const height = (box.y_max - box.y_min) * frame.scale;
  ctx.fillStyle = "rgba(255, 255, 255, 0.14)";
  ctx.fillRect(x, y, width, height);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, width, height);
  if (width < 48 || height < 30) return;
  ctx.fillStyle = "rgba(5, 5, 5, 0.88)";
  ctx.fillRect(x, Math.max(frame.y, y - 25), 104, 23);
  ctx.fillStyle = "#fff";
  ctx.font = "13px system-ui";
  ctx.fillText(label, x + 8, Math.max(frame.y + 16, y - 9));
}

function flattenDeepNodes(nodes, output = []) {
  nodes.forEach((node) => {
    output.push(node);
    flattenDeepNodes(node.children || [], output);
  });
  return output;
}

function drawPreview() {
  drawEmpty();
  if (!currentImage) return;
  const frame = imageFrame();
  ctx.drawImage(currentImage, frame.x, frame.y, frame.width, frame.height);

  if (activeView === "deep" && currentDeepResult) {
    flattenDeepNodes(currentDeepResult.root_candidates).forEach((node) => {
      const color = node.depth === 0 ? "#f5f5f5" : "#a8a9ad";
      drawBox(frame, node.candidate.bbox, `${node.path} ${node.candidate.anomaly_score.toFixed(2)}`, color);
    });
    return;
  }

  if (currentResult) {
    currentResult.candidates.forEach((candidate, index) => {
      const color = candidate.confidence < 0.58 ? "#d6a23d" : "#f5f5f5";
      drawBox(frame, candidate.bbox, `Pattern ${index + 1}`, color);
    });
  }
}

function profileItemsFromResult(result) {
  return (result?.candidates || []).map((candidate) => ({
    score: candidate.anomaly_score,
    confidence: candidate.confidence,
    novelty: candidate.model_novelty ?? 0,
  }));
}

function profileItemsFromDeep(result) {
  return flattenDeepNodes(result?.root_candidates || []).map((node) => ({
    score: node.candidate.anomaly_score,
    confidence: node.candidate.confidence,
    novelty: node.candidate.model_novelty ?? 0,
  }));
}

function setZoom(nextZoom, anchor = null) {
  const previousZoom = zoom;
  zoom = Math.max(1, Math.min(8, Number(nextZoom.toFixed(2))));
  if (anchor && currentImage && previousZoom !== zoom) {
    panX = anchor.x - ((anchor.x - panX - canvas.width / 2) * zoom) / previousZoom - canvas.width / 2;
    panY = anchor.y - ((anchor.y - panY - canvas.height / 2) * zoom) / previousZoom - canvas.height / 2;
  }
  if (zoom === 1) {
    panX = 0;
    panY = 0;
  }
  zoomResetButton.textContent = `${Math.round(zoom * 100)}%`;
  drawPreview();
}

async function apiFetch(path, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${path}`, options);
    if (!res.ok) {
      let detail = `Request failed with status ${res.status}.`;
      try {
        const payload = await res.json();
        detail = payload.detail || detail;
      } catch {
        detail = await res.text();
      }
      throw new Error(detail || `Request failed with status ${res.status}.`);
    }
    return await res.json();
  } catch (error) {
    if (error instanceof TypeError) {
      const offlineError = new Error("Orca API is not reachable. Running browser analysis instead.");
      offlineError.browserFallback = true;
      throw offlineError;
    }
    throw error;
  }
}

function setView(view) {
  activeView = view;
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  document.querySelectorAll(".view").forEach((item) => item.classList.toggle("active", item.id === view));
  viewTitleEl.textContent = viewLabels[view][0];
  viewSubtitleEl.textContent = viewLabels[view][1];
  drawPreview();
}

function updateMetrics(health = null) {
  candidateCountEl.textContent = currentResult?.candidates.length ?? 0;
  deepCountEl.textContent = currentDeepResult?.nodes_searched ?? 0;
  if (health) {
    reviewCountEl.textContent = health.pending_reviews;
    modelStateEl.textContent = health.coco_baseline?.trained ? "COCO" : "Local";
  }
}

async function refreshHealth() {
  try {
    const health = await apiFetch("/api/health");
    updateMetrics(health);
  } catch {
    modelStateEl.textContent = LOCAL_MODEL_LABEL;
  }
}

async function refreshReviews() {
  let reviews = [];
  try {
    reviews = await apiFetch("/api/reviews");
  } catch {
    reviewsEl.innerHTML = '<p class="empty">Reviews unavailable while the API is offline.</p>';
    return;
  }
  reviewCountEl.textContent = reviews.filter((review) => review.status === "pending").length;
  if (!reviews.length) {
    reviewsEl.innerHTML = '<p class="empty">No review items.</p>';
    return;
  }
  reviewsEl.innerHTML = reviews
    .map((review) => {
      const candidate = review.candidate;
      return `
        <article class="review">
          <header>
            <strong>${review.status}</strong>
            <span class="score">${candidate.anomaly_score.toFixed(2)}</span>
          </header>
          <div class="meta">
            <span>${review.image.filename}</span>
            <span>${candidate.features.descriptor}</span>
            <span>${review.question}</span>
          </div>
          ${
            review.status === "pending"
              ? `<div class="review-actions">
                   <button data-review="${review.review_id}" data-status="approved">Approve</button>
                   <button class="reject" data-review="${review.review_id}" data-status="rejected">Reject</button>
                 </div>`
              : ""
          }
        </article>`;
    })
    .join("");
}

function renderOverview() {
  const candidates = currentResult?.candidates.length ?? 0;
  const deepNodes = currentDeepResult?.nodes_searched ?? 0;
  const report = currentDeepResult?.report || currentResult?.report || "No analysis run yet.";
  const summary = report.split("\n")[0];
  overviewEl.innerHTML = `
    <article class="overview-block">
      <strong>${currentImage ? imageMetaEl.textContent : "No image selected"}</strong>
      <span>${candidates} candidates</span>
      <span>${deepNodes} deep-search nodes</span>
    </article>
    <article class="overview-block">
      <strong>Latest summary</strong>
      <span>${escapeHtml(summary)}</span>
    </article>`;
}

function renderFindings(result) {
  if (!result?.candidates.length) {
    findingsEl.innerHTML = '<p class="empty">No findings.</p>';
    return;
  }
  findingsEl.innerHTML = result.candidates
    .map((candidate, index) => {
      const decision = result.decisions[candidate.candidate_id];
      const matches = result.similar_patterns[candidate.candidate_id] || [];
      const match = matches.length ? `${matches[0].label} · ${matches[0].similarity.toFixed(2)}` : "none";
      return `
        <article class="finding">
          <header>
            <strong>Candidate ${index + 1}</strong>
            <span class="score">${candidate.anomaly_score.toFixed(2)}</span>
          </header>
          <div class="meta">
            <span>Confidence ${candidate.confidence.toFixed(2)} · ${candidate.source_pass}</span>
            <span>COCO novelty: ${candidate.model_novelty == null ? "not trained" : candidate.model_novelty.toFixed(2)}</span>
            <span>${candidate.features.descriptor}</span>
            <span>Memory match: ${match}</span>
            <span class="decision">${decision.action}: ${decision.reason}</span>
          </div>
        </article>`;
    })
    .join("");
}

function renderDeepNode(node) {
  const candidate = node.candidate;
  const novelty = candidate.model_novelty == null ? "not trained" : candidate.model_novelty.toFixed(2);
  return `
    <article class="deep-node depth-${node.depth}">
      <header>
        <strong>Path ${node.path}</strong>
        <span class="score">${candidate.anomaly_score.toFixed(2)}</span>
      </header>
      <div class="meta">
        <span>Depth ${node.depth} · Confidence ${candidate.confidence.toFixed(2)} · Novelty ${novelty}</span>
        <span>${candidate.features.descriptor}</span>
      </div>
    </article>
    ${(node.children || []).map(renderDeepNode).join("")}`;
}

function renderDeep(result) {
  if (!result?.root_candidates.length) {
    deepEl.innerHTML = '<p class="empty">No deep-search tree yet.</p>';
    return;
  }
  deepEl.innerHTML = `
    <article class="overview-block">
      <strong>${result.nodes_searched} nodes searched</strong>
      <span>Depth limit ${result.max_depth}</span>
    </article>
    ${result.root_candidates.map(renderDeepNode).join("")}`;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[char]);
}

function selectedFormData() {
  const file = fileInput.files[0];
  if (!file) return null;
  const form = new FormData();
  form.append("file", file);
  return form;
}

function uid(prefix) {
  return `${prefix}-${Math.random().toString(16).slice(2)}-${Date.now().toString(16)}`;
}

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function imageSample(region = null, maxSide = 920) {
  const source = region || { x_min: 0, y_min: 0, x_max: currentImage.width, y_max: currentImage.height };
  const sourceWidth = Math.max(1, source.x_max - source.x_min);
  const sourceHeight = Math.max(1, source.y_max - source.y_min);
  const scale = Math.min(1, maxSide / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  const offscreen = document.createElement("canvas");
  offscreen.width = width;
  offscreen.height = height;
  const offscreenCtx = offscreen.getContext("2d", { willReadFrequently: true });
  offscreenCtx.drawImage(currentImage, source.x_min, source.y_min, sourceWidth, sourceHeight, 0, 0, width, height);
  return {
    data: offscreenCtx.getImageData(0, 0, width, height).data,
    width,
    height,
    source,
    scale,
  };
}

function tileStats(sample, tileSize = 38) {
  const { data, width, height } = sample;
  const gray = new Float32Array(width * height);
  let globalMean = 0;
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    const value = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    gray[p] = value;
    globalMean += value;
  }
  globalMean /= Math.max(1, gray.length);

  const tiles = [];
  for (let y = 0; y < height; y += tileSize) {
    for (let x = 0; x < width; x += tileSize) {
      const x1 = Math.min(width, x + tileSize);
      const y1 = Math.min(height, y + tileSize);
      let count = 0;
      let mean = 0;
      let variance = 0;
      let edge = 0;
      for (let yy = y; yy < y1; yy += 1) {
        for (let xx = x; xx < x1; xx += 1) {
          const index = yy * width + xx;
          const value = gray[index];
          mean += value;
          if (xx > 0) edge += Math.abs(value - gray[index - 1]);
          if (yy > 0) edge += Math.abs(value - gray[index - width]);
          count += 1;
        }
      }
      mean /= Math.max(1, count);
      for (let yy = y; yy < y1; yy += 1) {
        for (let xx = x; xx < x1; xx += 1) {
          const delta = gray[yy * width + xx] - mean;
          variance += delta * delta;
        }
      }
      variance /= Math.max(1, count);
      edge /= Math.max(1, count * 2);
      const localContrast = Math.sqrt(variance);
      const brightnessDelta = Math.abs(mean - globalMean);
      const score = clamp(0.36 * (localContrast / 70) + 0.34 * (edge / 38) + 0.2 * (brightnessDelta / 95) + 0.1 * (mean / 255));
      tiles.push({ x, y, x1, y1, mean, localContrast, edge, score });
    }
  }
  return tiles;
}

function mergeTiles(tiles, sample, limit = 8) {
  const threshold = Math.max(0.22, tiles.reduce((sum, tile) => sum + tile.score, 0) / Math.max(1, tiles.length) + 0.08);
  const selected = tiles
    .filter((tile) => tile.score >= threshold)
    .sort((first, second) => second.score - first.score)
    .slice(0, 80);
  const boxes = [];
  const gap = 44;

  selected.forEach((tile) => {
    const existing = boxes.find((box) => !(tile.x > box.x_max + gap || tile.x1 < box.x_min - gap || tile.y > box.y_max + gap || tile.y1 < box.y_min - gap));
    if (existing) {
      existing.x_min = Math.min(existing.x_min, tile.x);
      existing.y_min = Math.min(existing.y_min, tile.y);
      existing.x_max = Math.max(existing.x_max, tile.x1);
      existing.y_max = Math.max(existing.y_max, tile.y1);
      existing.score = Math.max(existing.score, tile.score);
      existing.tiles.push(tile);
    } else {
      boxes.push({ x_min: tile.x, y_min: tile.y, x_max: tile.x1, y_max: tile.y1, score: tile.score, tiles: [tile] });
    }
  });

  return boxes
    .map((box) => {
      const sourceScale = 1 / sample.scale;
      const source = sample.source;
      const width = Math.max(1, box.x_max - box.x_min);
      const height = Math.max(1, box.y_max - box.y_min);
      const avgContrast = box.tiles.reduce((sum, tile) => sum + tile.localContrast, 0) / box.tiles.length;
      const avgEdge = box.tiles.reduce((sum, tile) => sum + tile.edge, 0) / box.tiles.length;
      const bbox = {
        x_min: Math.max(0, Math.round(source.x_min + box.x_min * sourceScale)),
        y_min: Math.max(0, Math.round(source.y_min + box.y_min * sourceScale)),
        x_max: Math.min(currentImage.width, Math.round(source.x_min + box.x_max * sourceScale)),
        y_max: Math.min(currentImage.height, Math.round(source.y_min + box.y_max * sourceScale)),
      };
      return { ...box, bbox, area: width * height, avgContrast, avgEdge };
    })
    .filter((box) => box.bbox.x_max - box.bbox.x_min > 10 && box.bbox.y_max - box.bbox.y_min > 10)
    .sort((first, second) => second.area * second.score - first.area * first.score)
    .slice(0, limit);
}

function browserCandidate(box, index, sourcePass = "browser") {
  const score = clamp(box.score);
  const confidence = clamp(0.52 + score * 0.36 + Math.min(0.12, box.tiles.length / 80));
  const novelty = clamp(0.18 + score * 0.42);
  const descriptor = `${box.avgEdge > 18 ? "dense edges" : "soft edges"}, ${box.avgContrast > 35 ? "high contrast" : "low contrast"}, browser pattern region`;
  return {
    candidate_id: uid("browser-candidate"),
    bbox: box.bbox,
    anomaly_score: Number(score.toFixed(4)),
    confidence: Number(confidence.toFixed(4)),
    baseline_similarity: Number((1 - novelty).toFixed(4)),
    model_novelty: Number(novelty.toFixed(4)),
    features: {
      edge_density: Number(clamp(box.avgEdge / 60).toFixed(4)),
      contrast: Number(box.avgContrast.toFixed(3)),
      texture_entropy: Number(clamp(box.tiles.length / 12, 0, 8).toFixed(3)),
      dominant_color_rgb: [0, 0, 0],
      spatial_frequency: Number(box.avgEdge.toFixed(3)),
      descriptor,
    },
    embedding: [score, confidence, novelty, box.avgContrast / 100, box.avgEdge / 100, box.tiles.length / 100, index / 10, 1].map((value) => Number(clamp(value, 0, 1).toFixed(6))),
    source_pass: sourcePass,
  };
}

function browserCandidates(region = null, limit = 8, sourcePass = "browser") {
  const sample = imageSample(region);
  const tileSize = Math.max(28, Math.round(Math.min(sample.width, sample.height) / 18));
  return mergeTiles(tileStats(sample, tileSize), sample, limit).map((box, index) => browserCandidate(box, index + 1, sourcePass));
}

function browserAnalysisResult() {
  const candidates = browserCandidates(null, 8);
  const decisions = {};
  const similarPatterns = {};
  candidates.forEach((candidate) => {
    similarPatterns[candidate.candidate_id] = [];
    decisions[candidate.candidate_id] = {
      action: candidate.confidence > 0.66 ? "store_memory" : "ask_human",
      reason: "Browser-side pattern search completed without the FastAPI backend.",
      needs_human: candidate.confidence <= 0.66,
      uncertainty: Number((1 - Math.max(candidate.confidence, candidate.anomaly_score)).toFixed(4)),
    };
  });
  return {
    run_id: uid("browser-run"),
    image: {
      image_id: uid("browser-image"),
      filename: currentFileName || "browser-image",
      width: currentImage.width,
      height: currentImage.height,
      mode: "RGB",
      created_at: new Date().toISOString(),
    },
    candidates,
    similar_patterns: similarPatterns,
    decisions,
    report: `Browser analysis run\nImage: ${currentFileName || "browser-image"} (${currentImage.width}x${currentImage.height})\nCandidates found: ${candidates.length}`,
    created_at: new Date().toISOString(),
  };
}

function browserDeepNode(candidate, depth, maxDepth, path) {
  if (depth >= maxDepth) {
    return { node_id: uid("browser-node"), depth, path, candidate, children: [] };
  }
  const children = browserCandidates(candidate.bbox, 3, "refined")
    .filter((child) => child.anomaly_score >= 0.2)
    .map((child, index) => browserDeepNode(child, depth + 1, maxDepth, `${path}.${index + 1}`));
  return { node_id: uid("browser-node"), depth, path, candidate, children };
}

function browserDeepResult(maxDepth) {
  const roots = browserCandidates(null, 4).map((candidate, index) => browserDeepNode(candidate, 0, maxDepth, String(index + 1)));
  const nodes = flattenDeepNodes(roots);
  return {
    run_id: uid("browser-deep"),
    image: {
      image_id: uid("browser-image"),
      filename: currentFileName || "browser-image",
      width: currentImage.width,
      height: currentImage.height,
      mode: "RGB",
      created_at: new Date().toISOString(),
    },
    max_depth: maxDepth,
    nodes_searched: nodes.length,
    root_candidates: roots,
    report: `Browser deep search run\nImage: ${currentFileName || "browser-image"} (${currentImage.width}x${currentImage.height})\nDepth limit: ${maxDepth}; nodes searched: ${nodes.length}`,
    created_at: new Date().toISOString(),
  };
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  currentResult = null;
  currentDeepResult = null;
  setZoom(1);
  drawProfile();
  renderFindings(null);
  renderDeep(null);
  reportEl.textContent = "";
  updateMetrics();
  if (!file) {
    currentImage = null;
    currentFileName = "";
    imageMetaEl.textContent = "No image selected";
    drawPreview();
    renderOverview();
    return;
  }
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    currentFileName = file.name;
    imageMetaEl.textContent = `${file.name} · ${img.width}×${img.height}`;
    drawPreview();
    renderOverview();
  };
  img.src = URL.createObjectURL(file);
});

analyzeButton.addEventListener("click", async () => {
  const form = selectedFormData();
  if (!form) return;
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing";
  try {
    currentResult = await apiFetch("/api/analyze", { method: "POST", body: form });
  } catch (error) {
    if (!error.browserFallback || !currentImage) {
      findingsEl.innerHTML = `<p class="empty">${escapeHtml(error.message)}</p>`;
      setView("findings");
      return;
    }
    currentResult = browserAnalysisResult();
    modelStateEl.textContent = LOCAL_MODEL_LABEL;
  }
  try {
    renderFindings(currentResult);
    reportEl.textContent = currentResult.report;
    drawProfile(profileItemsFromResult(currentResult));
    renderOverview();
    updateMetrics();
    setView("findings");
    await refreshHealth();
    await refreshReviews();
  } catch (error) {
    findingsEl.innerHTML = `<p class="empty">${escapeHtml(error.message)}</p>`;
    setView("findings");
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze";
  }
});

deepButton.addEventListener("click", async () => {
  const form = selectedFormData();
  if (!form) return;
  deepButton.disabled = true;
  deepButton.textContent = "Searching";
  try {
    const depth = Math.max(1, Math.min(5, Number(depthInput.value || 3)));
    currentDeepResult = await apiFetch(`/api/deep-analyze?max_depth=${depth}`, { method: "POST", body: form });
  } catch (error) {
    if (!error.browserFallback || !currentImage) {
      deepEl.innerHTML = `<p class="empty">${escapeHtml(error.message)}</p>`;
      setView("deep");
      return;
    }
    const depth = Math.max(1, Math.min(5, Number(depthInput.value || 3)));
    currentDeepResult = browserDeepResult(depth);
    modelStateEl.textContent = LOCAL_MODEL_LABEL;
  }
  try {
    renderDeep(currentDeepResult);
    reportEl.textContent = currentDeepResult.report;
    drawProfile(profileItemsFromDeep(currentDeepResult));
    renderOverview();
    updateMetrics();
    setView("deep");
    await refreshHealth();
  } catch (error) {
    deepEl.innerHTML = `<p class="empty">${escapeHtml(error.message)}</p>`;
    setView("deep");
  } finally {
    deepButton.disabled = false;
    deepButton.textContent = "Deep Search";
  }
});

document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", () => setView(item.dataset.view));
});

sidebarToggle.addEventListener("click", () => {
  document.querySelector(".app-shell").classList.toggle("sidebar-hidden");
});

zoomOutButton.addEventListener("click", () => setZoom(zoom / 1.25));
zoomInButton.addEventListener("click", () => setZoom(zoom * 1.25));
zoomResetButton.addEventListener("click", () => setZoom(1));

canvas.addEventListener("wheel", (event) => {
  if (!currentImage) return;
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const anchor = {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
  setZoom(event.deltaY < 0 ? zoom * 1.15 : zoom / 1.15, anchor);
});

canvas.addEventListener("pointerdown", (event) => {
  if (!currentImage || zoom === 1) return;
  isPanning = true;
  canvas.setPointerCapture(event.pointerId);
  panStart = { x: event.clientX, y: event.clientY, panX, panY };
});

canvas.addEventListener("pointermove", (event) => {
  if (!isPanning) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  panX = panStart.panX + (event.clientX - panStart.x) * scaleX;
  panY = panStart.panY + (event.clientY - panStart.y) * scaleY;
  drawPreview();
});

canvas.addEventListener("pointerup", () => {
  isPanning = false;
});

canvas.addEventListener("pointercancel", () => {
  isPanning = false;
});

reviewsEl.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-review]");
  if (!button) return;
  await apiFetch(`/api/reviews/${button.dataset.review}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      status: button.dataset.status,
      answer: button.dataset.status === "approved" ? "Approved from dashboard" : "Rejected from dashboard",
      label: "human-approved-pattern",
    }),
  });
  await refreshHealth();
  await refreshReviews();
});

drawEmpty();
drawProfile();
setZoom(1);
renderOverview();
renderFindings(null);
renderDeep(null);
refreshHealth();
refreshReviews();

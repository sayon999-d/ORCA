const fileInput = document.querySelector("#fileInput");
const depthInput = document.querySelector("#depthInput");
const searchPromptInput = document.querySelector("#searchPromptInput");
const analyzeButton = document.querySelector("#analyzeButton");
const deepButton = document.querySelector("#deepButton");
const saveSessionButton = document.querySelector("#saveSessionButton");
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
const timelineEl = document.querySelector("#timeline");
const clustersEl = document.querySelector("#clusters");
const evidenceEl = document.querySelector("#evidence");
const calibrationEl = document.querySelector("#calibration");
const compareEl = document.querySelector("#compare");
const datasetEl = document.querySelector("#dataset");
const sessionsEl = document.querySelector("#sessions");
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
let currentTimeline = [];
let selectedCandidateId = null;
let lastAnalysisMode = "idle";
let activeView = "overview";
let zoom = 1;
let panX = 0;
let panY = 0;
let isPanning = false;
let panStart = { x: 0, y: 0, panX: 0, panY: 0 };

const API_BASE = window.ORCA_API_BASE || localStorage.getItem("ORCA_API_BASE") || "";
const LOCAL_MODEL_LABEL = "Browser";
const MAX_API_UPLOAD_BYTES = 3_200_000;
const MAX_API_IMAGE_SIDE = 1800;
const STORE_KEYS = {
  memory: "orca.patternMemory.v1",
  calibration: "orca.calibration.v1",
  sessions: "orca.sessions.v1",
  dataset: "orca.dataset.v1",
};
const LOCAL_API_HOSTS = new Set(["localhost", "127.0.0.1", "0.0.0.0", "::1"]);
const HAS_CONFIGURED_API = API_BASE.trim().length > 0;
const CAN_USE_SAME_ORIGIN_API = ["http:", "https:"].includes(window.location.protocol) && LOCAL_API_HOSTS.has(window.location.hostname);

const viewLabels = {
  overview: ["Overview", "Current analysis state"],
  findings: ["Findings", "Candidate evidence"],
  deep: ["Deep Search", "Recursive search tree"],
  timeline: ["Timeline", "Auditable investigation path"],
  clusters: ["Clusters", "Recurring pattern groups"],
  evidence: ["Evidence", "Candidate crop analysis"],
  calibration: ["Calibration", "Feedback for confidence"],
  compare: ["Compare", "Analyzer comparison"],
  dataset: ["Dataset", "Training export builder"],
  sessions: ["Sessions", "Saved project states"],
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

function readStore(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key) || JSON.stringify(fallback));
  } catch {
    return fallback;
  }
}

function writeStore(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

function allCandidates() {
  const primary = currentResult?.candidates || [];
  const deep = flattenDeepNodes(currentDeepResult?.root_candidates || []).map((node) => node.candidate);
  const byId = new Map();
  [...primary, ...deep].forEach((candidate) => byId.set(candidate.candidate_id, candidate));
  return [...byId.values()];
}

function candidateById(candidateId) {
  return allCandidates().find((candidate) => candidate.candidate_id === candidateId) || allCandidates()[0] || null;
}

function localMemoryMatches(candidate, limit = 3) {
  return readStore(STORE_KEYS.memory, [])
    .map((item) => ({
      label: item.labels?.[0] || item.label || "unlabeled pattern",
      similarity: cosineSimilarity(item.embedding || [], candidate.embedding || []),
    }))
    .filter((item) => item.similarity > 0.78)
    .sort((first, second) => second.similarity - first.similarity)
    .slice(0, limit);
}

function cosineSimilarity(first = [], second = []) {
  const length = Math.min(first.length, second.length);
  let dot = 0;
  let firstNorm = 0;
  let secondNorm = 0;
  for (let index = 0; index < length; index += 1) {
    dot += first[index] * second[index];
    firstNorm += first[index] * first[index];
    secondNorm += second[index] * second[index];
  }
  const denominator = Math.sqrt(firstNorm) * Math.sqrt(secondNorm);
  return denominator ? dot / denominator : 0;
}

function promptScore(candidate) {
  const prompt = searchPromptInput.value.trim().toLowerCase();
  if (!prompt) return 0;
  const haystack = [
    candidate.features?.descriptor || "",
    candidate.model_novelty != null && candidate.model_novelty > 0.2 ? "novel unusual anomaly" : "",
    candidate.confidence > 0.7 ? "confident clear" : "uncertain",
    candidate.features?.edge_density > 0.14 ? "dense edges road grid fracture line network" : "",
    candidate.features?.contrast > 45 ? "bright high contrast city light defect" : "",
  ]
    .join(" ")
    .toLowerCase();
  const words = prompt.split(/\s+/).filter((word) => word.length > 2);
  if (!words.length) return 0;
  const matches = words.filter((word) => haystack.includes(word)).length;
  return matches / words.length;
}

function applyPromptBias(candidates) {
  return [...candidates].sort((first, second) => {
    const firstScore = first.anomaly_score + promptScore(first) * 0.18;
    const secondScore = second.anomaly_score + promptScore(second) * 0.18;
    return secondScore - firstScore;
  });
}

function investigationStep(type, title, detail, candidate = null) {
  currentTimeline.push({
    id: uid("step"),
    type,
    title,
    detail,
    candidate_id: candidate?.candidate_id || null,
    score: candidate?.anomaly_score ?? null,
    confidence: candidate?.confidence ?? null,
    created_at: new Date().toISOString(),
  });
}

function rememberCandidates(candidates, label = "unlabeled pattern") {
  const memory = readStore(STORE_KEYS.memory, []);
  candidates.forEach((candidate) => {
    const existing = memory.find((item) => cosineSimilarity(item.embedding, candidate.embedding) > 0.9);
    if (existing) {
      existing.count += 1;
      existing.last_seen = new Date().toISOString();
      existing.score = Math.max(existing.score, candidate.anomaly_score);
      existing.labels = Array.from(new Set([...(existing.labels || []), label].filter(Boolean)));
    } else {
      memory.push({
        id: uid("memory"),
        label,
        labels: label ? [label] : [],
        count: 1,
        score: candidate.anomaly_score,
        confidence: candidate.confidence,
        embedding: candidate.embedding,
        descriptor: candidate.features?.descriptor || "",
        created_at: new Date().toISOString(),
        last_seen: new Date().toISOString(),
      });
    }
  });
  writeStore(STORE_KEYS.memory, memory);
}

function addDatasetItem(candidate, label, split = "uncertain") {
  const dataset = readStore(STORE_KEYS.dataset, []);
  dataset.push({
    id: uid("dataset"),
    image: currentFileName || "image",
    label,
    split,
    bbox: candidate.bbox,
    score: candidate.anomaly_score,
    confidence: candidate.confidence,
    descriptor: candidate.features?.descriptor || "",
    created_at: new Date().toISOString(),
  });
  writeStore(STORE_KEYS.dataset, dataset);
}

function addCalibration(candidate, status) {
  const calibration = readStore(STORE_KEYS.calibration, []);
  calibration.push({
    id: uid("calibration"),
    candidate_id: candidate.candidate_id,
    image: currentFileName || "image",
    status,
    score: candidate.anomaly_score,
    confidence: candidate.confidence,
    created_at: new Date().toISOString(),
  });
  writeStore(STORE_KEYS.calibration, calibration);
  if (status === "true_positive") addDatasetItem(candidate, "accepted-pattern", "positive");
  if (status === "false_positive") addDatasetItem(candidate, "rejected-pattern", "negative");
  if (status === "uncertain") addDatasetItem(candidate, "uncertain-pattern", "uncertain");
  if (status === "ignored") addDatasetItem(candidate, "ignored-pattern", "ignored");
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
  if (!HAS_CONFIGURED_API && !CAN_USE_SAME_ORIGIN_API) {
    throw browserFallbackError("Static deployment detected. Running browser analysis instead.");
  }
  try {
    const res = await fetch(`${API_BASE}${path}`, options);
    if (!res.ok) {
      if (res.status === 413) {
        throw browserFallbackError("Hosted API rejected the image size. Running browser analysis instead.");
      }
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
      throw browserFallbackError("Orca API is not reachable. Running browser analysis instead.");
    }
    throw error;
  }
}

function canvasToBlob(sourceCanvas, type = "image/jpeg", quality = 0.82) {
  return new Promise((resolve) => sourceCanvas.toBlob(resolve, type, quality));
}

function apiUploadName(file) {
  const stem = (file.name || "orca-image").replace(/\.[^.]+$/, "");
  return `${stem}-orca-api.jpg`;
}

async function apiUploadPayload(file) {
  if (!currentImage) {
    return { blob: file, filename: file.name || "image", compressed: false, scale: 1 };
  }

  const longestSide = Math.max(currentImage.width, currentImage.height);
  const needsResize = longestSide > MAX_API_IMAGE_SIDE;
  const needsCompression = file.size > MAX_API_UPLOAD_BYTES || needsResize;
  if (!needsCompression) {
    return { blob: file, filename: file.name || "image", compressed: false, scale: 1 };
  }

  const sideAttempts = [MAX_API_IMAGE_SIDE, 1500, 1200];
  const qualityAttempts = [0.82, 0.72, 0.62];
  let best = null;

  for (const maxSide of sideAttempts) {
    const scale = Math.min(1, maxSide / longestSide);
    const uploadCanvas = document.createElement("canvas");
    uploadCanvas.width = Math.max(1, Math.round(currentImage.width * scale));
    uploadCanvas.height = Math.max(1, Math.round(currentImage.height * scale));
    const uploadCtx = uploadCanvas.getContext("2d");
    uploadCtx.drawImage(currentImage, 0, 0, uploadCanvas.width, uploadCanvas.height);

    for (const quality of qualityAttempts) {
      const blob = await canvasToBlob(uploadCanvas, "image/jpeg", quality);
      if (!blob) continue;
      best = { blob, filename: apiUploadName(file), compressed: true, scale, bytes: blob.size };
      if (blob.size <= MAX_API_UPLOAD_BYTES) return best;
    }
  }

  return best || { blob: file, filename: file.name || "image", compressed: false, scale: 1 };
}

function scaleBoxToOriginal(box, scale) {
  if (!currentImage || !scale || scale === 1) return box;
  const factor = 1 / scale;
  return {
    x_min: Math.max(0, Math.round(box.x_min * factor)),
    y_min: Math.max(0, Math.round(box.y_min * factor)),
    x_max: Math.min(currentImage.width, Math.round(box.x_max * factor)),
    y_max: Math.min(currentImage.height, Math.round(box.y_max * factor)),
  };
}

function normalizeCandidateCoordinates(candidate, payload) {
  if (!payload?.compressed || !payload.scale || payload.scale === 1) return candidate;
  return { ...candidate, bbox: scaleBoxToOriginal(candidate.bbox, payload.scale) };
}

function normalizeAnalysisResult(result, payload) {
  if (!payload?.compressed) return result;
  return {
    ...result,
    image: { ...result.image, width: currentImage.width, height: currentImage.height },
    candidates: result.candidates.map((candidate) => normalizeCandidateCoordinates(candidate, payload)),
  };
}

function normalizeDeepNodeCoordinates(node, payload) {
  if (!payload?.compressed) return node;
  return {
    ...node,
    candidate: normalizeCandidateCoordinates(node.candidate, payload),
    children: (node.children || []).map((child) => normalizeDeepNodeCoordinates(child, payload)),
  };
}

function normalizeDeepResult(result, payload) {
  if (!payload?.compressed) return result;
  return {
    ...result,
    image: { ...result.image, width: currentImage.width, height: currentImage.height },
    root_candidates: result.root_candidates.map((node) => normalizeDeepNodeCoordinates(node, payload)),
  };
}

function browserFallbackError(message) {
  const offlineError = new Error(message);
  offlineError.browserFallback = true;
  return offlineError;
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

function refreshAllViews() {
  renderOverview();
  renderTimeline();
  renderClusters();
  renderEvidence();
  renderCalibration();
  renderCompare();
  renderDataset();
  renderSessions();
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
    reviewsEl.innerHTML = '<p class="empty">Server review queue is unavailable in browser mode.</p>';
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

function renderTimeline() {
  if (!currentTimeline.length) {
    timelineEl.innerHTML = '<p class="empty">No investigation timeline yet.</p>';
    return;
  }
  timelineEl.innerHTML = currentTimeline
    .map((step, index) => `
      <article class="timeline-step">
        <strong>${index + 1}. ${escapeHtml(step.title)}</strong>
        <span>${escapeHtml(step.detail)}</span>
        ${step.score == null ? "" : `<span>score ${step.score.toFixed(2)} · confidence ${step.confidence.toFixed(2)}</span>`}
      </article>`)
    .join("");
}

function renderClusters() {
  const memory = readStore(STORE_KEYS.memory, []);
  if (!memory.length) {
    clustersEl.innerHTML = '<p class="empty">No recurring pattern clusters yet. Run analysis or label findings to build memory.</p>';
    return;
  }
  clustersEl.innerHTML = memory
    .sort((first, second) => second.count - first.count)
    .map((cluster) => `
      <article class="cluster-card">
        <header>
          <strong>${escapeHtml(cluster.labels?.[0] || cluster.label || "unlabeled pattern")}</strong>
          <span class="score">${cluster.count} seen</span>
        </header>
        <div class="meta">
          <span>${escapeHtml(cluster.descriptor || "No descriptor")}</span>
          <span>best score ${Number(cluster.score || 0).toFixed(2)} · confidence ${Number(cluster.confidence || 0).toFixed(2)}</span>
          <span>last seen ${new Date(cluster.last_seen).toLocaleString()}</span>
        </div>
        <div class="tag-list">${(cluster.labels || []).map((label) => `<span class="tag">${escapeHtml(label)}</span>`).join("")}</div>
      </article>`)
    .join("");
}

function renderEvidence() {
  const candidate = candidateById(selectedCandidateId);
  if (!candidate || !currentImage) {
    evidenceEl.innerHTML = '<p class="empty">Select or run a candidate to inspect evidence crops.</p>';
    return;
  }
  const matchingNodes = flattenDeepNodes(currentDeepResult?.root_candidates || []).filter((node) => node.candidate.candidate_id === candidate.candidate_id);
  evidenceEl.innerHTML = `
    <article class="evidence-card">
      <strong>Candidate evidence</strong>
      <span class="meta">score ${candidate.anomaly_score.toFixed(2)} · confidence ${candidate.confidence.toFixed(2)}</span>
      <div class="evidence-grid">
        <div><span class="meta">Original crop</span><canvas id="evidenceOriginal" width="320" height="220"></canvas></div>
        <div><span class="meta">Enhanced crop</span><canvas id="evidenceEnhanced" width="320" height="220"></canvas></div>
        <div><span class="meta">Edge map</span><canvas id="evidenceEdges" width="320" height="220"></canvas></div>
        <div><span class="meta">Heatmap</span><canvas id="evidenceHeatmap" width="320" height="220"></canvas></div>
      </div>
      <div class="meta">
        <strong>Deep-search result</strong>
        <span>${matchingNodes.length ? `${matchingNodes.length} related deep node(s) available` : "Run Deep Search to attach recursive evidence."}</span>
      </div>
    </article>`;
  drawEvidenceCanvases(candidate);
}

function renderCalibration() {
  const candidate = candidateById(selectedCandidateId);
  const calibration = readStore(STORE_KEYS.calibration, []);
  const counts = calibration.reduce((acc, item) => {
    acc[item.status] = (acc[item.status] || 0) + 1;
    return acc;
  }, {});
  calibrationEl.innerHTML = `
    <article class="calibration-card">
      <strong>Confidence calibration</strong>
      <span class="meta">True ${counts.true_positive || 0} · False ${counts.false_positive || 0} · Uncertain ${counts.uncertain || 0} · Ignored ${counts.ignored || 0}</span>
      ${
        candidate
          ? `<div class="action-row">
              <button data-calibration="true_positive">True positive</button>
              <button data-calibration="false_positive">False positive</button>
              <button data-calibration="uncertain">Uncertain</button>
              <button data-calibration="ignored">Ignored</button>
            </div>`
          : '<span class="empty">No selected candidate.</span>'
      }
    </article>
    <div class="calibration-grid">
      ${["true_positive", "false_positive", "uncertain", "ignored"].map((status) => `
        <article class="calibration-card">
          <strong>${status.replace("_", " ")}</strong>
          <span>${counts[status] || 0} records</span>
        </article>`).join("")}
    </div>`;
}

function renderCompare() {
  const candidates = allCandidates();
  const browserAvg = candidates.length ? candidates.reduce((sum, item) => sum + item.anomaly_score, 0) / candidates.length : 0;
  const confidenceAvg = candidates.length ? candidates.reduce((sum, item) => sum + item.confidence, 0) / candidates.length : 0;
  const noveltyAvg = candidates.length ? candidates.reduce((sum, item) => sum + (item.model_novelty || 0), 0) / candidates.length : 0;
  const calibration = readStore(STORE_KEYS.calibration, []);
  const trueCount = calibration.filter((item) => item.status === "true_positive").length;
  const falseCount = calibration.filter((item) => item.status === "false_positive").length;
  const calibratedLift = calibration.length ? (trueCount - falseCount) / calibration.length : 0;
  compareEl.innerHTML = `
    <article class="overview-block">
      <strong>Model comparison</strong>
      <span>Active mode: ${escapeHtml(lastAnalysisMode)}</span>
      <span>Browser analyzer average score: ${browserAvg.toFixed(2)}</span>
      <span>FastAPI/OpenCV: ${API_BASE ? "configured" : "not configured"}</span>
      <span>COCO novelty average: ${noveltyAvg.toFixed(2)}</span>
      <span>Confidence average: ${confidenceAvg.toFixed(2)}</span>
      <span>Calibration lift: ${calibratedLift >= 0 ? "+" : ""}${calibratedLift.toFixed(2)}</span>
      <span>Future custom model: ready slot</span>
    </article>`;
}

function renderDataset() {
  const dataset = readStore(STORE_KEYS.dataset, []);
  datasetEl.innerHTML = `
    <article class="dataset-card">
      <header>
        <strong>Dataset builder</strong>
        <span class="score">${dataset.length} items</span>
      </header>
      <div class="action-row">
        <button id="exportJsonButton">Export JSON</button>
        <button id="exportCsvButton">Export CSV</button>
        <button id="exportYoloButton">Export YOLO</button>
        <button id="exportCocoButton">Export COCO</button>
        <button id="exportPngButton">Annotated PNG</button>
        <button id="exportPdfButton">PDF report</button>
      </div>
    </article>
    ${dataset.slice(-12).reverse().map((item) => `
      <article class="dataset-card">
        <strong>${escapeHtml(item.label)}</strong>
        <span class="meta">${escapeHtml(item.split)} · ${escapeHtml(item.image)} · score ${Number(item.score).toFixed(2)}</span>
      </article>`).join("")}`;
}

function drawCropToCanvas(targetCanvas, candidate, mode = "original") {
  const targetCtx = targetCanvas.getContext("2d");
  const box = candidate.bbox;
  const width = Math.max(1, box.x_max - box.x_min);
  const height = Math.max(1, box.y_max - box.y_min);
  targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  targetCtx.drawImage(currentImage, box.x_min, box.y_min, width, height, 0, 0, targetCanvas.width, targetCanvas.height);
  const imageData = targetCtx.getImageData(0, 0, targetCanvas.width, targetCanvas.height);
  const data = imageData.data;
  if (mode === "enhanced") {
    for (let i = 0; i < data.length; i += 4) {
      data[i] = clamp((data[i] - 35) * 1.35 + 35, 0, 255);
      data[i + 1] = clamp((data[i + 1] - 35) * 1.35 + 35, 0, 255);
      data[i + 2] = clamp((data[i + 2] - 35) * 1.35 + 35, 0, 255);
    }
  }
  if (mode === "edges" || mode === "heatmap") {
    for (let y = 0; y < targetCanvas.height; y += 1) {
      for (let x = 0; x < targetCanvas.width; x += 1) {
        const i = (y * targetCanvas.width + x) * 4;
        const current = (data[i] + data[i + 1] + data[i + 2]) / 3;
        const rightIndex = (y * targetCanvas.width + Math.min(targetCanvas.width - 1, x + 1)) * 4;
        const downIndex = (Math.min(targetCanvas.height - 1, y + 1) * targetCanvas.width + x) * 4;
        const right = (data[rightIndex] + data[rightIndex + 1] + data[rightIndex + 2]) / 3;
        const down = (data[downIndex] + data[downIndex + 1] + data[downIndex + 2]) / 3;
        const edge = clamp((Math.abs(current - right) + Math.abs(current - down)) / 90, 0, 1);
        if (mode === "edges") {
          const value = edge * 255;
          data[i] = value;
          data[i + 1] = value;
          data[i + 2] = value;
        } else {
          data[i] = Math.max(data[i], edge * 255);
          data[i + 1] *= 0.55;
          data[i + 2] *= 0.45;
        }
      }
    }
  }
  targetCtx.putImageData(imageData, 0, 0);
}

function drawEvidenceCanvases(candidate) {
  drawCropToCanvas(document.querySelector("#evidenceOriginal"), candidate, "original");
  drawCropToCanvas(document.querySelector("#evidenceEnhanced"), candidate, "enhanced");
  drawCropToCanvas(document.querySelector("#evidenceEdges"), candidate, "edges");
  drawCropToCanvas(document.querySelector("#evidenceHeatmap"), candidate, "heatmap");
}

function downloadFile(filename, content, type = "application/json") {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function exportJson() {
  downloadFile("orca-evidence.json", JSON.stringify({ result: currentResult, deep: currentDeepResult, timeline: currentTimeline, dataset: readStore(STORE_KEYS.dataset, []) }, null, 2));
}

function exportCsv() {
  const rows = [["image", "label", "split", "x_min", "y_min", "x_max", "y_max", "score", "confidence"]];
  const dataset = readStore(STORE_KEYS.dataset, []);
  const sourceRows = dataset.length
    ? dataset
    : allCandidates().map((candidate, index) => ({
        image: currentFileName || "image",
        label: `candidate-${index + 1}`,
        split: "candidate",
        bbox: candidate.bbox,
        score: candidate.anomaly_score,
        confidence: candidate.confidence,
      }));
  sourceRows.forEach((item) => {
    rows.push([item.image, item.label, item.split, item.bbox.x_min, item.bbox.y_min, item.bbox.x_max, item.bbox.y_max, item.score, item.confidence]);
  });
  downloadFile("orca-dataset.csv", rows.map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\n"), "text/csv");
}

function exportYolo() {
  const lines = readStore(STORE_KEYS.dataset, []).map((item) => {
    const box = item.bbox;
    const cx = ((box.x_min + box.x_max) / 2) / Math.max(1, currentImage?.width || 1);
    const cy = ((box.y_min + box.y_max) / 2) / Math.max(1, currentImage?.height || 1);
    const width = (box.x_max - box.x_min) / Math.max(1, currentImage?.width || 1);
    const height = (box.y_max - box.y_min) / Math.max(1, currentImage?.height || 1);
    return `0 ${cx.toFixed(6)} ${cy.toFixed(6)} ${width.toFixed(6)} ${height.toFixed(6)}`;
  });
  downloadFile("orca-yolo.txt", lines.join("\n"), "text/plain");
}

function exportCoco() {
  const dataset = readStore(STORE_KEYS.dataset, []);
  const images = [{ id: 1, file_name: currentFileName || "image", width: currentImage?.width || 0, height: currentImage?.height || 0 }];
  const annotations = dataset.map((item, index) => ({
    id: index + 1,
    image_id: 1,
    category_id: 1,
    bbox: [item.bbox.x_min, item.bbox.y_min, item.bbox.x_max - item.bbox.x_min, item.bbox.y_max - item.bbox.y_min],
    area: (item.bbox.x_max - item.bbox.x_min) * (item.bbox.y_max - item.bbox.y_min),
    iscrowd: 0,
  }));
  downloadFile("orca-coco.json", JSON.stringify({ images, annotations, categories: [{ id: 1, name: "pattern" }] }, null, 2));
}

function exportAnnotatedPng() {
  if (!currentImage) return;
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = canvas.width;
  exportCanvas.height = canvas.height;
  const exportCtx = exportCanvas.getContext("2d");
  exportCtx.drawImage(canvas, 0, 0);
  exportCanvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "orca-annotated.png";
    anchor.click();
    URL.revokeObjectURL(url);
  });
}

function exportPdfReport() {
  const popup = window.open("", "_blank");
  if (!popup) return;
  popup.document.write(`<pre style="font:14px/1.45 system-ui; white-space:pre-wrap">${escapeHtml(reportEl.textContent || "No report")}</pre>`);
  popup.document.close();
  popup.focus();
  popup.print();
}

function sessionThumbnail() {
  if (!currentImage) return "";
  const thumb = document.createElement("canvas");
  thumb.width = 220;
  thumb.height = 140;
  const thumbCtx = thumb.getContext("2d");
  thumbCtx.fillStyle = "#111317";
  thumbCtx.fillRect(0, 0, thumb.width, thumb.height);
  const scale = Math.min(thumb.width / currentImage.width, thumb.height / currentImage.height);
  const width = currentImage.width * scale;
  const height = currentImage.height * scale;
  thumbCtx.drawImage(currentImage, (thumb.width - width) / 2, (thumb.height - height) / 2, width, height);
  return thumb.toDataURL("image/jpeg", 0.72);
}

function sessionImageData() {
  if (!currentImage) return "";
  const maxSide = 1300;
  const scale = Math.min(1, maxSide / Math.max(currentImage.width, currentImage.height));
  const imageCanvas = document.createElement("canvas");
  imageCanvas.width = Math.max(1, Math.round(currentImage.width * scale));
  imageCanvas.height = Math.max(1, Math.round(currentImage.height * scale));
  const imageCtx = imageCanvas.getContext("2d");
  imageCtx.drawImage(currentImage, 0, 0, imageCanvas.width, imageCanvas.height);
  return imageCanvas.toDataURL("image/jpeg", 0.86);
}

function renderSessions() {
  const sessions = readStore(STORE_KEYS.sessions, []);
  sessionsEl.innerHTML = `
    <article class="session-card">
      <strong>Project sessions</strong>
      <span class="meta">${sessions.length} saved sessions</span>
    </article>
    ${sessions.slice().reverse().map((session) => `
      <article class="session-card">
        <header>
          <strong>${escapeHtml(session.name)}</strong>
          <span>${new Date(session.created_at).toLocaleString()}</span>
        </header>
        ${session.thumbnail ? `<img class="session-thumb" src="${session.thumbnail}" alt="" />` : ""}
        <div class="meta">
          <span>${session.candidates} candidates · ${session.deep_nodes} deep nodes</span>
          <span>${escapeHtml(session.notes || "No notes")}</span>
        </div>
        <div class="action-row">
          <button data-open-session="${session.id}">Open session</button>
        </div>
      </article>`).join("")}`;
}

function openSession(sessionId) {
  const sessions = readStore(STORE_KEYS.sessions, []);
  const session = sessions.find((item) => item.id === sessionId);
  if (!session) return;
  currentResult = session.result || null;
  currentDeepResult = session.deep || null;
  currentTimeline = session.timeline || [];
  currentFileName = session.name || "saved-session";
  selectedCandidateId = currentResult?.candidates?.[0]?.candidate_id || currentDeepResult?.root_candidates?.[0]?.candidate?.candidate_id || null;
  lastAnalysisMode = "Saved session";
  searchPromptInput.value = session.notes || "";
  if (session.image_data) {
    const image = new Image();
    image.onload = () => {
      currentImage = image;
      imageMetaEl.textContent = `${currentFileName} · ${image.width}×${image.height}`;
      setZoom(1);
      drawPreview();
      drawProfile(currentDeepResult ? profileItemsFromDeep(currentDeepResult) : profileItemsFromResult(currentResult));
      renderFindings(currentResult);
      renderDeep(currentDeepResult);
      reportEl.textContent = currentDeepResult?.report || currentResult?.report || "";
      refreshAllViews();
      updateMetrics();
      setView("overview");
    };
    image.src = session.image_data;
    return;
  }
  imageMetaEl.textContent = `${currentFileName} · image not embedded`;
  drawPreview();
  drawProfile(currentDeepResult ? profileItemsFromDeep(currentDeepResult) : profileItemsFromResult(currentResult));
  renderFindings(currentResult);
  renderDeep(currentDeepResult);
  reportEl.textContent = currentDeepResult?.report || currentResult?.report || "";
  refreshAllViews();
  updateMetrics();
  setView("overview");
}

function renderFindings(result) {
  if (!result?.candidates.length) {
    findingsEl.innerHTML = '<p class="empty">No findings.</p>';
    return;
  }
  findingsEl.innerHTML = result.candidates
    .map((candidate, index) => {
      const decision = result.decisions?.[candidate.candidate_id] || { action: "review", reason: "No decision metadata was attached." };
      const matches = result.similar_patterns?.[candidate.candidate_id] || [];
      const localMatches = localMemoryMatches(candidate);
      const mergedMatches = matches.length ? matches : localMatches;
      const match = mergedMatches.length ? `${mergedMatches[0].label} · ${mergedMatches[0].similarity.toFixed(2)}` : "none";
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
          <div class="field-row">
            <input data-label-for="${candidate.candidate_id}" type="text" placeholder="Rename pattern, e.g. road network" />
          </div>
          <div class="action-row">
            <button data-select-candidate="${candidate.candidate_id}">Inspect evidence</button>
            <button data-save-label="${candidate.candidate_id}">Save label</button>
            <button data-add-dataset="${candidate.candidate_id}" data-split="positive">Add positive</button>
            <button data-add-dataset="${candidate.candidate_id}" data-split="negative">Add negative</button>
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
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[char]);
}

async function selectedFormData() {
  const file = fileInput.files[0];
  if (!file) return null;
  const payload = await apiUploadPayload(file);
  const form = new FormData();
  form.append("file", payload.blob, payload.filename);
  return { form, payload };
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
  const candidates = applyPromptBias(browserCandidates(null, 8));
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
  const roots = applyPromptBias(browserCandidates(null, 4)).map((candidate, index) => browserDeepNode(candidate, 0, maxDepth, String(index + 1)));
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
  currentTimeline = [];
  selectedCandidateId = null;
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
  const selection = await selectedFormData();
  if (!selection) return;
  const { form, payload } = selection;
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing";
  currentTimeline = [];
  investigationStep("input", "Image uploaded", currentFileName || "Selected image");
  if (payload.compressed) {
    investigationStep("preprocess", "API upload compressed", `${Math.round(payload.bytes / 1024)} KB JPEG copy sent to avoid hosted payload limits`);
  }
  if (searchPromptInput.value.trim()) {
    investigationStep("prompt", "Open-vocabulary note applied", searchPromptInput.value.trim());
  }
  try {
    currentResult = normalizeAnalysisResult(await apiFetch("/api/analyze", { method: "POST", body: form }), payload);
    currentResult.candidates = applyPromptBias(currentResult.candidates);
    lastAnalysisMode = "FastAPI/OpenCV";
  } catch (error) {
    if (!error.browserFallback || !currentImage) {
      findingsEl.innerHTML = `<p class="empty">${escapeHtml(error.message)}</p>`;
      setView("findings");
      return;
    }
    currentResult = browserAnalysisResult();
    lastAnalysisMode = "Browser";
    modelStateEl.textContent = LOCAL_MODEL_LABEL;
  }
  try {
    investigationStep("perception", "First-pass pattern regions", `${currentResult.candidates.length} candidates found`);
    currentResult.candidates.forEach((candidate, index) => investigationStep("candidate", `Pattern region ${index + 1}`, candidate.features?.descriptor || "Pattern evidence", candidate));
    rememberCandidates(currentResult.candidates);
    selectedCandidateId = currentResult.candidates[0]?.candidate_id || null;
    renderFindings(currentResult);
    reportEl.textContent = currentResult.report;
    drawProfile(profileItemsFromResult(currentResult));
    refreshAllViews();
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
  const selection = await selectedFormData();
  if (!selection) return;
  const { form, payload } = selection;
  deepButton.disabled = true;
  deepButton.textContent = "Searching";
  investigationStep("deep_start", "Deep Search started", `Depth ${Math.max(1, Math.min(5, Number(depthInput.value || 3)))}`);
  if (payload.compressed) {
    investigationStep("preprocess", "API upload compressed", `${Math.round(payload.bytes / 1024)} KB JPEG copy sent to avoid hosted payload limits`);
  }
  try {
    const depth = Math.max(1, Math.min(5, Number(depthInput.value || 3)));
    currentDeepResult = normalizeDeepResult(await apiFetch(`/api/deep-analyze?max_depth=${depth}`, { method: "POST", body: form }), payload);
    const originalNodes = currentDeepResult.root_candidates || [];
    currentDeepResult.root_candidates = applyPromptBias(originalNodes.map((node) => node.candidate)).map((candidate, index) => {
      const node = originalNodes.find((item) => item.candidate.candidate_id === candidate.candidate_id);
      return { ...(node || { node_id: uid("node"), depth: 0, children: [] }), path: String(index + 1), candidate };
    });
    lastAnalysisMode = "FastAPI/OpenCV deep";
  } catch (error) {
    if (!error.browserFallback || !currentImage) {
      deepEl.innerHTML = `<p class="empty">${escapeHtml(error.message)}</p>`;
      setView("deep");
      return;
    }
    const depth = Math.max(1, Math.min(5, Number(depthInput.value || 3)));
    currentDeepResult = browserDeepResult(depth);
    lastAnalysisMode = "Browser deep";
    modelStateEl.textContent = LOCAL_MODEL_LABEL;
  }
  try {
    investigationStep("deep_result", "Deep Search tree built", `${currentDeepResult.nodes_searched} nodes searched`);
    flattenDeepNodes(currentDeepResult.root_candidates).forEach((node) => investigationStep("deep_node", `Deep node ${node.path}`, node.candidate.features?.descriptor || "Recursive evidence", node.candidate));
    rememberCandidates(flattenDeepNodes(currentDeepResult.root_candidates).map((node) => node.candidate));
    selectedCandidateId = selectedCandidateId || currentDeepResult.root_candidates[0]?.candidate?.candidate_id || null;
    renderDeep(currentDeepResult);
    reportEl.textContent = currentDeepResult.report;
    drawProfile(profileItemsFromDeep(currentDeepResult));
    refreshAllViews();
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

saveSessionButton.addEventListener("click", () => {
  const sessions = readStore(STORE_KEYS.sessions, []);
  const notes = window.prompt("Session notes", searchPromptInput.value.trim()) || "";
  const session = {
    id: uid("session"),
    name: currentFileName || `Session ${sessions.length + 1}`,
    notes,
    result: currentResult,
    deep: currentDeepResult,
    timeline: currentTimeline,
    thumbnail: sessionThumbnail(),
    image_data: sessionImageData(),
    candidates: currentResult?.candidates.length || 0,
    deep_nodes: currentDeepResult?.nodes_searched || 0,
    created_at: new Date().toISOString(),
  };
  sessions.push(session);
  if (!writeStore(STORE_KEYS.sessions, sessions)) {
    session.image_data = "";
    session.notes = `${notes}${notes ? " " : ""}(Full image restore omitted because browser storage is full.)`;
    writeStore(STORE_KEYS.sessions, sessions);
  }
  renderSessions();
  setView("sessions");
});

sessionsEl.addEventListener("click", (event) => {
  const button = event.target.closest("[data-open-session]");
  if (!button) return;
  openSession(button.dataset.openSession);
});

findingsEl.addEventListener("click", (event) => {
  const selectButton = event.target.closest("[data-select-candidate]");
  const saveButton = event.target.closest("[data-save-label]");
  const datasetButton = event.target.closest("[data-add-dataset]");
  if (selectButton) {
    selectedCandidateId = selectButton.dataset.selectCandidate;
    renderEvidence();
    setView("evidence");
  }
  if (saveButton) {
    const candidate = candidateById(saveButton.dataset.saveLabel);
    const input = findingsEl.querySelector(`[data-label-for="${saveButton.dataset.saveLabel}"]`);
    const label = input?.value.trim() || "reviewed pattern";
    if (candidate) {
      rememberCandidates([candidate], label);
      addDatasetItem(candidate, label, "positive");
      renderClusters();
      renderDataset();
    }
  }
  if (datasetButton) {
    const candidate = candidateById(datasetButton.dataset.addDataset);
    if (candidate) {
      addDatasetItem(candidate, datasetButton.dataset.split === "negative" ? "rejected-pattern" : "accepted-pattern", datasetButton.dataset.split);
      renderDataset();
    }
  }
});

calibrationEl.addEventListener("click", (event) => {
  const button = event.target.closest("[data-calibration]");
  const candidate = candidateById(selectedCandidateId);
  if (!button || !candidate) return;
  addCalibration(candidate, button.dataset.calibration);
  renderCalibration();
  renderDataset();
});

datasetEl.addEventListener("click", (event) => {
  if (event.target.closest("#exportJsonButton")) exportJson();
  if (event.target.closest("#exportCsvButton")) exportCsv();
  if (event.target.closest("#exportYoloButton")) exportYolo();
  if (event.target.closest("#exportCocoButton")) exportCoco();
  if (event.target.closest("#exportPngButton")) exportAnnotatedPng();
  if (event.target.closest("#exportPdfButton")) exportPdfReport();
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
refreshAllViews();
renderFindings(null);
renderDeep(null);
refreshHealth();
refreshReviews();

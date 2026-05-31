![Orca](assets/orca-banner.svg)

Orca is a model-driven image intelligence dashboard. It can run fully in the browser for static hosting, and it can optionally use the Python/OpenCV backend for stronger local analysis, COCO novelty scoring, memory, and human review.

![Orca pipeline architecture](assets/pipeline-architecture.svg)

## What It Includes

- Static browser-side analyzer for GitHub Pages or any static host.
- Optional typed Pydantic contracts for the Python pixel-to-agent bridge.
- OpenCV perception engine that groups nearby visual evidence into pattern regions, then returns bounding boxes, anomaly scores, visual features, and embeddings.
- Optional COCO-trained baseline model that learns normal image patch embeddings from `http://images.cocodataset.org/`.
- Recursive deep search that zooms into suspicious regions, enhances crops, and builds a typed search tree.
- Canvas zoom and pan controls for inspecting image evidence without zooming the browser page.
- Pattern profile graph under the inspection canvas for comparing score, confidence, and novelty.
- Investigation Timeline that records image upload, prompt biasing, first-pass regions, crop refinement, deep-search nodes, and final evidence.
- Pattern Clustering View backed by local vector-style memory so recurring unknown regions can be grouped over time.
- Human label feedback loop for renaming a pattern and storing that label for future matching.
- Before/After Evidence Viewer with original crop, enhanced crop, edge map, heatmap, and deep-search attachment.
- Confidence Calibration panel for marking true positives, false positives, uncertain findings, and ignored findings.
- Export tools for JSON evidence bundles, CSV candidate rows, YOLO labels, COCO annotations, annotated PNGs, and printable PDF reports.
- Project Sessions that save image state, findings, deep-search tree, timeline, and notes in browser storage.
- Open-vocabulary search notes that bias candidate ranking toward what the reviewer is searching for.
- Model Comparison panel for browser analysis, FastAPI/OpenCV analysis, COCO novelty, calibration lift, and future custom models.
- Dataset Builder that turns reviewed candidates into positive, negative, uncertain, and ignored training examples.
- Dynamic refinement for low-confidence candidates.
- JSONL vector memory with cosine similarity for recurring unknown patterns.
- Human-in-the-loop review queue.
- Browser dashboard for image upload, findings, reports, zoom, graphing, and review actions.
- Optional FastAPI backend for local or hosted API analysis.

## Run

### Static Browser Mode

Open `static/index.html` through GitHub Pages or any static host. Analyze and Deep Search run in the browser using Canvas image processing. On non-localhost static deployments, Orca does not call `/api/*` unless `ORCA_API_BASE` is explicitly configured.

Static mode supports:

- Image upload.
- Pattern-region detection.
- Deep Search.
- Canvas zoom and pan.
- Pattern profile graph.
- Investigation timeline.
- Pattern clustering.
- Label feedback.
- Evidence crop comparison.
- Calibration.
- Export bundles.
- Saved sessions.
- Dataset export.

### Optional Python API Mode

```bash
python scripts/create_sample_images.py
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000` and upload `data/samples/manufacturing_unknown_patterns.png`.

You can also run from inside `app/`:

```bash
python main.py
```

## Analyze Images

Use **Analyze** for the first-pass pattern search. Orca groups nearby points, edges, and high-contrast details into larger candidate pattern regions so satellite/city-light style images do not show only isolated point marks.

Use **Deep Search** for recursive inspection. Orca crops each suspicious region, enhances it, zooms in, and searches again up to the selected depth.

Use **Search note** to describe what you are looking for, such as `road-like grid patterns near bright clusters`. Orca uses the note as a ranking bias so relevant candidates appear first without requiring a fixed object class.

Use the canvas zoom controls beside **Inspection**:

- `+` zooms into the image.
- `-` zooms out.
- `100%` resets zoom and pan.
- Mouse wheel zooms over the canvas.
- Drag the canvas to pan when zoomed in.

If no API is reachable, Orca automatically falls back to browser-side analysis. To use a hosted API from a static deployment, set:

```js
localStorage.setItem("ORCA_API_BASE", "https://your-orca-api.example.com")
```

Refresh the page after setting `ORCA_API_BASE`.

## Investigation Workspace

The right-side workspace is split into focused inspection views:

- **Timeline**: an audit trail of each run, from upload to candidate discovery and deep-search traversal.
- **Clusters**: recurring unknown patterns grouped from saved embeddings and human labels.
- **Evidence**: candidate crops shown as original, enhanced, edge map, and heatmap views.
- **Calibration**: reviewer decisions that separate true positives from false positives and uncertain results.
- **Compare**: browser analyzer, FastAPI/OpenCV analyzer, COCO novelty, confidence, and calibration lift in one panel.
- **Dataset**: export accepted positives, rejected negatives, uncertain items, and ignored regions to JSON, CSV, YOLO, or COCO formats.
- **Sessions**: save and reopen an investigation with image preview, findings, deep-search tree, notes, and timeline.

## Deploy The FastAPI API To Vercel

The GitHub Pages dashboard is static. For the Python/OpenCV API, deploy the FastAPI backend separately to Vercel.

This repo includes:

- `api/index.py`: Vercel Python entrypoint that imports the FastAPI app.
- `vercel.json`: routes all Vercel traffic to the FastAPI app.
- `.vercelignore`: keeps local caches, uploads, and virtualenv files out of deployment.
- `static/orca-config.js`: static dashboard API configuration file.

Deploy:

```bash
vercel login
vercel --prod
```

After deployment, copy the production URL, for example:

```text
https://orca-api.vercel.app
```

Then connect GitHub Pages to that API by adding a GitHub repository variable:

```text
ORCA_API_BASE=https://orca-api.vercel.app
```

The Pages workflow writes that value into `static/orca-config.js` during deployment.

For stricter CORS on Vercel, set this Vercel environment variable:

```text
ORCA_ALLOWED_ORIGINS=https://sayon999-d.github.io
```

If `ORCA_API_BASE` is not set, Orca still works on GitHub Pages using browser-side Canvas analysis.

## Train A COCO Baseline

COCO is large, so start with a tiny subset:

```bash
python scripts/train_coco_baseline.py --download --count 8
```

This downloads a few images from `http://images.cocodataset.org/val2017/`, extracts patch embeddings, and writes `data/coco_baseline.json`. Restart the API after training so the agent uses the baseline.

Once trained, each candidate includes:

- `baseline_similarity`: nearest COCO patch similarity.
- `model_novelty`: how unlike the COCO baseline the candidate looks.

## Test

```bash
pytest
```

## Replace The CV Model

Swap `PerceptionEngine.analyze_array` in `app/perception.py` with your trained model inference. Keep returning `AnomalyCandidate` objects so the agent stays type-safe and independent from raw tensors.

The key contract is:

```python
AnomalyCandidate(
    bbox=BoundingBox(...),
    anomaly_score=0.0,
    confidence=0.0,
    features=VisualFeatures(...),
    embedding=[...],
)
```

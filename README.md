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

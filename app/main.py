from __future__ import annotations

import shutil
import sys
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import AnomalyInvestigator
from app.coco_baseline import CocoBaselineModel
from app.contracts import PatternMemoryRecord, ReviewStatus, ReviewUpdate
from app.memory import ReviewQueue, VectorMemory
from app.perception import PerceptionEngine

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path("/tmp/orca-data") if os.environ.get("VERCEL") else ROOT / "data"
DATA_DIR = Path(os.environ.get("ORCA_DATA_DIR", DEFAULT_DATA_DIR))
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

coco_baseline = CocoBaselineModel(DATA_DIR / "coco_baseline.json")
perception = PerceptionEngine(max_candidates=8, baseline_model=coco_baseline)
memory = VectorMemory(DATA_DIR / "pattern_memory.jsonl")
review_queue = ReviewQueue(DATA_DIR / "review_queue.json")
investigator = AnomalyInvestigator(perception, memory, review_queue)

app = FastAPI(title="Orca", version="0.1.0")
allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ORCA_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "memory_records": len(memory.all()),
        "pending_reviews": len(review_queue.pending()),
        "coco_baseline": coco_baseline.summary(),
    }


@app.get("/api/model")
def model_summary():
    return coco_baseline.summary()


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    target = UPLOAD_DIR / f"{Path(file.filename or 'image').stem}-{len(list(UPLOAD_DIR.iterdir())) + 1}{Path(file.filename or '.png').suffix}"
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    try:
        return investigator.analyze(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/deep-analyze")
async def deep_analyze_image(file: UploadFile = File(...), max_depth: int = 3):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")
    if max_depth < 1 or max_depth > 5:
        raise HTTPException(status_code=400, detail="max_depth must be between 1 and 5.")

    target = UPLOAD_DIR / f"{Path(file.filename or 'image').stem}-deep-{len(list(UPLOAD_DIR.iterdir())) + 1}{Path(file.filename or '.png').suffix}"
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    try:
        return perception.deep_analyze_path(target, max_depth=max_depth)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/memory")
def list_memory():
    return memory.all()


@app.get("/api/reviews")
def list_reviews():
    return review_queue.all()


@app.post("/api/reviews/{review_id}")
def resolve_review(review_id: str, update: ReviewUpdate):
    item = review_queue.resolve(review_id, update.status, update.answer)
    if not item:
        raise HTTPException(status_code=404, detail="Review item not found.")

    if update.status == ReviewStatus.approved:
        memory.upsert(
            PatternMemoryRecord(
                label=update.label or "human-approved-pattern",
                image_id=item.image.image_id,
                candidate_id=item.candidate.candidate_id,
                bbox=item.candidate.bbox,
                anomaly_score=item.candidate.anomaly_score,
                embedding=item.candidate.embedding,
                notes=update.answer or "Approved by human reviewer.",
                status=ReviewStatus.approved,
            )
        )
    return item


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

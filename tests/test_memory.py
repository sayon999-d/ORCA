from pathlib import Path

import _repo_path  # noqa: F401
from app.contracts import BoundingBox, PatternMemoryRecord, ReviewStatus
from app.memory import VectorMemory


def record(label: str, embedding: list[float]) -> PatternMemoryRecord:
    return PatternMemoryRecord(
        label=label,
        image_id="image-1",
        candidate_id="candidate-1",
        bbox=BoundingBox(x_min=1, y_min=2, x_max=20, y_max=30),
        anomaly_score=0.71,
        embedding=embedding,
        status=ReviewStatus.approved,
    )


def test_vector_memory_finds_similar_pattern(tmp_path: Path) -> None:
    memory = VectorMemory(tmp_path / "memory.jsonl", match_threshold=0.8)
    memory.upsert(record("scratch", [1, 0, 0, 0, 0, 0, 0, 0]))

    matches = memory.search([0.99, 0.01, 0, 0, 0, 0, 0, 0])

    assert matches
    assert matches[0].label == "scratch"


def test_vector_memory_merges_seen_count(tmp_path: Path) -> None:
    memory = VectorMemory(tmp_path / "memory.jsonl", match_threshold=0.8)
    memory.upsert(record("scratch", [1, 0, 0, 0, 0, 0, 0, 0]))
    merged = memory.upsert(record("scratch", [1, 0, 0, 0, 0, 0, 0, 0]))

    assert merged.seen_count == 2
    assert len(memory.all()) == 1

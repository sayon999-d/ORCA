from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.contracts import PatternMemoryRecord, ReviewStatus, SimilarPattern


class VectorMemory:
    def __init__(self, path: Path, match_threshold: float = 0.88) -> None:
        self.path = path
        self.match_threshold = match_threshold
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def all(self) -> list[PatternMemoryRecord]:
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(PatternMemoryRecord.model_validate_json(line))
        return records

    def search(self, embedding: list[float], limit: int = 5) -> list[SimilarPattern]:
        query = np.array(embedding, dtype=np.float32)
        matches = []
        for record in self.all():
            similarity = self._cosine(query, np.array(record.embedding, dtype=np.float32))
            if similarity >= self.match_threshold:
                matches.append(
                    SimilarPattern(
                        memory_id=record.memory_id,
                        label=record.label,
                        similarity=round(similarity, 4),
                        first_seen=record.first_seen,
                    )
                )
        matches.sort(key=lambda item: item.similarity, reverse=True)
        return matches[:limit]

    def upsert(self, record: PatternMemoryRecord) -> PatternMemoryRecord:
        records = self.all()
        best_index = None
        best_similarity = 0.0
        query = np.array(record.embedding, dtype=np.float32)
        for index, existing in enumerate(records):
            similarity = self._cosine(query, np.array(existing.embedding, dtype=np.float32))
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index

        if best_index is not None and best_similarity >= self.match_threshold:
            existing = records[best_index]
            merged = existing.model_copy(
                update={
                    "last_seen": datetime.now(timezone.utc),
                    "seen_count": existing.seen_count + 1,
                    "anomaly_score": max(existing.anomaly_score, record.anomaly_score),
                    "status": record.status if record.status != ReviewStatus.pending else existing.status,
                    "notes": record.notes or existing.notes,
                }
            )
            records[best_index] = merged
            self._write(records)
            return merged

        records.append(record)
        self._write(records)
        return record

    def update_status(self, memory_id: str, status: ReviewStatus, notes: str | None = None) -> PatternMemoryRecord | None:
        records = self.all()
        updated = None
        for index, record in enumerate(records):
            if record.memory_id == memory_id:
                updated = record.model_copy(update={"status": status, "notes": notes or record.notes})
                records[index] = updated
                break
        if updated:
            self._write(records)
        return updated

    def _write(self, records: list[PatternMemoryRecord]) -> None:
        payload = "\n".join(item.model_dump_json() for item in records)
        self.path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")

    def _cosine(self, first: np.ndarray, second: np.ndarray) -> float:
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator == 0:
            return 0.0
        return float(np.clip(np.dot(first, second) / denominator, 0, 1))


class ReviewQueue:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def all(self):
        from app.contracts import ReviewItem

        return [ReviewItem.model_validate(item) for item in json.loads(self.path.read_text(encoding="utf-8"))]

    def pending(self):
        return [item for item in self.all() if item.status == ReviewStatus.pending]

    def add(self, item):
        items = self.all()
        items.append(item)
        self._write(items)
        return item

    def resolve(self, review_id: str, status: ReviewStatus, answer: str | None = None):
        items = self.all()
        resolved = None
        for index, item in enumerate(items):
            if item.review_id == review_id:
                resolved = item.model_copy(
                    update={
                        "status": status,
                        "answer": answer,
                        "resolved_at": datetime.now(timezone.utc),
                    }
                )
                items[index] = resolved
                break
        if resolved:
            self._write(items)
        return resolved

    def _write(self, items) -> None:
        self.path.write_text(
            json.dumps([item.model_dump(mode="json") for item in items], indent=2),
            encoding="utf-8",
        )


from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReviewStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    ignored = "ignored"


class ActionType(str, Enum):
    report = "report"
    ask_human = "ask_human"
    store_memory = "store_memory"
    refine_image = "refine_image"
    ignore = "ignore"


class BoundingBox(BaseModel):
    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(gt=0)
    y_max: int = Field(gt=0)

    @field_validator("x_max")
    @classmethod
    def x_max_must_be_positive(cls, value: int) -> int:
        return value

    def width(self) -> int:
        return self.x_max - self.x_min

    def height(self) -> int:
        return self.y_max - self.y_min

    def area(self) -> int:
        return max(0, self.width()) * max(0, self.height())


class ImageMetadata(BaseModel):
    image_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mode: Literal["RGB", "BGR", "GRAY"]
    created_at: datetime = Field(default_factory=utc_now)


class VisualFeatures(BaseModel):
    edge_density: float = Field(ge=0, le=1)
    contrast: float = Field(ge=0)
    texture_entropy: float = Field(ge=0)
    dominant_color_rgb: tuple[int, int, int]
    spatial_frequency: float = Field(ge=0)
    descriptor: str


class AnomalyCandidate(BaseModel):
    candidate_id: str = Field(default_factory=lambda: str(uuid4()))
    bbox: BoundingBox
    anomaly_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    baseline_similarity: float | None = Field(default=None, ge=0, le=1)
    model_novelty: float | None = Field(default=None, ge=0, le=1)
    features: VisualFeatures
    embedding: list[float] = Field(min_length=8)
    source_pass: Literal["initial", "refined"] = "initial"


class SimilarPattern(BaseModel):
    memory_id: str
    label: str
    similarity: float = Field(ge=0, le=1)
    first_seen: datetime


class AgentDecision(BaseModel):
    action: ActionType
    reason: str
    needs_human: bool = False
    uncertainty: float = Field(ge=0, le=1)


class AnalysisResult(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    image: ImageMetadata
    candidates: list[AnomalyCandidate]
    similar_patterns: dict[str, list[SimilarPattern]]
    decisions: dict[str, AgentDecision]
    report: str
    created_at: datetime = Field(default_factory=utc_now)


class DeepSearchNode(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    depth: int = Field(ge=0)
    path: str
    candidate: AnomalyCandidate
    children: list["DeepSearchNode"] = Field(default_factory=list)


class DeepAnalysisResult(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    image: ImageMetadata
    max_depth: int = Field(ge=1, le=5)
    nodes_searched: int = Field(ge=0)
    root_candidates: list[DeepSearchNode]
    report: str
    created_at: datetime = Field(default_factory=utc_now)


class PatternMemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    image_id: str
    candidate_id: str
    bbox: BoundingBox
    anomaly_score: float = Field(ge=0, le=1)
    embedding: list[float] = Field(min_length=8)
    notes: str = ""
    status: ReviewStatus = ReviewStatus.pending
    first_seen: datetime = Field(default_factory=utc_now)
    last_seen: datetime = Field(default_factory=utc_now)
    seen_count: int = Field(default=1, ge=1)


class ReviewItem(BaseModel):
    review_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    image: ImageMetadata
    candidate: AnomalyCandidate
    similar_patterns: list[SimilarPattern]
    question: str
    status: ReviewStatus = ReviewStatus.pending
    answer: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None


class ReviewUpdate(BaseModel):
    status: ReviewStatus
    answer: str | None = None
    label: str | None = None

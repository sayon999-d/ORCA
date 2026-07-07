from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.contracts import AnomalyCandidate, BackendModelSummary, ImageMetadata


@dataclass(frozen=True)
class BackendModelConfig:
    model_name: str = "Orca Backend Model"
    model_version: str = "1.0"
    mode: str = "generic"


class BackendModelBridge:

    def __init__(self, *, space_mode: bool = False, config: BackendModelConfig | None = None) -> None:
        self.config = config or BackendModelConfig(mode="space" if space_mode else "generic")
        self.space_mode = space_mode

    def describe(self) -> dict[str, object]:
        return {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "mode": self.config.mode,
            "space_mode": self.space_mode,
            "capabilities": [
                "analysis-summary",
                "scene-estimation",
                "candidate-aggregation",
                "space-aware-routing" if self.space_mode else "generic-routing",
            ],
        }

    def summarize(
        self,
        image: np.ndarray,
        metadata: ImageMetadata,
        candidates: list[AnomalyCandidate],
    ) -> BackendModelSummary:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        mean_intensity = float(gray.mean() / 255.0) if gray.size else 0.0
        contrast = float(gray.std() / 128.0) if gray.size else 0.0
        edge_density = float(np.count_nonzero(cv2.Canny(gray, 40, 120)) / max(1, gray.size)) if gray.size else 0.0

        candidate_count = len(candidates)
        avg_score = float(np.mean([candidate.anomaly_score for candidate in candidates])) if candidates else 0.0
        avg_confidence = float(np.mean([candidate.confidence for candidate in candidates])) if candidates else 0.0

        signal_density = float(np.clip((edge_density * 1.7) + (candidate_count / max(1, (metadata.width * metadata.height) / 65000)), 0, 1))
        confidence = float(np.clip(0.35 + 0.3 * avg_confidence + 0.2 * avg_score + 0.15 * signal_density, 0, 1))

        top_labels = self._top_labels(candidates)
        scene_type = self._scene_type(mean_intensity, contrast, candidate_count, signal_density)
        notes = self._notes(scene_type, candidate_count, avg_score, avg_confidence, signal_density)

        return BackendModelSummary(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            mode="space" if self.space_mode else "generic",
            scene_type=scene_type,
            confidence=round(confidence, 4),
            signal_density=round(signal_density, 4),
            candidate_count=candidate_count,
            average_candidate_score=round(avg_score, 4),
            average_candidate_confidence=round(avg_confidence, 4),
            top_labels=top_labels,
            notes=notes,
        )

    def _top_labels(self, candidates: list[AnomalyCandidate], limit: int = 3) -> list[str]:
        labels: list[str] = []
        for candidate in candidates[:limit]:
            descriptor = candidate.features.descriptor if candidate.features else ""
            label = descriptor.split(",")[0].strip() if descriptor else "unknown"
            if label and label not in labels:
                labels.append(label)
        return labels

    def _scene_type(
        self,
        mean_intensity: float,
        contrast: float,
        candidate_count: int,
        signal_density: float,
    ) -> str:
        if self.space_mode:
            if signal_density > 0.55 and candidate_count >= 3:
                return "dense astronomy field"
            if mean_intensity < 0.2:
                return "dark sky field"
            if contrast > 0.35:
                return "high-contrast space scene"
            return "astronomy scene"

        if candidate_count >= 5:
            return "pattern-dense scene"
        if signal_density > 0.45:
            return "high-detail scene"
        if contrast > 0.35:
            return "structured scene"
        return "quiet scene"

    def _notes(
        self,
        scene_type: str,
        candidate_count: int,
        avg_score: float,
        avg_confidence: float,
        signal_density: float,
    ) -> str:
        tone = "space-aware" if self.space_mode else "general-purpose"
        return (
            f"{self.config.model_name} is running as a {tone} backend model. "
            f"It summarized {candidate_count} candidates for a {scene_type} with "
            f"average score {avg_score:.2f}, average confidence {avg_confidence:.2f}, "
            f"and signal density {signal_density:.2f}."
        )

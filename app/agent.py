from __future__ import annotations

from pathlib import Path

import cv2

from app.contracts import (
    ActionType,
    AgentDecision,
    AnalysisResult,
    AnomalyCandidate,
    PatternMemoryRecord,
    ReviewItem,
    ReviewStatus,
)
from app.memory import ReviewQueue, VectorMemory
from app.perception import PerceptionEngine


class AnomalyInvestigator:
    def __init__(
        self,
        perception: PerceptionEngine,
        memory: VectorMemory,
        review_queue: ReviewQueue,
        low_confidence_threshold: float = 0.58,
        human_review_threshold: float = 0.7,
    ) -> None:
        self.perception = perception
        self.memory = memory
        self.review_queue = review_queue
        self.low_confidence_threshold = low_confidence_threshold
        self.human_review_threshold = human_review_threshold

    def analyze(self, image_path: Path) -> AnalysisResult:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        metadata, initial_candidates = self.perception.analyze_path(image_path)
        candidates = self._refine_low_confidence(image, initial_candidates)
        similar_patterns = {}
        decisions = {}

        provisional = AnalysisResult(
            image=metadata,
            candidates=candidates,
            similar_patterns={},
            decisions={},
            report="",
        )

        for candidate in candidates:
            matches = self.memory.search(candidate.embedding)
            similar_patterns[candidate.candidate_id] = matches
            decision = self._decide(candidate, matches)
            decisions[candidate.candidate_id] = decision

            if decision.action == ActionType.store_memory:
                label = matches[0].label if matches else "unknown-pattern"
                self.memory.upsert(
                    PatternMemoryRecord(
                        label=label,
                        image_id=metadata.image_id,
                        candidate_id=candidate.candidate_id,
                        bbox=candidate.bbox,
                        anomaly_score=candidate.anomaly_score,
                        embedding=candidate.embedding,
                        status=ReviewStatus.approved if matches else ReviewStatus.pending,
                        notes=decision.reason,
                    )
                )

            if decision.needs_human:
                self.review_queue.add(
                    ReviewItem(
                        run_id=provisional.run_id,
                        image=metadata,
                        candidate=candidate,
                        similar_patterns=matches,
                        question=(
                            "The system found an unknown pattern with insufficient certainty. "
                            "Should this become a named pattern, be ignored, or be marked as a defect?"
                        ),
                    )
                )

        result = provisional.model_copy(
            update={
                "similar_patterns": similar_patterns,
                "decisions": decisions,
            }
        )
        return result.model_copy(update={"report": self._report(result)})

    def _refine_low_confidence(self, image, candidates: list[AnomalyCandidate]) -> list[AnomalyCandidate]:
        final = list(candidates)
        for candidate in candidates:
            if candidate.confidence >= self.low_confidence_threshold:
                continue
            refined = self.perception.refine_candidate(image, candidate)
            if refined and refined[0].anomaly_score > candidate.anomaly_score:
                final.append(refined[0])
        final.sort(key=lambda item: item.anomaly_score, reverse=True)
        deduped = []
        seen = set()
        for item in final:
            key = (item.bbox.x_min // 8, item.bbox.y_min // 8, item.bbox.x_max // 8, item.bbox.y_max // 8)
            if key not in seen:
                deduped.append(item)
                seen.add(key)
        return deduped[: self.perception.max_candidates]

    def _decide(self, candidate: AnomalyCandidate, matches) -> AgentDecision:
        uncertainty = round(1.0 - max(candidate.confidence, candidate.anomaly_score), 4)
        if matches and matches[0].similarity >= 0.94:
            return AgentDecision(
                action=ActionType.store_memory,
                reason=f"Strong memory match to '{matches[0].label}' at {matches[0].similarity:.2f} similarity.",
                needs_human=False,
                uncertainty=max(0.0, uncertainty - 0.25),
            )
        if candidate.confidence < self.low_confidence_threshold:
            return AgentDecision(
                action=ActionType.ask_human,
                reason="Low confidence after refinement; preserving state for human validation.",
                needs_human=True,
                uncertainty=min(1.0, uncertainty + 0.2),
            )
        if candidate.anomaly_score >= self.human_review_threshold:
            return AgentDecision(
                action=ActionType.ask_human,
                reason="High anomaly score but no reliable memory match; human label needed before automation trusts it.",
                needs_human=True,
                uncertainty=uncertainty,
            )
        return AgentDecision(
            action=ActionType.store_memory,
            reason="Moderate unknown pattern; stored for clustering and future recurrence checks.",
            needs_human=True,
            uncertainty=uncertainty,
        )

    def _report(self, result: AnalysisResult) -> str:
        if not result.candidates:
            return "No meaningful unknown patterns were detected in this image."

        lines = [
            f"Analysis run {result.run_id}",
            f"Image: {result.image.filename} ({result.image.width}x{result.image.height})",
            f"Candidates found: {len(result.candidates)}",
            "",
        ]
        for index, candidate in enumerate(result.candidates, start=1):
            decision = result.decisions[candidate.candidate_id]
            matches = result.similar_patterns.get(candidate.candidate_id, [])
            match_text = f"{matches[0].label} ({matches[0].similarity:.2f})" if matches else "none"
            novelty_text = f"{candidate.model_novelty:.2f}" if candidate.model_novelty is not None else "not trained"
            lines.extend(
                [
                    f"{index}. score={candidate.anomaly_score:.2f}, confidence={candidate.confidence:.2f}, bbox={candidate.bbox.model_dump()}",
                    f"   visual={candidate.features.descriptor}; coco_novelty={novelty_text}; memory_match={match_text}",
                    f"   action={decision.action.value}; reason={decision.reason}",
                ]
            )
        return "\n".join(lines)

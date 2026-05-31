from pathlib import Path

import _repo_path  # noqa: F401
import cv2
import numpy as np

from app.agent import AnomalyInvestigator
from app.memory import ReviewQueue, VectorMemory
from app.perception import PerceptionEngine


def synthetic_image(path: Path) -> None:
    image = np.full((360, 480, 3), 210, dtype=np.uint8)
    cv2.rectangle(image, (40, 80), (430, 290), (200, 203, 205), -1)
    cv2.circle(image, (300, 180), 42, (30, 30, 35), 3)
    cv2.line(image, (260, 150), (340, 210), (20, 20, 25), 3)
    cv2.line(image, (340, 150), (260, 210), (20, 20, 25), 3)
    cv2.imwrite(str(path), image)


def synthetic_pattern_image(path: Path) -> None:
    image = np.full((640, 900, 3), 15, dtype=np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(180):
        x = int(rng.normal(520, 95))
        y = int(rng.normal(330, 70))
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), int(rng.integers(2, 5)), (190, 190, 175), -1)
    for _ in range(16):
        start = (int(rng.integers(380, 650)), int(rng.integers(240, 420)))
        end = (start[0] + int(rng.integers(-120, 140)), start[1] + int(rng.integers(-90, 90)))
        cv2.line(image, start, end, (130, 135, 125), 2)
    cv2.imwrite(str(path), image)


def test_perception_returns_typed_candidates(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    synthetic_image(path)
    metadata, candidates = PerceptionEngine(max_candidates=4).analyze_path(path)

    assert metadata.width == 480
    assert candidates
    assert candidates[0].anomaly_score > 0
    assert candidates[0].bbox.area() > 0


def test_perception_groups_points_into_pattern_regions(tmp_path: Path) -> None:
    path = tmp_path / "pattern.png"
    synthetic_pattern_image(path)

    _, candidates = PerceptionEngine(max_candidates=4).analyze_path(path)

    assert candidates
    assert candidates[0].bbox.width() > 120
    assert candidates[0].bbox.height() > 90


def test_agent_produces_report_and_review_queue(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    synthetic_image(path)
    investigator = AnomalyInvestigator(
        PerceptionEngine(max_candidates=4),
        VectorMemory(tmp_path / "memory.jsonl"),
        ReviewQueue(tmp_path / "reviews.json"),
    )

    result = investigator.analyze(path)

    assert result.report.startswith("Analysis run")
    assert result.candidates
    assert result.decisions


def test_deep_analysis_returns_search_tree(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    synthetic_image(path)

    result = PerceptionEngine(max_candidates=4).deep_analyze_path(path, max_depth=2, branch_limit=2)

    assert result.max_depth == 2
    assert result.nodes_searched >= len(result.root_candidates)
    assert result.report.startswith("Deep search run") or result.report.startswith("Deep search found")

from pathlib import Path

import cv2
import numpy as np

from app.coco_baseline import CocoBaselineModel
from app.perception import PerceptionEngine


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    image = np.full((180, 220, 3), color, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (150, 130), (color[0] // 2, color[1] // 2, color[2] // 2), -1)
    cv2.imwrite(str(path), image)


def test_coco_baseline_trains_saves_and_scores(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    make_image(first, (180, 190, 200))
    make_image(second, (60, 150, 210))

    perception = PerceptionEngine()
    baseline = CocoBaselineModel(tmp_path / "baseline.json")
    metadata = baseline.train_from_paths([first, second], perception.embed_patch, patch_size=64, stride=64)

    assert metadata.image_count == 2
    assert metadata.patch_count > 0
    assert baseline.is_trained

    reloaded = CocoBaselineModel(tmp_path / "baseline.json")
    embedding = perception.embed_patch(cv2.imread(str(first), cv2.IMREAD_COLOR)[:64, :64])
    similarity = reloaded.nearest_similarity(embedding)

    assert similarity is not None
    assert 0 <= similarity <= 1


def test_perception_includes_coco_novelty_when_baseline_exists(tmp_path: Path) -> None:
    image_path = tmp_path / "normal.jpg"
    make_image(image_path, (180, 190, 200))

    baseline = CocoBaselineModel(tmp_path / "baseline.json")
    trainer = PerceptionEngine()
    baseline.train_from_paths([image_path], trainer.embed_patch, patch_size=64, stride=64)

    perception = PerceptionEngine(baseline_model=baseline)
    _, candidates = perception.analyze_path(image_path)

    for candidate in candidates:
        assert candidate.baseline_similarity is not None
        assert candidate.model_novelty is not None


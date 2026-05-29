from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from pydantic import BaseModel, Field


class CocoBaselineMetadata(BaseModel):
    source_url: str = "http://images.cocodataset.org/"
    trained_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    image_count: int = 0
    patch_count: int = 0
    embedding_dim: int = 0
    patch_size: int = 96
    stride: int = 96


class CocoBaselinePayload(BaseModel):
    metadata: CocoBaselineMetadata
    embeddings: list[list[float]]


class CocoBaselineModel:

    def __init__(self, path: Path) -> None:
        self.path = path
        self.metadata = CocoBaselineMetadata()
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.load()

    @property
    def is_trained(self) -> bool:
        return self.embeddings.size > 0

    def summary(self) -> dict[str, int | str | bool]:
        return {
            "trained": self.is_trained,
            "source_url": self.metadata.source_url,
            "image_count": self.metadata.image_count,
            "patch_count": self.metadata.patch_count,
            "embedding_dim": self.metadata.embedding_dim,
        }

    def train_from_paths(
        self,
        image_paths: list[Path],
        embed_patch: Callable[[np.ndarray], list[float]],
        patch_size: int = 96,
        stride: int = 96,
        max_patches_per_image: int = 32,
    ) -> CocoBaselineMetadata:
        vectors: list[list[float]] = []
        used_images = 0

        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            used_images += 1
            for patch in self._patches(image, patch_size, stride, max_patches_per_image):
                vectors.append(embed_patch(patch))

        if not vectors:
            raise ValueError("No usable image patches were found for COCO baseline training.")

        self.embeddings = np.array(vectors, dtype=np.float32)
        self.metadata = CocoBaselineMetadata(
            image_count=used_images,
            patch_count=len(vectors),
            embedding_dim=int(self.embeddings.shape[1]),
            patch_size=patch_size,
            stride=stride,
        )
        self.save()
        return self.metadata

    def nearest_similarity(self, embedding: list[float]) -> float | None:
        if not self.is_trained:
            return None
        query = np.array(embedding, dtype=np.float32)
        denominator = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query)
        denominator = np.where(denominator == 0, 1, denominator)
        similarities = np.clip((self.embeddings @ query) / denominator, 0, 1)
        return round(float(similarities.max()), 4)

    def novelty_score(self, embedding: list[float]) -> float | None:
        similarity = self.nearest_similarity(embedding)
        if similarity is None:
            return None
        return round(1.0 - similarity, 4)

    def load(self) -> None:
        if not self.path.exists():
            return
        payload = CocoBaselinePayload.model_validate_json(self.path.read_text(encoding="utf-8"))
        self.metadata = payload.metadata
        self.embeddings = np.array(payload.embeddings, dtype=np.float32)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = CocoBaselinePayload(
            metadata=self.metadata,
            embeddings=[[round(float(value), 6) for value in row] for row in self.embeddings.tolist()],
        )
        self.path.write_text(payload.model_dump_json(), encoding="utf-8")

    def _patches(
        self,
        image: np.ndarray,
        patch_size: int,
        stride: int,
        max_patches: int,
    ) -> list[np.ndarray]:
        height, width = image.shape[:2]
        if height < patch_size or width < patch_size:
            scale = patch_size / min(height, width)
            image = cv2.resize(image, (int(width * scale) + 1, int(height * scale) + 1), interpolation=cv2.INTER_AREA)
            height, width = image.shape[:2]

        patches = []
        for y in range(0, max(1, height - patch_size + 1), stride):
            for x in range(0, max(1, width - patch_size + 1), stride):
                patch = image[y : y + patch_size, x : x + patch_size]
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    patches.append(patch)

        if not patches:
            center_y = max(0, (height - patch_size) // 2)
            center_x = max(0, (width - patch_size) // 2)
            patches.append(image[center_y : center_y + patch_size, center_x : center_x + patch_size])

        if len(patches) <= max_patches:
            return patches

        indexes = np.linspace(0, len(patches) - 1, max_patches, dtype=int)
        return [patches[index] for index in indexes]


from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np
from PIL import Image

from app.contracts import AnomalyCandidate, BoundingBox, DeepAnalysisResult, DeepSearchNode, ImageMetadata, VisualFeatures

if TYPE_CHECKING:
    from app.coco_baseline import CocoBaselineModel


@dataclass(frozen=True)
class DetectedRegion:
    bbox: BoundingBox
    score: float
    confidence: float
    label: str


class ObjectDetectionBackend(Protocol):
    def detect(self, image: np.ndarray) -> list[DetectedRegion]:
        """Return candidate regions detected in the image."""


class AstronomyDetectionBackend:
    """Detect bright astronomical sources, circular bodies, and streaks."""

    def __init__(
        self,
        background_blur: int = 71,
        mad_multiplier: float = 2.4,
        min_area_ratio: float = 0.00002,
        max_area_ratio: float = 0.08,
    ) -> None:
        self.background_blur = self._odd_kernel(background_blur)
        self.mad_multiplier = mad_multiplier
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def detect(self, image: np.ndarray) -> list[DetectedRegion]:
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image.copy()

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        image_area = max(1, height * width)

        background = cv2.GaussianBlur(gray, (self.background_blur, self.background_blur), 0)
        residual = cv2.subtract(gray, background)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual.astype(np.float32) - median)))
        threshold = median + (self.mad_multiplier * max(1.0, 1.4826 * mad))

        bright_mask = (residual >= threshold).astype(np.uint8) * 255
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        regions: list[DetectedRegion] = []
        count, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if area <= 0:
                continue
            area_ratio = area / image_area
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue
            patch = residual[y : y + h, x : x + w]
            if patch.size == 0:
                continue

            signal = float(np.clip((patch.mean() + patch.std()) / 70.0, 0, 1))
            aspect_ratio = max(w, h) / max(1, min(w, h))
            compactness = float(np.clip(area / max(1.0, math.pi * (max(w, h) / 2.0) ** 2), 0, 1))
            label = "bright-source"
            if aspect_ratio >= 2.8:
                label = "streak"
            elif compactness >= 0.55:
                label = "circular-source"

            score = float(np.clip(0.55 * signal + 0.25 * compactness + 0.20 * min(1.0, area_ratio / 0.01), 0, 1))
            confidence = float(np.clip(0.60 * score + 0.40 * min(1.0, area_ratio / 0.02), 0, 1))
            regions.append(
                DetectedRegion(
                    bbox=BoundingBox(x_min=int(x), y_min=int(y), x_max=int(x + w), y_max=int(y + h)),
                    score=round(score, 4),
                    confidence=round(confidence, 4),
                    label=label,
                )
            )

        circles = cv2.HoughCircles(
            cv2.GaussianBlur(gray, (9, 9), 1.6),
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(8, min(height, width) // 14),
            param1=90,
            param2=18,
            minRadius=2,
            maxRadius=max(6, min(height, width) // 8),
        )
        if circles is not None:
            for raw in np.round(circles[0]).astype(int):
                x, y, radius = raw
                if radius <= 0:
                    continue
                x0 = max(0, x - radius - 2)
                y0 = max(0, y - radius - 2)
                x1 = min(width, x + radius + 2)
                y1 = min(height, y + radius + 2)
                area = max(1, (x1 - x0) * (y1 - y0))
                area_ratio = area / image_area
                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    continue
                regions.append(
                    DetectedRegion(
                        bbox=BoundingBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1),
                        score=0.88,
                        confidence=0.82,
                        label="circular-source",
                    )
                )

        lines = cv2.HoughLinesP(
            cv2.Canny(gray, 50, 140),
            rho=1,
            theta=np.pi / 180.0,
            threshold=max(20, min(height, width) // 18),
            minLineLength=max(20, min(height, width) // 10),
            maxLineGap=max(6, min(height, width) // 40),
        )
        if lines is not None:
            for line in lines[:, 0]:
                x1, y1, x2, y2 = map(int, line)
                left = max(0, min(x1, x2) - 4)
                top = max(0, min(y1, y2) - 4)
                right = min(width, max(x1, x2) + 4)
                bottom = min(height, max(y1, y2) + 4)
                area = max(1, (right - left) * (bottom - top))
                area_ratio = area / image_area
                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    continue
                length = float(math.hypot(x2 - x1, y2 - y1))
                regions.append(
                    DetectedRegion(
                        bbox=BoundingBox(x_min=left, y_min=top, x_max=right, y_max=bottom),
                        score=round(float(np.clip(length / max(width, height), 0.25, 0.95)), 4),
                        confidence=0.72,
                        label="streak",
                    )
                )

        return self._dedupe(regions)

    def _dedupe(self, regions: list[DetectedRegion]) -> list[DetectedRegion]:
        selected: list[DetectedRegion] = []
        for region in sorted(regions, key=lambda item: (item.bbox.area(), item.score), reverse=True):
            if all(self._iou(region.bbox, kept.bbox) < 0.35 for kept in selected):
                selected.append(region)
        return selected

    def _iou(self, first: BoundingBox, second: BoundingBox) -> float:
        x0 = max(first.x_min, second.x_min)
        y0 = max(first.y_min, second.y_min)
        x1 = min(first.x_max, second.x_max)
        y1 = min(first.y_max, second.y_max)
        intersection = max(0, x1 - x0) * max(0, y1 - y0)
        union = first.area() + second.area() - intersection
        return intersection / union if union else 0.0

    def _odd_kernel(self, value: int) -> int:
        size = max(3, int(value))
        if size % 2 == 0:
            size += 1
        return size


class PerceptionEngine:
    """A replaceable CV layer that outputs typed anomaly evidence."""

    def __init__(
        self,
        max_candidates: int = 8,
        baseline_model: "CocoBaselineModel | None" = None,
        detection_backend: ObjectDetectionBackend | None = None,
    ) -> None:
        self.max_candidates = max_candidates
        self.baseline_model = baseline_model
        self.detection_backend = detection_backend

    def analyze_path(self, path: Path, source_pass: str = "initial") -> tuple[ImageMetadata, list[AnomalyCandidate]]:
        image = self._load_image(path)
        metadata = ImageMetadata(
            filename=path.name,
            width=image.shape[1],
            height=image.shape[0],
            mode="BGR",
        )
        return metadata, self.analyze_array(image, source_pass=source_pass)

    def deep_analyze_path(
        self,
        path: Path,
        max_depth: int = 3,
        branch_limit: int = 3,
        min_child_score: float = 0.2,
    ) -> DeepAnalysisResult:
        image = self._load_image(path)
        metadata = ImageMetadata(
            filename=path.name,
            width=image.shape[1],
            height=image.shape[0],
            mode="BGR",
        )
        roots = []
        nodes_searched = 0
        for index, candidate in enumerate(self.analyze_array(image)[:branch_limit], start=1):
            node, count = self._deep_search_node(
                image=image,
                candidate=candidate,
                depth=0,
                max_depth=max_depth,
                branch_limit=branch_limit,
                min_child_score=min_child_score,
                path=f"{index}",
            )
            roots.append(node)
            nodes_searched += count

        result = DeepAnalysisResult(
            image=metadata,
            max_depth=max_depth,
            nodes_searched=nodes_searched,
            root_candidates=roots,
            report="",
        )
        return result.model_copy(update={"report": self._deep_report(result)})

    def analyze_array(self, image: np.ndarray, source_pass: str = "initial") -> list[AnomalyCandidate]:
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image.copy()

        profile = self._analysis_profile()
        image_height, image_width = bgr.shape[:2]
        image_area = max(1, image_height * image_width)
        min_dim = min(image_height, image_width)

        regions = self._detect_regions(bgr, profile=profile, min_dim=min_dim)
        candidates: list[AnomalyCandidate] = []

        for region in regions:
            box = region.bbox
            area_ratio = box.area() / image_area
            candidate = self._candidate_from_box(
                bgr,
                box.x_min,
                box.y_min,
                box.width(),
                box.height(),
                area_ratio,
                source_pass,
                region_score=region.score,
                region_confidence=region.confidence,
                region_label=region.label,
            )
            if candidate:
                candidates.append(candidate)

        if len(candidates) < max(2, self.max_candidates // 3):
            for region in self._fallback_regions(bgr, profile=profile, min_dim=min_dim):
                box = region.bbox
                area_ratio = box.area() / image_area
                candidate = self._candidate_from_box(
                    bgr,
                    box.x_min,
                    box.y_min,
                    box.width(),
                    box.height(),
                    area_ratio,
                    source_pass,
                    region_score=region.score,
                    region_confidence=region.confidence,
                    region_label=region.label,
                )
                if candidate:
                    candidates.append(candidate)

        if len(candidates) < 2:
            candidates.extend(self._compact_search(bgr, source_pass))

        candidates.sort(key=lambda item: (item.bbox.area(), item.anomaly_score), reverse=True)
        return self._dedupe_candidates(candidates)[: self.max_candidates]

    def _analysis_profile(self) -> dict[str, float | int]:
        return {
            "edge_low": 45,
            "edge_high": 130,
            "delta_multiplier": 2.2,
            "delta_blur": 41,
            "pattern_kernel_scale": 85,
            "pattern_kernel_min": 9,
            "min_pattern_ratio": 0.025,
            "min_pattern_side": 12,
            "min_detail_ratio": 0.00003,
            "min_detail_side": 8,
            "candidate_floor": 0.18,
        }

    def _score_weights(self) -> tuple[float, float, float, float]:
        return (0.35, 0.25, 0.25, 0.15)

    def _detect_regions(
        self,
        bgr: np.ndarray,
        profile: dict[str, float | int],
        min_dim: int,
    ) -> list[DetectedRegion]:
        if self.detection_backend is not None:
            return self.detection_backend.detect(bgr)
        return self._heuristic_regions(bgr, profile=profile, min_dim=min_dim)

    def _heuristic_regions(
        self,
        bgr: np.ndarray,
        profile: dict[str, float | int],
        min_dim: int,
    ) -> list[DetectedRegion]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, int(profile["edge_low"]), int(profile["edge_high"]))

        local_mean = cv2.GaussianBlur(gray, (int(profile["delta_blur"]), int(profile["delta_blur"])), 0)
        local_delta = cv2.absdiff(gray, local_mean)
        delta_threshold = float(local_delta.mean() + float(profile["delta_multiplier"]) * local_delta.std())
        high_delta = (local_delta > delta_threshold).astype(np.uint8) * 255

        combined = cv2.bitwise_or(edges, high_delta)
        open_kernel = np.ones((self._odd_kernel(max(3, min_dim // 900), 3), self._odd_kernel(max(3, min_dim // 900), 3)), np.uint8)
        pattern_kernel_size = self._odd_kernel(
            max(int(profile["pattern_kernel_min"]), min_dim // int(profile["pattern_kernel_scale"])),
            int(profile["pattern_kernel_min"]),
        )
        pattern_kernel = np.ones((pattern_kernel_size, pattern_kernel_size), np.uint8)

        detail_mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_kernel, iterations=1)
        pattern_mask = cv2.morphologyEx(detail_mask, cv2.MORPH_CLOSE, pattern_kernel, iterations=2)
        pattern_mask = cv2.dilate(pattern_mask, pattern_kernel, iterations=1)

        pattern_regions = self._regions_from_contours(
            cv2.findContours(pattern_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            bgr.shape[:2],
            max(int(profile["min_pattern_side"]), int(min_dim * float(profile["min_pattern_ratio"]))),
            0.75,
            0.00008,
            "pattern",
        )
        if len(pattern_regions) >= max(2, self.max_candidates // 3):
            return pattern_regions

        detail_regions = self._regions_from_contours(
            cv2.findContours(detail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            bgr.shape[:2],
            max(int(profile["min_detail_side"]), int(min_dim * 0.012)),
            0.65,
            float(profile["min_detail_ratio"]),
            "detail",
        )
        return pattern_regions + detail_regions

    def _fallback_regions(self, bgr: np.ndarray, profile: dict[str, float | int], min_dim: int) -> list[DetectedRegion]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, int(profile["edge_low"]), int(profile["edge_high"]))
        return self._regions_from_mask(edges, bgr.shape[:2], max(int(profile["min_detail_side"]), int(min_dim * 0.012)), 0.65, 0.00003, "detail")

    def _compact_search(self, bgr: np.ndarray, source_pass: str) -> list[AnomalyCandidate]:
        height, width = bgr.shape[:2]
        patch_size = max(24, min(height, width) // 4)
        y = max(0, (height - patch_size) // 2)
        x = max(0, (width - patch_size) // 2)
        candidate = self._candidate_from_box(
            bgr,
            x,
            y,
            patch_size,
            patch_size,
            (patch_size * patch_size) / max(1, height * width),
            source_pass,
        )
        return [candidate] if candidate else []

    def _regions_from_contours(
        self,
        contours,
        image_shape: tuple[int, int],
        min_side: int,
        max_area_ratio: float,
        min_area_ratio: float,
        label: str,
    ) -> list[DetectedRegion]:
        image_height, image_width = image_shape
        image_area = max(1, image_height * image_width)
        regions: list[DetectedRegion] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_side or h < min_side:
                continue
            area_ratio = (w * h) / image_area
            if area_ratio > max_area_ratio or area_ratio < min_area_ratio:
                continue
            regions.append(
                DetectedRegion(
                    bbox=BoundingBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h),
                    score=round(float(np.clip(area_ratio / max_area_ratio, 0.15, 0.95)), 4),
                    confidence=round(float(np.clip(area_ratio / max_area_ratio, 0.20, 0.90)), 4),
                    label=label,
                )
            )
        return self._dedupe_regions(regions)

    def _regions_from_mask(
        self,
        mask: np.ndarray,
        image_shape: tuple[int, int],
        min_side: int,
        max_area_ratio: float,
        min_area_ratio: float,
        label: str,
    ) -> list[DetectedRegion]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._regions_from_contours(contours, image_shape, min_side, max_area_ratio, min_area_ratio, label)

    def _candidate_from_box(
        self,
        bgr: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        area_ratio: float,
        source_pass: str,
        region_score: float | None = None,
        region_confidence: float | None = None,
        region_label: str | None = None,
    ) -> AnomalyCandidate | None:
        patch = bgr[y : y + h, x : x + w]
        score, confidence, features, embedding = self._score_patch(patch, area_ratio, region_label=region_label)
        if region_score is not None:
            score = round(float(np.clip((0.7 * score) + (0.3 * region_score), 0, 1)), 4)
        if region_confidence is not None:
            confidence = round(float(np.clip((0.7 * confidence) + (0.3 * region_confidence), 0, 1)), 4)

        baseline_similarity = self.baseline_model.nearest_similarity(embedding) if self.baseline_model else None
        model_novelty = round(1.0 - baseline_similarity, 4) if baseline_similarity is not None else None
        if model_novelty is not None:
            score = round(float(np.clip((0.7 * score) + (0.3 * model_novelty), 0, 1)), 4)
            confidence = round(float(np.clip((0.8 * confidence) + (0.2 * min(1.0, score + model_novelty)), 0, 1)), 4)

        if score < float(self._analysis_profile()["candidate_floor"]):
            return None

        return AnomalyCandidate(
            bbox=BoundingBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h),
            anomaly_score=score,
            confidence=confidence,
            baseline_similarity=baseline_similarity,
            model_novelty=model_novelty,
            features=features,
            embedding=embedding,
            source_pass=source_pass,  # type: ignore[arg-type]
        )

    def _score_patch(
        self,
        patch: np.ndarray,
        area_ratio: float,
        region_label: str | None = None,
    ) -> tuple[float, float, VisualFeatures, list[float]]:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        edge_density = float(np.count_nonzero(edges) / max(1, edges.size))
        contrast = float(gray.std())
        entropy = self._entropy(gray)
        spatial_frequency = self._spatial_frequency(gray)
        dominant_rgb = self._dominant_rgb(patch)

        contrast_score = min(1.0, contrast / 80.0)
        entropy_score = min(1.0, entropy / 8.0)
        frequency_score = min(1.0, spatial_frequency / 45.0)
        size_score = 1.0 - min(1.0, abs(area_ratio - 0.08) / 0.2)
        edge_w, contrast_w, frequency_w, entropy_w = self._score_weights()
        score = float(
            np.clip(
                edge_w * edge_density + contrast_w * contrast_score + frequency_w * frequency_score + entropy_w * entropy_score,
                0,
                1,
            )
        )
        confidence = float(np.clip(0.55 * score + 0.25 * size_score + 0.2 * min(1.0, edge_density * 5), 0, 1))

        descriptor_bits = [
            "dense edges" if edge_density > 0.12 else "sparse edges",
            "high contrast" if contrast > 45 else "low contrast",
            "textured" if entropy > 4.5 else "smooth",
            "high frequency" if spatial_frequency > 25 else "low frequency",
        ]
        if region_label:
            descriptor_bits.insert(0, region_label.replace("-", " "))

        features = VisualFeatures(
            edge_density=round(edge_density, 4),
            contrast=round(contrast, 3),
            texture_entropy=round(entropy, 3),
            dominant_color_rgb=dominant_rgb,
            spatial_frequency=round(spatial_frequency, 3),
            descriptor=", ".join(descriptor_bits),
        )
        return round(score, 4), round(confidence, 4), features, self._embedding(patch)

    def _score_weights(self) -> tuple[float, float, float, float]:
        return (0.35, 0.25, 0.25, 0.15)

    def _dedupe_candidates(self, candidates: list[AnomalyCandidate]) -> list[AnomalyCandidate]:
        selected: list[AnomalyCandidate] = []
        for candidate in candidates:
            if all(self._iou(candidate.bbox, kept.bbox) < 0.35 for kept in selected):
                selected.append(candidate)
        return selected

    def _dedupe_regions(self, regions: list[DetectedRegion]) -> list[DetectedRegion]:
        selected: list[DetectedRegion] = []
        for region in sorted(regions, key=lambda item: (item.bbox.area(), item.score), reverse=True):
            if all(self._iou(region.bbox, kept.bbox) < 0.35 for kept in selected):
                selected.append(region)
        return selected

    def _odd_kernel(self, value: int, minimum: int) -> int:
        size = max(minimum, int(value))
        if size % 2 == 0:
            size += 1
        return size

    def refine_candidate(self, image: np.ndarray, candidate: AnomalyCandidate) -> list[AnomalyCandidate]:
        box = candidate.bbox
        margin = max(12, int(max(box.width(), box.height()) * 0.2))
        x0 = max(0, box.x_min - margin)
        y0 = max(0, box.y_min - margin)
        x1 = min(image.shape[1], box.x_max + margin)
        y1 = min(image.shape[0], box.y_max + margin)
        crop = image[y0:y1, x0:x1]
        enhanced = self.enhance(crop)
        refined = self.analyze_array(enhanced, source_pass="refined")

        adjusted = []
        for item in refined:
            adjusted.append(
                item.model_copy(
                    update={
                        "bbox": BoundingBox(
                            x_min=item.bbox.x_min + x0,
                            y_min=item.bbox.y_min + y0,
                            x_max=item.bbox.x_max + x0,
                            y_max=item.bbox.y_max + y0,
                        )
                    }
                )
            )
        return adjusted

    def crop_candidate(self, image: np.ndarray, candidate: AnomalyCandidate, margin_ratio: float = 0.18) -> tuple[np.ndarray, int, int]:
        box = candidate.bbox
        margin = max(8, int(max(box.width(), box.height()) * margin_ratio))
        x0 = max(0, box.x_min - margin)
        y0 = max(0, box.y_min - margin)
        x1 = min(image.shape[1], box.x_max + margin)
        y1 = min(image.shape[0], box.y_max + margin)
        return image[y0:y1, x0:x1], x0, y0

    def enhance(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        merged = cv2.merge((enhanced_l, a_channel, b_channel))
        sharpened = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(sharpened, -1, kernel)

    def embed_patch(self, patch: np.ndarray) -> list[float]:
        return self._embedding(patch)

    def _deep_search_node(
        self,
        image: np.ndarray,
        candidate: AnomalyCandidate,
        depth: int,
        max_depth: int,
        branch_limit: int,
        min_child_score: float,
        path: str,
    ) -> tuple[DeepSearchNode, int]:
        node = DeepSearchNode(depth=depth, path=path, candidate=candidate)
        searched = 1
        if depth >= max_depth:
            return node, searched

        crop, offset_x, offset_y = self.crop_candidate(image, candidate)
        if crop.size == 0 or crop.shape[0] < 18 or crop.shape[1] < 18:
            return node, searched

        zoom = cv2.resize(crop, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
        enhanced = self.enhance(zoom)
        child_candidates = self.analyze_array(enhanced, source_pass="refined")
        child_candidates = [item for item in child_candidates if item.anomaly_score >= min_child_score]

        children = []
        for child_index, child in enumerate(child_candidates[:branch_limit], start=1):
            mapped = child.model_copy(
                update={
                    "bbox": BoundingBox(
                        x_min=offset_x + int(child.bbox.x_min / 1.8),
                        y_min=offset_y + int(child.bbox.y_min / 1.8),
                        x_max=offset_x + int(child.bbox.x_max / 1.8),
                        y_max=offset_y + int(child.bbox.y_max / 1.8),
                    )
                }
            )
            child_node, child_count = self._deep_search_node(
                image=image,
                candidate=mapped,
                depth=depth + 1,
                max_depth=max_depth,
                branch_limit=branch_limit,
                min_child_score=min_child_score,
                path=f"{path}.{child_index}",
            )
            children.append(child_node)
            searched += child_count

        return node.model_copy(update={"children": children}), searched

    def _deep_report(self, result: DeepAnalysisResult) -> str:
        if not result.root_candidates:
            return "Deep search found no suspicious regions to expand."

        lines = [
            f"Deep search run {result.run_id}",
            f"Image: {result.image.filename} ({result.image.width}x{result.image.height})",
            f"Depth limit: {result.max_depth}; nodes searched: {result.nodes_searched}",
            "",
        ]

        def walk(node: DeepSearchNode) -> None:
            candidate = node.candidate
            novelty = f"{candidate.model_novelty:.2f}" if candidate.model_novelty is not None else "not trained"
            indent = "  " * node.depth
            lines.append(
                f"{indent}{node.path} depth={node.depth} score={candidate.anomaly_score:.2f} "
                f"confidence={candidate.confidence:.2f} novelty={novelty} bbox={candidate.bbox.model_dump()}"
            )
            lines.append(f"{indent}   {candidate.features.descriptor}")
            for child in node.children:
                walk(child)

        for root in result.root_candidates:
            walk(root)
        return "\n".join(lines)

    def _load_image(self, path: Path) -> np.ndarray:
        with Image.open(path) as raw:
            rgb = raw.convert("RGB")
            return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

    def _embedding(self, patch: np.ndarray) -> list[float]:
        resized = cv2.resize(patch, (16, 16), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gradients_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gradients_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        magnitudes, angles = cv2.cartToPolar(gradients_x, gradients_y, angleInDegrees=True)
        bins = np.zeros(8, dtype=np.float32)
        for idx in range(8):
            mask = (angles >= idx * 45) & (angles < (idx + 1) * 45)
            bins[idx] = float(magnitudes[mask].mean()) if np.any(mask) else 0.0
        stats = np.array([gray.mean(), gray.std(), self._entropy(gray), self._spatial_frequency(gray)], dtype=np.float32)
        vector = np.concatenate([hist_h.astype(np.float32), bins, stats])
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return [0.0 for _ in vector]
        return [round(float(value / norm), 6) for value in vector]

    def _dominant_rgb(self, patch: np.ndarray) -> tuple[int, int, int]:
        mean_bgr = patch.reshape(-1, 3).mean(axis=0)
        return (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))

    def _entropy(self, gray: np.ndarray) -> float:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        probs = hist / max(1.0, hist.sum())
        probs = probs[probs > 0]
        return float(-(probs * np.log2(probs)).sum())

    def _spatial_frequency(self, gray: np.ndarray) -> float:
        if gray.size == 0:
            return 0.0
        row_freq = np.diff(gray.astype(np.float32), axis=0)
        col_freq = np.diff(gray.astype(np.float32), axis=1)
        return float(math.sqrt((row_freq**2).mean() + (col_freq**2).mean()))

    def _iou(self, first: BoundingBox, second: BoundingBox) -> float:
        x0 = max(first.x_min, second.x_min)
        y0 = max(first.y_min, second.y_min)
        x1 = min(first.x_max, second.x_max)
        y1 = min(first.y_max, second.y_max)
        intersection = max(0, x1 - x0) * max(0, y1 - y0)
        union = first.area() + second.area() - intersection
        return intersection / union if union else 0.0


class SpacePerceptionEngine(PerceptionEngine):
    def __init__(
        self,
        max_candidates: int = 10,
        baseline_model: "CocoBaselineModel | None" = None,
        detection_backend: ObjectDetectionBackend | None = None,
    ) -> None:
        super().__init__(
            max_candidates=max_candidates,
            baseline_model=baseline_model,
            detection_backend=detection_backend or AstronomyDetectionBackend(),
        )

    def _analysis_profile(self) -> dict[str, float | int]:
        return {
            "edge_low": 24,
            "edge_high": 96,
            "delta_multiplier": 1.45,
            "delta_blur": 61,
            "pattern_kernel_scale": 120,
            "pattern_kernel_min": 5,
            "min_pattern_ratio": 0.006,
            "min_pattern_side": 6,
            "min_detail_ratio": 0.00001,
            "min_detail_side": 4,
            "candidate_floor": 0.08,
        }

    def _score_weights(self) -> tuple[float, float, float, float]:
        return (0.18, 0.24, 0.16, 0.14)

    def _detect_regions(
        self,
        bgr: np.ndarray,
        profile: dict[str, float | int],
        min_dim: int,
    ) -> list[DetectedRegion]:
        return self.detection_backend.detect(bgr)

    def _score_patch(
        self,
        patch: np.ndarray,
        area_ratio: float,
        region_label: str | None = None,
    ) -> tuple[float, float, VisualFeatures, list[float]]:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 120)
        edge_density = float(np.count_nonzero(edges) / max(1, edges.size))
        contrast = float(gray.std())
        entropy = self._entropy(gray)
        spatial_frequency = self._spatial_frequency(gray)
        dominant_rgb = self._dominant_rgb(patch)

        background_level = float(np.median(gray))
        signal_level = float(np.clip((gray.mean() - background_level + gray.std()) / 80.0, 0, 1))
        contrast_score = min(1.0, contrast / 48.0)
        entropy_score = min(1.0, entropy / 7.0)
        frequency_score = min(1.0, spatial_frequency / 32.0)
        signal_score = min(1.0, signal_level)
        size_score = 1.0 - min(1.0, abs(area_ratio - 0.01) / 0.05)
        edge_w, contrast_w, frequency_w, entropy_w = self._score_weights()
        score = float(
            np.clip(
                (edge_w * edge_density)
                + (contrast_w * contrast_score)
                + (frequency_w * frequency_score)
                + (entropy_w * entropy_score)
                + (0.25 * signal_score),
                0,
                1,
            )
        )
        confidence = float(np.clip(0.45 * score + 0.25 * size_score + 0.30 * signal_score, 0, 1))

        descriptor_bits = [
            "bright source" if signal_score > 0.45 else "faint source",
            "circular" if area_ratio < 0.03 else "extended",
            "elongated" if edge_density > 0.18 else "compact",
            "noisy" if entropy > 4.0 else "clean",
        ]
        if region_label:
            descriptor_bits.insert(0, region_label.replace("-", " "))

        features = VisualFeatures(
            edge_density=round(edge_density, 4),
            contrast=round(contrast, 3),
            texture_entropy=round(entropy, 3),
            dominant_color_rgb=dominant_rgb,
            spatial_frequency=round(spatial_frequency, 3),
            descriptor=", ".join(descriptor_bits),
        )
        return round(score, 4), round(confidence, 4), features, self._embedding(patch)

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from app.contracts import AnomalyCandidate, BoundingBox, DeepAnalysisResult, DeepSearchNode, ImageMetadata, VisualFeatures

if TYPE_CHECKING:
    from app.coco_baseline import CocoBaselineModel


class PerceptionEngine:
    """A replaceable CV layer that outputs typed anomaly evidence."""

    def __init__(self, max_candidates: int = 8, baseline_model: "CocoBaselineModel | None" = None) -> None:
        self.max_candidates = max_candidates
        self.baseline_model = baseline_model

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

        image_height, image_width = bgr.shape[:2]
        image_area = image_height * image_width
        min_dim = min(image_height, image_width)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 130)

        local_mean = cv2.GaussianBlur(gray, (41, 41), 0)
        local_delta = cv2.absdiff(gray, local_mean)
        delta_threshold = float(local_delta.mean() + 2.2 * local_delta.std())
        high_delta = (local_delta > delta_threshold).astype(np.uint8) * 255

        combined = cv2.bitwise_or(edges, high_delta)
        open_kernel = np.ones((self._odd_kernel(min_dim // 900, 3), self._odd_kernel(min_dim // 900, 3)), np.uint8)
        pattern_kernel_size = self._odd_kernel(min_dim // 85, 9)
        pattern_kernel = np.ones((pattern_kernel_size, pattern_kernel_size), np.uint8)

        detail_mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_kernel, iterations=1)
        pattern_mask = cv2.morphologyEx(detail_mask, cv2.MORPH_CLOSE, pattern_kernel, iterations=2)
        pattern_mask = cv2.dilate(pattern_mask, pattern_kernel, iterations=1)

        pattern_contours, _ = cv2.findContours(pattern_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detail_contours, _ = cv2.findContours(detail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        min_pattern_side = max(12, int(min_dim * 0.025))
        for contour in pattern_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_pattern_side or h < min_pattern_side:
                continue
            area_ratio = (w * h) / image_area
            if area_ratio > 0.75 or area_ratio < 0.00008:
                continue
            candidate = self._candidate_from_box(bgr, x, y, w, h, area_ratio, source_pass)
            if candidate:
                candidates.append(candidate)

        if len(candidates) < max(2, self.max_candidates // 3):
            min_detail_side = max(8, int(min_dim * 0.012))
            for contour in detail_contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w < min_detail_side or h < min_detail_side:
                    continue
                area_ratio = (w * h) / image_area
                if area_ratio > 0.65 or area_ratio < 0.00003:
                    continue
                candidate = self._candidate_from_box(bgr, x, y, w, h, area_ratio, source_pass)
                if candidate:
                    candidates.append(candidate)

        candidates.sort(key=lambda item: (item.bbox.area(), item.anomaly_score), reverse=True)
        return self._dedupe(candidates)[: self.max_candidates]

    def _candidate_from_box(
        self,
        bgr: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        area_ratio: float,
        source_pass: str,
    ) -> AnomalyCandidate | None:
        patch = bgr[y : y + h, x : x + w]
        score, confidence, features, embedding = self._score_patch(patch, area_ratio)
        baseline_similarity = self.baseline_model.nearest_similarity(embedding) if self.baseline_model else None
        model_novelty = round(1.0 - baseline_similarity, 4) if baseline_similarity is not None else None
        if model_novelty is not None:
            score = round(float(np.clip((0.7 * score) + (0.3 * model_novelty), 0, 1)), 4)
            confidence = round(float(np.clip((0.8 * confidence) + (0.2 * min(1.0, score + model_novelty)), 0, 1)), 4)
        if score < 0.18:
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

    def _odd_kernel(self, value: int, minimum: int) -> int:
        size = max(minimum, value)
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

    def _score_patch(self, patch: np.ndarray, area_ratio: float) -> tuple[float, float, VisualFeatures, list[float]]:
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
        score = float(np.clip(0.35 * edge_density + 0.25 * contrast_score + 0.25 * frequency_score + 0.15 * entropy_score, 0, 1))
        confidence = float(np.clip(0.55 * score + 0.25 * size_score + 0.2 * min(1.0, edge_density * 5), 0, 1))

        descriptor_bits = []
        descriptor_bits.append("dense edges" if edge_density > 0.12 else "sparse edges")
        descriptor_bits.append("high contrast" if contrast > 45 else "low contrast")
        descriptor_bits.append("textured" if entropy > 4.5 else "smooth")
        descriptor_bits.append("high frequency" if spatial_frequency > 25 else "low frequency")

        features = VisualFeatures(
            edge_density=round(edge_density, 4),
            contrast=round(contrast, 3),
            texture_entropy=round(entropy, 3),
            dominant_color_rgb=dominant_rgb,
            spatial_frequency=round(spatial_frequency, 3),
            descriptor=", ".join(descriptor_bits),
        )
        return round(score, 4), round(confidence, 4), features, self._embedding(patch)

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

    def _dedupe(self, candidates: list[AnomalyCandidate]) -> list[AnomalyCandidate]:
        selected: list[AnomalyCandidate] = []
        for candidate in candidates:
            if all(self._iou(candidate.bbox, kept.bbox) < 0.35 for kept in selected):
                selected.append(candidate)
        return selected

    def _iou(self, first: BoundingBox, second: BoundingBox) -> float:
        x0 = max(first.x_min, second.x_min)
        y0 = max(first.y_min, second.y_min)
        x1 = min(first.x_max, second.x_max)
        y1 = min(first.y_max, second.y_max)
        intersection = max(0, x1 - x0) * max(0, y1 - y0)
        union = first.area() + second.area() - intersection
        return intersection / union if union else 0.0

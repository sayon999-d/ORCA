from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.coco_baseline import CocoBaselineModel
from app.perception import PerceptionEngine
from scripts.download_coco_subset import DEFAULT_VAL2017_IDS, download_coco_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the COCO normal-image baseline used by the anomaly agent.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/coco/val2017"))
    parser.add_argument("--model-path", type=Path, default=Path("data/coco_baseline.json"))
    parser.add_argument("--download", action="store_true", help="Download a small COCO subset before training.")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--max-patches-per-image", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download:
        download_coco_images(args.image_dir, DEFAULT_VAL2017_IDS[: args.count])

    image_paths = sorted(args.image_dir.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(
            f"No COCO images found in {args.image_dir}. "
            "Run with --download or place images there manually."
        )

    perception = PerceptionEngine()
    baseline = CocoBaselineModel(args.model_path)
    metadata = baseline.train_from_paths(
        image_paths,
        perception.embed_patch,
        patch_size=args.patch_size,
        stride=args.stride,
        max_patches_per_image=args.max_patches_per_image,
    )
    print(
        "Trained COCO baseline: "
        f"{metadata.image_count} images, {metadata.patch_count} patches, "
        f"{metadata.embedding_dim} dimensions -> {args.model_path}"
    )


if __name__ == "__main__":
    main()


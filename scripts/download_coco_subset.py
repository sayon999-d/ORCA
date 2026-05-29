from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

COCO_BASE_URL = "http://images.cocodataset.org/"

DEFAULT_VAL2017_IDS = [
    139,
    285,
    632,
    724,
    776,
    785,
    802,
    872,
    885,
    1000,
    1268,
    1296,
    1365,
    1425,
    1490,
    1533,
]


def coco_image_url(image_id: int, split: str = "val2017") -> str:
    return f"{COCO_BASE_URL}{split}/{image_id:012d}.jpg"


def download_coco_images(
    output_dir: Path,
    image_ids: list[int],
    split: str = "val2017",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for image_id in image_ids:
        target = output_dir / f"{image_id:012d}.jpg"
        if target.exists() and target.stat().st_size > 0:
            downloaded.append(target)
            continue

        url = coco_image_url(image_id, split)
        try:
            print(f"Downloading {url}")
            urlretrieve(url, target)
            downloaded.append(target)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"Skipped {url}: {exc}", file=sys.stderr)
            if target.exists():
                target.unlink()

    return downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small subset of COCO images.")
    parser.add_argument("--output", type=Path, default=Path("data/coco/val2017"))
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--ids", nargs="*", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_ids = args.ids if args.ids else DEFAULT_VAL2017_IDS[: args.count]
    paths = download_coco_images(args.output, image_ids, args.split)
    print(f"Downloaded {len(paths)} image(s) into {args.output}")


if __name__ == "__main__":
    main()


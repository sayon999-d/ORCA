from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.coco_baseline import CocoBaselineModel
from app.perception import PerceptionEngine

SPACE_DATASET_URL = "https://www.kaggle.com/datasets/razaimam45/spacenet-an-optimally-distributed-astronomy-data"
SPACE_DATASET_SLUG = "razaimam45/spacenet-an-optimally-distributed-astronomy-data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            if line.startswith("KGAT_") and "KAGGLE_API_TOKEN" not in os.environ:
                os.environ["KAGGLE_API_TOKEN"] = line
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Orca's astronomy baseline from a downloaded Kaggle image folder.")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data/space", help="Folder containing downloaded astronomy images.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "data/space_baseline.json")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--max-patches-per-image", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of images to use. 0 means all images.")
    parser.add_argument("--download", action="store_true", help="Download Kaggle astronomy samples before training.")
    parser.add_argument("--download-limit", type=int, default=20, help="Maximum Kaggle image files to download. Defaults to 20.")
    parser.add_argument("--download-dir", type=Path, default=ROOT / "data/kaggle-downloads")
    parser.add_argument("--full-download", action="store_true", help="Download the full Kaggle dataset archive. This dataset is very large.")
    parser.add_argument("--no-download-if-missing", action="store_true", help="Fail instead of auto-downloading when no local images are found.")
    args = parser.parse_args()
    args.image_dir = project_path(args.image_dir)
    args.model_path = project_path(args.model_path)
    args.download_dir = project_path(args.download_dir)
    return args


def image_paths(root: Path, limit: int = 0) -> list[Path]:
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:limit] if limit > 0 else paths


def has_kaggle_auth() -> bool:
    return bool(
        os.environ.get("KAGGLE_API_TOKEN")
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        or (Path.home() / ".kaggle/kaggle.json").exists()
    )


def kaggle_image_files(kaggle_executable: str, sample_count: int) -> list[str]:
    result = subprocess.run(
        [
            kaggle_executable,
            "datasets",
            "files",
            SPACE_DATASET_SLUG,
            "--csv",
            "--page-size",
            "200",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = csv.DictReader(result.stdout.splitlines())
    names = [
        (row.get("name") or "").strip()
        for row in rows
        if Path((row.get("name") or "").strip()).suffix.lower() in IMAGE_EXTENSIONS
    ]
    return names[:sample_count]


def download_full_kaggle_dataset(kaggle_executable: str, image_dir: Path, download_dir: Path) -> None:
    print("Full Kaggle dataset download requested. This may download tens of GB.")
    download_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            kaggle_executable,
            "datasets",
            "download",
            SPACE_DATASET_SLUG,
            "-p",
            str(download_dir),
            "--force",
        ],
        check=True,
    )
    archives = sorted(download_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not archives:
        raise SystemExit(f"Kaggle download completed, but no zip archive was found in {download_dir}.")
    image_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(image_dir)


def download_kaggle_dataset(image_dir: Path, download_dir: Path, sample_count: int = 20, full_download: bool = False) -> None:
    kaggle_executable = shutil.which("kaggle")
    if not kaggle_executable:
        raise SystemExit("Kaggle CLI is not installed. Run: python -m pip install kaggle")
    if not has_kaggle_auth():
        raise SystemExit("Set Kaggle credentials in .env or ~/.kaggle/kaggle.json before downloading.")

    if full_download:
        download_full_kaggle_dataset(kaggle_executable, image_dir, download_dir)
        return

    sample_count = max(1, sample_count)
    files = kaggle_image_files(kaggle_executable, sample_count)
    if not files:
        raise SystemExit("No image files were listed by Kaggle. Use --full-download only if you really want the full dataset.")

    image_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(files)} Kaggle image sample(s) into {image_dir}")
    for file_name in files:
        subprocess.run(
            [
                kaggle_executable,
                "datasets",
                "download",
                SPACE_DATASET_SLUG,
                "-f",
                file_name,
                "-p",
                str(image_dir),
                "--force",
                "--quiet",
            ],
            check=True,
        )
        downloaded = image_dir / Path(file_name).name
        if downloaded.suffix.lower() == ".zip":
            with zipfile.ZipFile(downloaded) as archive:
                archive.extractall(image_dir)
            downloaded.unlink(missing_ok=True)


def main() -> None:
    load_env_file(ROOT / ".env")
    load_env_file(Path.cwd() / ".env")
    args = parse_args()
    if args.download:
        download_kaggle_dataset(args.image_dir, args.download_dir, args.download_limit, args.full_download)

    paths = image_paths(args.image_dir, args.limit)
    if not paths and not args.no_download_if_missing and has_kaggle_auth():
        auto_count = args.limit if args.limit > 0 else args.download_limit
        print(f"No local images found in {args.image_dir}. Downloading {auto_count} Kaggle sample(s)...")
        download_kaggle_dataset(args.image_dir, args.download_dir, auto_count, args.full_download)
        paths = image_paths(args.image_dir, args.limit)

    if not paths:
        raise SystemExit(
            f"No images found in {args.image_dir}. Download and unzip the Kaggle dataset, "
            "or run this script with --download."
        )

    perception = PerceptionEngine()
    baseline = CocoBaselineModel(args.model_path)
    metadata = baseline.train_from_paths(
        paths,
        perception.embed_patch,
        patch_size=args.patch_size,
        stride=args.stride,
        max_patches_per_image=args.max_patches_per_image,
        source_url=SPACE_DATASET_URL,
    )
    print(
        "Trained space baseline: "
        f"{metadata.image_count} images, {metadata.patch_count} patches, "
        f"{metadata.embedding_dim} dimensions -> {args.model_path}"
    )


if __name__ == "__main__":
    main()

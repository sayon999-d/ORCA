from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    output_dir = Path("data/samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((720, 960, 3), 218, dtype=np.uint8)
    noise = np.random.default_rng(42).normal(0, 5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.rectangle(image, (90, 130), (840, 590), (205, 209, 210), -1)
    for x in range(120, 820, 70):
        cv2.line(image, (x, 150), (x + 35, 565), (185, 190, 194), 2)

    cv2.ellipse(image, (610, 360), (95, 36), 22, 0, 360, (45, 48, 52), 3)
    for offset in range(-30, 45, 15):
        cv2.line(image, (530, 355 + offset), (690, 365 - offset), (28, 31, 35), 2)
    cv2.circle(image, (292, 445), 34, (120, 60, 45), -1)
    cv2.imwrite(str(output_dir / "manufacturing_unknown_patterns.png"), image)
    print(output_dir / "manufacturing_unknown_patterns.png")


if __name__ == "__main__":
    main()


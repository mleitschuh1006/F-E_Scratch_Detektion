from pathlib import Path
import random

import cv2
import numpy as np

from utils.dataset import (
    create_tiled_dataset,
    find_image_mask_pairs,
    split_dataset,
)
from utils.labels import (
    create_yolo_label_files,
    mask_to_yolo_segmentation_labels,
)
from utils.visualization import draw_title


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "dataset_tiled_seg"

TILE_SIZE = 320
OVERLAP = 0.20

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42

MASK_THRESHOLD = 127
MIN_AREA_PX = 1

CLASS_ID = 0


# ============================================================
# Instance-segmentation control visualization
# ============================================================

def visualize_random_samples(
    dataset_dir: Path,
    output_dir: Path,
    split: str = "train",
    num_samples: int = 20,
    seed: int = 42,
) -> None:

    image_dir = dataset_dir / "images" / split
    mask_dir = dataset_dir / "masks" / split
    label_dir = dataset_dir / "labels" / split

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(image_dir.glob("*.png"))

    if not image_paths:
        raise ValueError(f"No images found in: {image_dir}")

    selected_images = random.Random(seed).sample(
        image_paths,
        min(num_samples, len(image_paths)),
    )

    print(f"\nCreating {len(selected_images)} control images from '{split}' split...")

    for image_path in selected_images:
        mask_path = mask_dir / image_path.name
        label_path = label_dir / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Could not load pair: {image_path.name}")
            continue

        image_height, image_width = image.shape[:2]
        mask_visualization = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        polygon_image = image.copy()

        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue

                values = line.split()
                coordinates = list(map(float, values[1:]))

                if len(coordinates) < 6 or len(coordinates) % 2 != 0:
                    print(f"Warning: Invalid polygon in {label_path.name}: {line}")
                    continue

                polygon_points = []

                for i in range(0, len(coordinates), 2):
                    x = int(round(coordinates[i] * image_width))
                    y = int(round(coordinates[i + 1] * image_height))
                    x = np.clip(x, 0, image_width - 1)
                    y = np.clip(y, 0, image_height - 1)
                    polygon_points.append([x, y])

                polygon_points = np.array(
                    polygon_points,
                    dtype=np.int32,
                ).reshape((-1, 1, 2))

                cv2.polylines(
                    polygon_image,
                    [polygon_points],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=1,
                )

        draw_title(image, "Original")
        draw_title(mask_visualization, "Mask")
        draw_title(polygon_image, "YOLO Segmentation Polygons")

        comparison = cv2.hconcat([image, mask_visualization, polygon_image])
        cv2.imwrite(
            str(output_dir / f"{image_path.stem}_check.png"),
            comparison,
        )

    print(f"Control images saved to: {output_dir}")


# ============================================================
# Main
# ============================================================

def main() -> None:

    pairs = find_image_mask_pairs(SOURCE_DIR)

    splits = split_dataset(
        pairs=pairs,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED,
    )

    create_tiled_dataset(
        splits=splits,
        output_dir=OUTPUT_DIR,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
    )

    create_yolo_label_files(
        dataset_dir=OUTPUT_DIR,
        converter=mask_to_yolo_segmentation_labels,
        label_description="YOLO segmentation",
        class_id=CLASS_ID,
        threshold=MASK_THRESHOLD,
        min_area_px=MIN_AREA_PX,
    )

    visualize_random_samples(
        dataset_dir=OUTPUT_DIR,
        output_dir=OUTPUT_DIR / "control",
        split="train",
        num_samples=20,
        seed=RANDOM_SEED,
    )


if __name__ == "__main__":
    main()

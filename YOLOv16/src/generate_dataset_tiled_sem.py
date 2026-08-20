from functools import partial
from pathlib import Path
import random

import cv2

from utils.dataset import (
    create_tiled_dataset,
    find_image_mask_pairs,
    split_dataset,
)
from utils.labels import (
    check_semantic_masks,
    convert_to_semantic_mask,
)
from utils.visualization import (
    create_comparison_image,
    draw_mask_overlay,
)


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "dataset_tiled_sem"

TILE_SIZE = 320
OVERLAP = 0.20

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42
MASK_THRESHOLD = 127
SCRATCH_CLASS_ID = 1

# ============================================================
# Semantic control visualization
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

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Could not load pair: {image_path.name}")
            continue

        mask_visualization = (mask * 255).astype("uint8")
        overlay_image = draw_mask_overlay(
            image=image,
            scratch_mask=(mask == SCRATCH_CLASS_ID).astype("uint8"),
        )

        comparison = create_comparison_image(
            original_image=image,
            mask_visualization=mask_visualization,
            overlay_image=overlay_image,
            titles=("Original", "Semantic Mask (0/1)", "Scratch Overlay"),
        )

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

    semantic_transform = partial(
        convert_to_semantic_mask,
        threshold=MASK_THRESHOLD,
        scratch_class_id=SCRATCH_CLASS_ID,
    )

    create_tiled_dataset(
        splits=splits,
        output_dir=OUTPUT_DIR,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        mask_transform=semantic_transform,
    )

    check_semantic_masks(
        dataset_dir=OUTPUT_DIR,
        scratch_class_id=SCRATCH_CLASS_ID,
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

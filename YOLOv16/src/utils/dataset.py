"""Shared helpers for preparing tiled image/mask datasets."""

from collections.abc import Callable
from pathlib import Path
import random

import cv2
import numpy as np

from utils.tiling import calculate_tile_positions


MaskTransform = Callable[[np.ndarray], np.ndarray]


def find_image_mask_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    """Find PNG image/mask pairs with identical filenames."""

    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"

    pairs: list[tuple[Path, Path]] = []

    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            print(f"Warning: No mask found for {image_path.name}")
            continue

        pairs.append((image_path, mask_path))

    print(f"Found {len(pairs)} image-mask pairs.")
    return pairs


def split_dataset(
    pairs: list[tuple[Path, Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, list[tuple[Path, Path]]]:
    """Split original images before tiling to prevent data leakage."""

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation and test ratios must sum to 1.0.")

    shuffled_pairs = pairs.copy()
    random.Random(seed).shuffle(shuffled_pairs)

    num_images = len(shuffled_pairs)
    train_end = int(num_images * train_ratio)
    val_end = train_end + int(num_images * val_ratio)

    splits = {
        "train": shuffled_pairs[:train_end],
        "val": shuffled_pairs[train_end:val_end],
        "test": shuffled_pairs[val_end:],
    }

    print("\nDataset split:")
    print(f"Train: {len(splits['train'])}")
    print(f"Val:   {len(splits['val'])}")
    print(f"Test:  {len(splits['test'])}")

    return splits


def tile_image_and_mask(
    image_path: Path,
    mask_path: Path,
    image_output_dir: Path,
    mask_output_dir: Path,
    tile_size: int,
    overlap: float,
    mask_transform: MaskTransform | None = None,
) -> int:
    """Tile one image and its mask using identical coordinates.

    mask_transform can be used for task-specific preprocessing, e.g. converting
    a 0/255 binary mask into semantic class IDs 0/1.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    image_height, image_width = image.shape[:2]
    mask_height, mask_width = mask.shape[:2]

    if (image_width, image_height) != (mask_width, mask_height):
        raise ValueError(
            f"Image and mask dimensions do not match:\n"
            f"{image_path}: {image_width}x{image_height}\n"
            f"{mask_path}: {mask_width}x{mask_height}"
        )

    if mask_transform is not None:
        mask = mask_transform(mask)

    image_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    positions = calculate_tile_positions(
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        overlap=overlap,
    )

    stem = image_path.stem

    for x, y in positions:
        x2 = min(x + tile_size, image_width)
        y2 = min(y + tile_size, image_height)

        image_tile = image[y:y2, x:x2]
        mask_tile = mask[y:y2, x:x2]

        tile_name = f"{stem}_x{x:04d}_y{y:04d}.png"

        cv2.imwrite(str(image_output_dir / tile_name), image_tile)
        cv2.imwrite(str(mask_output_dir / tile_name), mask_tile)

    return len(positions)


def create_tiled_dataset(
    splits: dict[str, list[tuple[Path, Path]]],
    output_dir: Path,
    tile_size: int,
    overlap: float,
    mask_transform: MaskTransform | None = None,
) -> None:
    """Tile all image/mask pairs while preserving train/val/test splits."""

    for split_name, pairs in splits.items():
        image_output_dir = output_dir / "images" / split_name
        mask_output_dir = output_dir / "masks" / split_name

        total_tiles = 0
        print(f"\nCreating {split_name} tiles...")

        for image_path, mask_path in pairs:
            total_tiles += tile_image_and_mask(
                image_path=image_path,
                mask_path=mask_path,
                image_output_dir=image_output_dir,
                mask_output_dir=mask_output_dir,
                tile_size=tile_size,
                overlap=overlap,
                mask_transform=mask_transform,
            )

        print(f"{split_name}: {total_tiles} tiles created.")

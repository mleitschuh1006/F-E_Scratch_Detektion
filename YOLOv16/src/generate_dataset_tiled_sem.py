from pathlib import Path
import random

import cv2
import numpy as np


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

BACKGROUND_CLASS_ID = 0
SCRATCH_CLASS_ID = 1


# ============================================================
# 1. Find matching image-mask pairs
# ============================================================

def find_image_mask_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"

    pairs = []

    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            print(f"Warning: No mask found for {image_path.name}")
            continue

        pairs.append((image_path, mask_path))

    print(f"Found {len(pairs)} image-mask pairs.")
    return pairs


# ============================================================
# 2. Split images into train / validation / test
# ============================================================

def split_dataset(
    pairs: list[tuple[Path, Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> dict:

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation and test ratios must sum to 1.0.")

    pairs = pairs.copy()

    random.seed(seed)
    random.shuffle(pairs)

    num_images = len(pairs)

    train_end = int(num_images * train_ratio)
    val_end = train_end + int(num_images * val_ratio)

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    print("\nDataset split:")
    print(f"Train: {len(splits['train'])}")
    print(f"Val:   {len(splits['val'])}")
    print(f"Test:  {len(splits['test'])}")

    return splits


# ============================================================
# 3. Calculate tile positions
# ============================================================

def calculate_tile_positions(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: float,
) -> list[tuple[int, int]]:

    stride = int(tile_size * (1 - overlap))

    if stride <= 0:
        raise ValueError("Overlap is too large.")

    def calculate_positions(length: int) -> list[int]:
        if length <= tile_size:
            return [0]

        positions = list(range(0, length - tile_size + 1, stride))
        final_position = length - tile_size

        if positions[-1] != final_position:
            positions.append(final_position)

        return positions

    x_positions = calculate_positions(image_width)
    y_positions = calculate_positions(image_height)

    return [(x, y) for y in y_positions for x in x_positions]


# ============================================================
# 4. Convert binary annotation mask to semantic class mask
# ============================================================

def convert_to_semantic_mask(
    mask: np.ndarray,
    threshold: int = 127,
) -> np.ndarray:
    """
    Convert original binary mask to semantic class IDs.

    Input:
        background = 0
        scratch    = typically 255

    Output:
        background = 0
        scratch    = 1

    Ultralytics semantic segmentation uses 255 as ignore label,
    so scratch pixels must not remain 255.
    """

    semantic_mask = np.zeros(mask.shape, dtype=np.uint8)
    semantic_mask[mask > threshold] = SCRATCH_CLASS_ID

    return semantic_mask


# ============================================================
# 5. Tile one image-mask pair
# ============================================================

def tile_image_and_mask(
    image_path: Path,
    mask_path: Path,
    image_output_dir: Path,
    mask_output_dir: Path,
    tile_size: int = 320,
    overlap: float = 0.20,
    threshold: int = 127,
) -> int:

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

    image_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    semantic_mask = convert_to_semantic_mask(
        mask=mask,
        threshold=threshold,
    )

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
        mask_tile = semantic_mask[y:y2, x:x2]

        tile_name = f"{stem}_x{x:04d}_y{y:04d}.png"

        cv2.imwrite(str(image_output_dir / tile_name), image_tile)

        # Save raw class IDs 0 and 1.
        cv2.imwrite(str(mask_output_dir / tile_name), mask_tile)

    return len(positions)


# ============================================================
# 6. Tile complete dataset
# ============================================================

def create_tiled_dataset(
    splits: dict,
    output_dir: Path,
    tile_size: int = 320,
    overlap: float = 0.20,
    threshold: int = 127,
):

    for split_name, pairs in splits.items():

        image_output_dir = output_dir / "images" / split_name
        mask_output_dir = output_dir / "masks" / split_name

        total_tiles = 0

        print(f"\nCreating {split_name} tiles...")

        for image_path, mask_path in pairs:

            num_tiles = tile_image_and_mask(
                image_path=image_path,
                mask_path=mask_path,
                image_output_dir=image_output_dir,
                mask_output_dir=mask_output_dir,
                tile_size=tile_size,
                overlap=overlap,
                threshold=threshold,
            )

            total_tiles += num_tiles

        print(f"{split_name}: {total_tiles} tiles created.")


# ============================================================
# 7. Check semantic mask values
# ============================================================

def check_semantic_masks(dataset_dir: Path):

    print("\nChecking semantic mask values...")

    for split_name in ["train", "val", "test"]:

        mask_dir = dataset_dir / "masks" / split_name
        mask_paths = sorted(mask_dir.glob("*.png"))

        scratch_tiles = 0
        background_only_tiles = 0

        for mask_path in mask_paths:

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {mask_path}")

            unique_values = set(np.unique(mask).tolist())

            if not unique_values.issubset({0, 1}):
                raise ValueError(
                    f"Invalid semantic values in {mask_path}: "
                    f"{sorted(unique_values)}"
                )

            if np.any(mask == SCRATCH_CLASS_ID):
                scratch_tiles += 1
            else:
                background_only_tiles += 1

        print(
            f"{split_name}: "
            f"{len(mask_paths)} masks | "
            f"{scratch_tiles} with scratch | "
            f"{background_only_tiles} background only"
        )


# ============================================================
# 8. Create Ultralytics semantic dataset YAML
# ============================================================

def create_dataset_yaml(dataset_dir: Path):

    yaml_content = f"""path: {dataset_dir.resolve()}
train: images/train
val: images/val
test: images/test

masks_dir: masks

names:
  0: background
  1: scratch
"""

    yaml_path = dataset_dir / "dataset_sem.yaml"

    yaml_path.write_text(
        yaml_content,
        encoding="utf-8",
    )

    print(f"\nDataset YAML created: {yaml_path}")


# ============================================================
# 9. Visualize random samples
# ============================================================

def visualize_random_samples(
    dataset_dir: Path,
    output_dir: Path,
    split: str = "train",
    num_samples: int = 20,
    seed: int = 42,
):

    image_dir = dataset_dir / "images" / split
    mask_dir = dataset_dir / "masks" / split

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.png"))

    if not image_paths:
        raise ValueError(f"No images found in: {image_dir}")

    random.seed(seed)

    num_samples = min(num_samples, len(image_paths))
    selected_images = random.sample(image_paths, num_samples)

    print(
        f"\nCreating {num_samples} control images "
        f"from '{split}' split..."
    )

    for image_path in selected_images:

        mask_path = mask_dir / image_path.name

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not load image: {image_path}")
            continue

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            print(f"Could not load mask: {mask_path}")
            continue

        # Class ID 1 is scaled to white only for visualization.
        mask_visual = (mask * 255).astype(np.uint8)

        mask_visualization = cv2.cvtColor(
            mask_visual,
            cv2.COLOR_GRAY2BGR,
        )

        overlay_image = image.copy()

        scratch_pixels = mask == SCRATCH_CLASS_ID

        overlay_image[scratch_pixels] = (
            0.5 * overlay_image[scratch_pixels]
            + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)

        cv2.putText(
            image,
            "Original",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            mask_visualization,
            "Semantic Mask (0/1)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            overlay_image,
            "Scratch Overlay",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        comparison = cv2.hconcat(
            [
                image,
                mask_visualization,
                overlay_image,
            ]
        )

        output_path = (
            output_dir /
            f"{image_path.stem}_check.png"
        )

        cv2.imwrite(
            str(output_path),
            comparison,
        )

    print(f"Control images saved to: {output_dir}")


# ============================================================
# Main
# ============================================================

def main():

    pairs = find_image_mask_pairs(
        SOURCE_DIR
    )

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
        threshold=MASK_THRESHOLD,
    )

    check_semantic_masks(
        dataset_dir=OUTPUT_DIR,
    )

    create_dataset_yaml(
        dataset_dir=OUTPUT_DIR,
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

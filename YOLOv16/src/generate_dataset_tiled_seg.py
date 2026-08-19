from pathlib import Path
import random

import cv2
import numpy as np


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
CLASS_NAME = "scratch"


# ============================================================
# 1. Find matching image-mask pairs
# ============================================================

def find_image_mask_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    """
    Find image-mask pairs with identical filenames.

    Example:
        images/71_max_flat.png
        masks/71_max_flat.png
    """

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
    """
    Split complete original images into train, validation and test.

    Important:
    Splitting is done BEFORE tiling to prevent data leakage.
    """

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
    """
    Calculate top-left positions of overlapping tiles.

    The final tiles are shifted so that the full image
    is always covered.
    """

    stride = int(tile_size * (1 - overlap))

    if stride <= 0:
        raise ValueError("Overlap is too large.")

    def calculate_positions(length: int) -> list[int]:

        # Image smaller than tile
        if length <= tile_size:
            return [0]

        positions = list(range(0, length - tile_size + 1, stride))

        final_position = length - tile_size

        if positions[-1] != final_position:
            positions.append(final_position)

        return positions

    x_positions = calculate_positions(image_width)
    y_positions = calculate_positions(image_height)

    positions = []

    for y in y_positions:
        for x in x_positions:
            positions.append((x, y))

    return positions


# ============================================================
# 4. Tile one image-mask pair
# ============================================================

def tile_image_and_mask(
    image_path: Path,
    mask_path: Path,
    image_output_dir: Path,
    mask_output_dir: Path,
    tile_size: int = 1024,
    overlap: float = 0.20,
) -> int:
    """
    Cut one image and its corresponding mask into identical tiles.
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

    image_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    positions = calculate_tile_positions(
        image_width,
        image_height,
        tile_size,
        overlap,
    )

    stem = image_path.stem

    for x, y in positions:

        x2 = min(x + tile_size, image_width)
        y2 = min(y + tile_size, image_height)

        image_tile = image[y:y2, x:x2]
        mask_tile = mask[y:y2, x:x2]

        tile_name = f"{stem}_x{x:04d}_y{y:04d}.png"

        cv2.imwrite(
            str(image_output_dir / tile_name),
            image_tile,
        )

        cv2.imwrite(
            str(mask_output_dir / tile_name),
            mask_tile,
        )

    return len(positions)


# ============================================================
# 5. Tile complete dataset
# ============================================================

def create_tiled_dataset(
    splits: dict,
    output_dir: Path,
    tile_size: int = 1024,
    overlap: float = 0.20,
):
    """
    Tile all image-mask pairs while keeping train/val/test separated.
    """

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
            )

            total_tiles += num_tiles

        print(f"{split_name}: {total_tiles} tiles created.")


# ============================================================
# 6. Convert one mask to YOLO segmentation polygons
# ============================================================

def mask_to_yolo_segmentation_labels(
    mask_path: Path,
    class_id: int = 0,
    threshold: int = 127,
    min_area_px: int = 1,
) -> list[str]:
    """
    Convert connected white components of a binary mask
    into YOLO segmentation labels.

    Each connected component = one scratch.

    YOLO segmentation format:
        class_id x1 y1 x2 y2 x3 y3 ...

    All x/y coordinates are normalized to 0...1.
    """

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    image_height, image_width = mask.shape

    # Convert to binary mask
    binary_mask = (mask > threshold).astype(np.uint8)

    # Detect connected components.
    # This preserves the same "one connected component = one scratch"
    # logic as in the detection version of the script.
    num_labels, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    yolo_labels = []

    # Component 0 = background
    for component_id in range(1, num_labels):

        area = stats[component_id, cv2.CC_STAT_AREA]

        if area < min_area_px:
            continue

        # Create a binary mask containing only this scratch
        component_mask = (
            (component_labels == component_id).astype(np.uint8) * 255
        )

        # Extract the outer contour of the scratch
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        # A connected component normally has one outer contour.
        # Use the longest contour if OpenCV returns more than one.
        contour = max(contours, key=lambda c: len(c))

        points = contour.reshape(-1, 2)

        # YOLO segmentation needs at least 3 polygon points.
        # Very tiny components can collapse to 1-2 contour points.
        # In that case, use the component's pixel bounding rectangle
        # as the smallest valid polygon representation.
        if len(points) < 3:

            x = stats[component_id, cv2.CC_STAT_LEFT]
            y = stats[component_id, cv2.CC_STAT_TOP]
            width = stats[component_id, cv2.CC_STAT_WIDTH]
            height = stats[component_id, cv2.CC_STAT_HEIGHT]

            x2 = min(x + width, image_width)
            y2 = min(y + height, image_height)

            points = np.array(
                [
                    [x, y],
                    [x2, y],
                    [x2, y2],
                    [x, y2],
                ],
                dtype=np.float32,
            )

        # Normalize polygon coordinates to 0...1
        normalized_points = []

        for x, y in points:
            x_norm = np.clip(float(x) / image_width, 0.0, 1.0)
            y_norm = np.clip(float(y) / image_height, 0.0, 1.0)

            normalized_points.extend(
                [
                    f"{x_norm:.6f}",
                    f"{y_norm:.6f}",
                ]
            )

        label = f"{class_id} " + " ".join(normalized_points)

        yolo_labels.append(label)

    return yolo_labels


# ============================================================
# 7. Generate YOLO segmentation labels for all mask tiles
# ============================================================

def create_yolo_segmentation_labels(
    dataset_dir: Path,
    threshold: int = 127,
    min_area_px: int = 1,
    class_id: int = 0,
):
    """
    Generate YOLO segmentation .txt label files for all mask tiles.
    """

    for split_name in ["train", "val", "test"]:

        mask_dir = dataset_dir / "masks" / split_name
        label_dir = dataset_dir / "labels" / split_name

        label_dir.mkdir(parents=True, exist_ok=True)

        mask_paths = sorted(mask_dir.glob("*.png"))

        total_objects = 0

        print(f"\nCreating YOLO segmentation labels for {split_name}...")

        for mask_path in mask_paths:

            labels = mask_to_yolo_segmentation_labels(
                mask_path=mask_path,
                class_id=class_id,
                threshold=threshold,
                min_area_px=min_area_px,
            )

            label_path = label_dir / f"{mask_path.stem}.txt"

            label_path.write_text(
                "\n".join(labels),
                encoding="utf-8",
            )

            total_objects += len(labels)

        print(
            f"{split_name}: "
            f"{len(mask_paths)} label files, "
            f"{total_objects} scratches."
        )


# ============================================================
# 8. Visualize YOLO segmentation labels
# ============================================================

def visualize_random_samples(
    dataset_dir: Path,
    output_dir: Path,
    split: str = "train",
    num_samples: int = 20,
    seed: int = 42,
):
    """
    Randomly select tiled images and visualize:

    1. original image tile
    2. corresponding binary mask
    3. original image with YOLO segmentation polygons

    The three images are placed next to each other and saved
    as one control image.
    """

    image_dir = dataset_dir / "images" / split
    mask_dir = dataset_dir / "masks" / split
    label_dir = dataset_dir / "labels" / split

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
        label_path = label_dir / f"{image_path.stem}.txt"

        # ----------------------------------------------------
        # Load original image
        # ----------------------------------------------------

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not load image: {image_path}")
            continue

        image_height, image_width = image.shape[:2]

        # ----------------------------------------------------
        # Load mask
        # ----------------------------------------------------

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            print(f"Could not load mask: {mask_path}")
            continue

        mask_visualization = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR,
        )

        # ----------------------------------------------------
        # Copy image for polygon visualization
        # ----------------------------------------------------

        polygon_image = image.copy()

        # ----------------------------------------------------
        # Read YOLO segmentation labels
        # ----------------------------------------------------

        if label_path.exists():

            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

            for line in lines:

                if not line.strip():
                    continue

                values = line.split()

                class_id = int(values[0])
                coordinates = list(map(float, values[1:]))

                if len(coordinates) < 6 or len(coordinates) % 2 != 0:
                    print(
                        f"Warning: Invalid polygon in {label_path.name}: "
                        f"{line}"
                    )
                    continue

                polygon_points = []

                for i in range(0, len(coordinates), 2):

                    x_norm = coordinates[i]
                    y_norm = coordinates[i + 1]

                    x = int(round(x_norm * image_width))
                    y = int(round(y_norm * image_height))

                    x = np.clip(x, 0, image_width - 1)
                    y = np.clip(y, 0, image_height - 1)

                    polygon_points.append([x, y])

                polygon_points = np.array(
                    polygon_points,
                    dtype=np.int32,
                ).reshape((-1, 1, 2))

                # Draw polygon outline
                cv2.polylines(
                    polygon_image,
                    [polygon_points],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=1,
                )

        # ----------------------------------------------------
        # Add titles
        # ----------------------------------------------------

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
            "Mask",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            polygon_image,
            "YOLO Segmentation Polygons",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # ----------------------------------------------------
        # Put images next to each other
        # ----------------------------------------------------

        comparison = cv2.hconcat(
            [
                image,
                mask_visualization,
                polygon_image,
            ]
        )

        # ----------------------------------------------------
        # Save comparison
        # ----------------------------------------------------

        output_path = (
            output_dir /
            f"{image_path.stem}_check.png"
        )

        cv2.imwrite(
            str(output_path),
            comparison,
        )

    print(
        f"Control images saved to: {output_dir}"
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Step 1:
    # Find original image-mask pairs
    # --------------------------------------------------------

    pairs = find_image_mask_pairs(
        SOURCE_DIR
    )

    # --------------------------------------------------------
    # Step 2:
    # Split ORIGINAL images into train / val / test
    # --------------------------------------------------------

    splits = split_dataset(
        pairs=pairs,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # Step 3:
    # Tile images and masks
    # --------------------------------------------------------

    create_tiled_dataset(
        splits=splits,
        output_dir=OUTPUT_DIR,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
    )

    # --------------------------------------------------------
    # Step 4:
    # Convert tiled masks into YOLO segmentation polygons
    # --------------------------------------------------------

    create_yolo_segmentation_labels(
        dataset_dir=OUTPUT_DIR,
        threshold=MASK_THRESHOLD,
        min_area_px=MIN_AREA_PX,
        class_id=CLASS_ID,
    )

    # --------------------------------------------------------
    # Step 5:
    # Visual inspection
    # --------------------------------------------------------

    visualize_random_samples(
        dataset_dir=OUTPUT_DIR,
        output_dir=OUTPUT_DIR / "control",
        split="train",
        num_samples=20,
        seed=42,
    )


if __name__ == "__main__":
    main()

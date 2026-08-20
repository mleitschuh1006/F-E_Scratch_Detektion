"""Conversions from binary masks to YOLO detection/segmentation labels."""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np


def mask_to_yolo_detection_labels(
    mask_path: Path,
    class_id: int = 0,
    threshold: int = 127,
    min_area_px: int = 1,
) -> list[str]:
    """Convert connected mask components to YOLO bounding-box labels."""

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    image_height, image_width = mask.shape
    binary_mask = (mask > threshold).astype(np.uint8)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    labels: list[str] = []

    for component_id in range(1, num_labels):
        x = stats[component_id, cv2.CC_STAT_LEFT]
        y = stats[component_id, cv2.CC_STAT_TOP]
        width = stats[component_id, cv2.CC_STAT_WIDTH]
        height = stats[component_id, cv2.CC_STAT_HEIGHT]
        area = stats[component_id, cv2.CC_STAT_AREA]

        if area < min_area_px:
            continue

        x_center = (x + width / 2) / image_width
        y_center = (y + height / 2) / image_height
        width_norm = width / image_width
        height_norm = height / image_height

        labels.append(
            f"{class_id} "
            f"{x_center:.6f} {y_center:.6f} "
            f"{width_norm:.6f} {height_norm:.6f}"
        )

    return labels


def mask_to_yolo_segmentation_labels(
    mask_path: Path,
    class_id: int = 0,
    threshold: int = 127,
    min_area_px: int = 1,
) -> list[str]:
    """Convert connected mask components to YOLO polygon labels."""

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    image_height, image_width = mask.shape
    binary_mask = (mask > threshold).astype(np.uint8)

    num_labels, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    labels: list[str] = []

    for component_id in range(1, num_labels):
        area = stats[component_id, cv2.CC_STAT_AREA]

        if area < min_area_px:
            continue

        component_mask = (
            (component_labels == component_id).astype(np.uint8) * 255
        )

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        contour = max(contours, key=lambda item: len(item))
        points = contour.reshape(-1, 2)

        # YOLO polygons need at least three points.
        if len(points) < 3:
            x = stats[component_id, cv2.CC_STAT_LEFT]
            y = stats[component_id, cv2.CC_STAT_TOP]
            width = stats[component_id, cv2.CC_STAT_WIDTH]
            height = stats[component_id, cv2.CC_STAT_HEIGHT]

            x2 = min(x + width, image_width)
            y2 = min(y + height, image_height)

            points = np.array(
                [[x, y], [x2, y], [x2, y2], [x, y2]],
                dtype=np.float32,
            )

        normalized_points: list[str] = []

        for x, y in points:
            x_norm = np.clip(float(x) / image_width, 0.0, 1.0)
            y_norm = np.clip(float(y) / image_height, 0.0, 1.0)
            normalized_points.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])

        labels.append(f"{class_id} " + " ".join(normalized_points))

    return labels


def create_yolo_label_files(
    dataset_dir: Path,
    converter: Callable[..., list[str]],
    label_description: str,
    **converter_kwargs,
) -> None:
    """Generate one YOLO txt label file for every tiled mask."""

    for split_name in ["train", "val", "test"]:
        mask_dir = dataset_dir / "masks" / split_name
        label_dir = dataset_dir / "labels" / split_name
        label_dir.mkdir(parents=True, exist_ok=True)

        mask_paths = sorted(mask_dir.glob("*.png"))
        total_objects = 0

        print(f"\nCreating {label_description} labels for {split_name}...")

        for mask_path in mask_paths:
            labels = converter(mask_path=mask_path, **converter_kwargs)
            label_path = label_dir / f"{mask_path.stem}.txt"
            label_path.write_text("\n".join(labels), encoding="utf-8")
            total_objects += len(labels)

        print(
            f"{split_name}: {len(mask_paths)} label files, "
            f"{total_objects} scratches."
        )


def convert_to_semantic_mask(
    mask: np.ndarray,
    threshold: int = 127,
    scratch_class_id: int = 1,
) -> np.ndarray:
    """Convert a 0/255 mask into a semantic class-ID mask."""

    semantic_mask = np.zeros(mask.shape, dtype=np.uint8)
    semantic_mask[mask > threshold] = scratch_class_id
    return semantic_mask


def check_semantic_masks(
    dataset_dir: Path,
    allowed_values: set[int] | None = None,
    scratch_class_id: int = 1,
) -> None:
    """Validate semantic masks and report positive/negative tile counts."""

    if allowed_values is None:
        allowed_values = {0, scratch_class_id}

    print("\nChecking semantic mask values...")

    for split_name in ["train", "val", "test"]:
        mask_dir = dataset_dir / "masks" / split_name
        mask_paths = sorted(mask_dir.glob("*.png"))

        scratch_tiles = 0
        background_only_tiles = 0

        for mask_path in mask_paths:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {mask_path}")

            unique_values = set(np.unique(mask).tolist())

            if not unique_values.issubset(allowed_values):
                raise ValueError(
                    f"Invalid semantic values in {mask_path}: "
                    f"{sorted(unique_values)}"
                )

            if np.any(mask == scratch_class_id):
                scratch_tiles += 1
            else:
                background_only_tiles += 1

        print(
            f"{split_name}: {len(mask_paths)} masks | "
            f"{scratch_tiles} with scratch | "
            f"{background_only_tiles} background only"
        )

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "yolo26n_320_seg_scratch"
    / "weights"
    / "best.pt"
)

INPUT_IMAGE = (
    PROJECT_DIR
    / "prediction_images"
    / "example.png"
)

OUTPUT_DIR = PROJECT_DIR / "prediction_results"


# ============================================================
# Configuration
# ============================================================

TILE_SIZE = 320
OVERLAP = 0.20

CONF_THRESHOLD = 0.25
SCRATCH_CLASS_ID = 0

# For overlapping tiles:
# 0.50 means a pixel is kept when at least half of the
# covering tiles predict it as scratch.
SCRATCH_VOTE_THRESHOLD = 0.50


# ============================================================
# Calculate tile positions
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

        positions = list(
            range(
                0,
                length - tile_size + 1,
                stride,
            )
        )

        final_position = length - tile_size

        if positions[-1] != final_position:
            positions.append(final_position)

        return positions

    x_positions = calculate_positions(image_width)
    y_positions = calculate_positions(image_height)

    return [
        (x, y)
        for y in y_positions
        for x in x_positions
    ]


# ============================================================
# Predict complete image using tiles
# ============================================================

def predict_tiled_image(
    model: YOLO,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:

    image_height, image_width = image.shape[:2]

    positions = calculate_tile_positions(
        image_width=image_width,
        image_height=image_height,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
    )

    print(f"Number of tiles: {len(positions)}")

    scratch_votes = np.zeros(
        (image_height, image_width),
        dtype=np.uint16,
    )

    coverage = np.zeros(
        (image_height, image_width),
        dtype=np.uint16,
    )

    total_instances = 0

    for tile_index, (x_offset, y_offset) in enumerate(positions):

        x2 = min(x_offset + TILE_SIZE, image_width)
        y2 = min(y_offset + TILE_SIZE, image_height)

        tile = image[
            y_offset:y2,
            x_offset:x2,
        ]

        tile_height, tile_width = tile.shape[:2]

        # Count how many tiles cover each pixel.
        coverage[
            y_offset:y2,
            x_offset:x2,
        ] += 1

        # ----------------------------------------------------
        # YOLO instance segmentation
        # ----------------------------------------------------

        results = model.predict(
            source=tile,
            imgsz=TILE_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False,
        )

        result = results[0]

        if (
            result.masks is None
            or result.boxes is None
            or len(result.boxes) == 0
        ):
            continue

        masks = result.masks.data.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        # One binary scratch mask for the complete tile.
        tile_scratch_mask = np.zeros(
            (tile_height, tile_width),
            dtype=np.uint8,
        )

        for instance_mask, class_id, confidence in zip(
            masks,
            class_ids,
            confidences,
        ):

            if class_id != SCRATCH_CLASS_ID:
                continue

            if confidence < CONF_THRESHOLD:
                continue

            total_instances += 1

            # Resize mask back to actual tile dimensions if needed.
            if instance_mask.shape != (
                tile_height,
                tile_width,
            ):
                instance_mask = cv2.resize(
                    instance_mask.astype(np.float32),
                    (tile_width, tile_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            binary_instance_mask = (
                instance_mask >= 0.5
            ).astype(np.uint8)

            # Merge all detected scratch instances in this tile.
            tile_scratch_mask = np.maximum(
                tile_scratch_mask,
                binary_instance_mask,
            )

        scratch_votes[
            y_offset:y2,
            x_offset:x2,
        ] += tile_scratch_mask.astype(np.uint16)

    # --------------------------------------------------------
    # Merge overlapping tile predictions
    # --------------------------------------------------------

    scratch_ratio = np.zeros(
        (image_height, image_width),
        dtype=np.float32,
    )

    valid_pixels = coverage > 0

    scratch_ratio[valid_pixels] = (
        scratch_votes[valid_pixels]
        / coverage[valid_pixels]
    )

    final_mask = (
        scratch_ratio >= SCRATCH_VOTE_THRESHOLD
    ).astype(np.uint8)

    return final_mask, scratch_ratio, total_instances


# ============================================================
# Draw scratch overlay
# ============================================================

def draw_segmentation(
    image: np.ndarray,
    scratch_mask: np.ndarray,
) -> np.ndarray:

    result_image = image.copy()

    scratch_pixels = scratch_mask == 1

    result_image[scratch_pixels] = (
        0.5 * result_image[scratch_pixels]
        + 0.5 * np.array(
            [0, 255, 0],
            dtype=np.float32,
        )
    ).astype(np.uint8)

    return result_image


# ============================================================
# Create side-by-side comparison
# ============================================================

def create_comparison_image(
    original_image: np.ndarray,
    mask_visualization: np.ndarray,
    overlay_image: np.ndarray,
) -> np.ndarray:

    mask_bgr = cv2.cvtColor(
        mask_visualization,
        cv2.COLOR_GRAY2BGR,
    )

    original_with_title = original_image.copy()
    mask_with_title = mask_bgr.copy()
    overlay_with_title = overlay_image.copy()

    font_scale = max(
        1.5,
        original_image.shape[0] / 180,
    )

    thickness = max(
        2,
        int(font_scale * 2),
    )

    text_x = 25
    text_y = int(30 + 20 * font_scale)

    def draw_title(
        image: np.ndarray,
        text: str,
    ):

        # Black outline
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 3,
            cv2.LINE_AA,
        )

        # Green text
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )

    draw_title(
        original_with_title,
        "Original",
    )

    draw_title(
        mask_with_title,
        "Predicted Mask",
    )

    draw_title(
        overlay_with_title,
        "Scratch Overlay",
    )

    return cv2.hconcat(
        [
            original_with_title,
            mask_with_title,
            overlay_with_title,
        ]
    )


# ============================================================
# Predict image
# ============================================================

def predict_image(
    image_path: Path,
):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model = YOLO(str(MODEL_PATH))

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    print(
        f"Image size: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    (
        scratch_mask,
        scratch_ratio,
        total_instances,
    ) = predict_tiled_image(
        model=model,
        image=image,
    )

    num_scratch_pixels = int(
        np.sum(scratch_mask == 1)
    )

    total_pixels = scratch_mask.size

    scratch_fraction = (
        num_scratch_pixels / total_pixels
    )

    print(
        f"Scratch instances before tile merging: "
        f"{total_instances}"
    )

    print(
        f"Scratch pixels: {num_scratch_pixels}"
    )

    print(
        f"Scratch fraction: "
        f"{scratch_fraction * 100:.4f} %"
    )

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    mask_visualization = (
        scratch_mask * 255
    ).astype(np.uint8)

    overlay_image = draw_segmentation(
        image=image,
        scratch_mask=scratch_mask,
    )

    comparison_image = create_comparison_image(
        original_image=image,
        mask_visualization=mask_visualization,
        overlay_image=overlay_image,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_original.png"
    )

    mask_output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_mask.png"
    )

    overlay_output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_overlay.png"
    )

    comparison_output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_comparison.png"
    )

    cv2.imwrite(
        str(original_output_path),
        image,
    )

    cv2.imwrite(
        str(mask_output_path),
        mask_visualization,
    )

    cv2.imwrite(
        str(overlay_output_path),
        overlay_image,
    )

    cv2.imwrite(
        str(comparison_output_path),
        comparison_image,
    )

    print()
    print(f"Original saved:   {original_output_path}")
    print(f"Mask saved:       {mask_output_path}")
    print(f"Overlay saved:    {overlay_output_path}")
    print(f"Comparison saved: {comparison_output_path}")


# ============================================================
# Main
# ============================================================

def main():

    predict_image(
        INPUT_IMAGE
    )


if __name__ == "__main__":
    main()

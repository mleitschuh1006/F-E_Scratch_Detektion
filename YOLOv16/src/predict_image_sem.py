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
    / "yolo26n_sem_scratch"
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

BACKGROUND_CLASS_ID = 0
SCRATCH_CLASS_ID = 1

# A pixel is classified as scratch if at least this fraction
# of overlapping tiles predicts it as scratch.
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

    stride = int(
        tile_size * (1 - overlap)
    )

    if stride <= 0:
        raise ValueError(
            "Overlap is too large."
        )

    def calculate_positions(
        length: int,
    ) -> list[int]:

        if length <= tile_size:
            return [0]

        positions = list(
            range(
                0,
                length - tile_size + 1,
                stride,
            )
        )

        final_position = (
            length - tile_size
        )

        if positions[-1] != final_position:
            positions.append(
                final_position
            )

        return positions

    x_positions = calculate_positions(
        image_width
    )

    y_positions = calculate_positions(
        image_height
    )

    positions = []

    for y in y_positions:
        for x in x_positions:
            positions.append(
                (x, y)
            )

    return positions


# ============================================================
# Predict complete image using tiles
# ============================================================

def predict_tiled_image(
    model: YOLO,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    image_height, image_width = (
        image.shape[:2]
    )

    positions = calculate_tile_positions(
        image_width=image_width,
        image_height=image_height,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
    )

    print(
        f"Number of tiles: "
        f"{len(positions)}"
    )

    scratch_votes = np.zeros(
        (image_height, image_width),
        dtype=np.uint16,
    )

    coverage = np.zeros(
        (image_height, image_width),
        dtype=np.uint16,
    )

    for tile_index, (
        x_offset,
        y_offset,
    ) in enumerate(positions):

        # ----------------------------------------------------
        # Extract tile
        # ----------------------------------------------------

        x2 = min(
            x_offset + TILE_SIZE,
            image_width,
        )

        y2 = min(
            y_offset + TILE_SIZE,
            image_height,
        )

        tile = image[
            y_offset:y2,
            x_offset:x2,
        ]

        tile_height, tile_width = (
            tile.shape[:2]
        )

        # ----------------------------------------------------
        # YOLO semantic inference
        # ----------------------------------------------------

        results = model.predict(
            source=tile,
            imgsz=TILE_SIZE,
            verbose=False,
        )

        result = results[0]

        if (
            result.semantic_mask is None
            or result.semantic_mask.data is None
        ):
            print(
                f"Warning: No semantic mask "
                f"for tile {tile_index}"
            )
            continue

        semantic_mask = (
            result.semantic_mask.data
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Ensure mask dimensions match tile
        # ----------------------------------------------------

        if semantic_mask.shape != (
            tile_height,
            tile_width,
        ):

            semantic_mask = cv2.resize(
                semantic_mask.astype(
                    np.uint8
                ),
                (
                    tile_width,
                    tile_height,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        # ----------------------------------------------------
        # Extract scratch class
        # ----------------------------------------------------

        scratch_mask = (
            semantic_mask
            == SCRATCH_CLASS_ID
        ).astype(np.uint16)

        # ----------------------------------------------------
        # Add prediction to full-image vote maps
        # ----------------------------------------------------

        scratch_votes[
            y_offset:y2,
            x_offset:x2,
        ] += scratch_mask

        coverage[
            y_offset:y2,
            x_offset:x2,
        ] += 1

    # --------------------------------------------------------
    # Combine overlapping tile predictions
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
        scratch_ratio
        >= SCRATCH_VOTE_THRESHOLD
    ).astype(np.uint8)

    return final_mask, scratch_ratio


# ============================================================
# Draw semantic mask overlay
# ============================================================

def draw_segmentation(
    image: np.ndarray,
    scratch_mask: np.ndarray,
) -> np.ndarray:

    result_image = image.copy()

    scratch_pixels = (
        scratch_mask
        == SCRATCH_CLASS_ID
    )

    # Green overlay on predicted scratch pixels
    result_image[scratch_pixels] = (
        0.5
        * result_image[scratch_pixels]
        + 0.5
        * np.array(
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

    original_with_title = (
        original_image.copy()
    )

    mask_with_title = (
        mask_bgr.copy()
    )

    overlay_with_title = (
        overlay_image.copy()
    )

    cv2.putText(
        original_with_title,
        "Original",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        mask_with_title,
        "Predicted Mask",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        overlay_with_title,
        "Scratch Overlay",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        2,
    )

    comparison = cv2.hconcat(
        [
            original_with_title,
            mask_with_title,
            overlay_with_title,
        ]
    )

    return comparison


# ============================================================
# Predict image
# ============================================================

def predict_image(
    image_path: Path,
):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: "
            f"{MODEL_PATH}"
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = YOLO(
        str(MODEL_PATH)
    )

    # --------------------------------------------------------
    # Load complete image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Could not load image: "
            f"{image_path}"
        )

    print(
        f"Image size: "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    # --------------------------------------------------------
    # Tiled semantic prediction
    # --------------------------------------------------------

    scratch_mask, scratch_ratio = (
        predict_tiled_image(
            model=model,
            image=image,
        )
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    num_scratch_pixels = int(
        np.sum(
            scratch_mask
            == SCRATCH_CLASS_ID
        )
    )

    total_pixels = (
        scratch_mask.size
    )

    scratch_fraction = (
        num_scratch_pixels
        / total_pixels
    )

    print(
        f"Scratch pixels: "
        f"{num_scratch_pixels}"
    )

    print(
        f"Scratch fraction: "
        f"{scratch_fraction * 100:.4f} %"
    )

    # --------------------------------------------------------
    # Create visualizations
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
    # Save all results
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
    print(
        f"Original saved:   "
        f"{original_output_path}"
    )

    print(
        f"Mask saved:       "
        f"{mask_output_path}"
    )

    print(
        f"Overlay saved:    "
        f"{overlay_output_path}"
    )

    print(
        f"Comparison saved: "
        f"{comparison_output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    predict_image(
        INPUT_IMAGE
    )


if __name__ == "__main__":
    main()

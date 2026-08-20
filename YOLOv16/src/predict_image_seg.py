from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from utils.prediction import (
    add_tile_mask_vote,
    create_mask_vote_maps,
    finalize_mask_votes,
)
from utils.tiling import iter_image_tiles
from utils.visualization import (
    create_comparison_image,
    draw_mask_overlay,
    save_mask_prediction_outputs,
)


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

INPUT_IMAGE = PROJECT_DIR / "prediction_images" / "example.png"
OUTPUT_DIR = PROJECT_DIR / "prediction_results"


# ============================================================
# Configuration
# ============================================================

TILE_SIZE = 320
OVERLAP = 0.20
CONF_THRESHOLD = 0.25
SCRATCH_CLASS_ID = 0
SCRATCH_VOTE_THRESHOLD = 0.50


# ============================================================
# Tiled instance segmentation
# ============================================================

def predict_tiled_image(
    model: YOLO,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:

    image_height, image_width = image.shape[:2]
    scratch_votes, coverage = create_mask_vote_maps(image_height, image_width)

    tiles = list(iter_image_tiles(image, TILE_SIZE, OVERLAP))
    print(f"Number of tiles: {len(tiles)}")

    total_instances = 0

    for _, x1, y1, _, _, tile in tiles:
        tile_height, tile_width = tile.shape[:2]
        tile_scratch_mask = np.zeros((tile_height, tile_width), dtype=np.uint8)

        result = model.predict(
            source=tile,
            imgsz=TILE_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False,
        )[0]

        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for instance_mask, class_id, confidence in zip(
                masks,
                class_ids,
                confidences,
            ):
                if class_id != SCRATCH_CLASS_ID or confidence < CONF_THRESHOLD:
                    continue

                total_instances += 1

                if instance_mask.shape != (tile_height, tile_width):
                    instance_mask = cv2.resize(
                        instance_mask.astype(np.float32),
                        (tile_width, tile_height),
                        interpolation=cv2.INTER_LINEAR,
                    )

                binary_mask = (instance_mask >= 0.5).astype(np.uint8)
                tile_scratch_mask = np.maximum(tile_scratch_mask, binary_mask)

        # Also add an all-zero mask when no scratch is detected. This way the
        # tile still contributes to the overlap vote as a background decision.
        add_tile_mask_vote(
            scratch_votes=scratch_votes,
            coverage=coverage,
            tile_mask=tile_scratch_mask,
            x1=x1,
            y1=y1,
        )

    final_mask, scratch_ratio = finalize_mask_votes(
        scratch_votes=scratch_votes,
        coverage=coverage,
        vote_threshold=SCRATCH_VOTE_THRESHOLD,
    )

    return final_mask, scratch_ratio, total_instances


# ============================================================
# Predict image
# ============================================================

def predict_image(image_path: Path) -> None:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(str(MODEL_PATH))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    print(f"Image size: {image.shape[1]} x {image.shape[0]}")

    scratch_mask, _, total_instances = predict_tiled_image(
        model=model,
        image=image,
    )

    num_scratch_pixels = int(np.sum(scratch_mask == 1))
    scratch_fraction = num_scratch_pixels / scratch_mask.size

    print(f"Scratch instances before tile merging: {total_instances}")
    print(f"Scratch pixels: {num_scratch_pixels}")
    print(f"Scratch fraction: {scratch_fraction * 100:.4f} %")

    mask_visualization = (scratch_mask * 255).astype(np.uint8)
    overlay_image = draw_mask_overlay(image, scratch_mask)
    comparison_image = create_comparison_image(
        original_image=image,
        mask_visualization=mask_visualization,
        overlay_image=overlay_image,
    )

    paths = save_mask_prediction_outputs(
        output_dir=OUTPUT_DIR,
        image_stem=image_path.stem,
        original_image=image,
        scratch_mask=scratch_mask,
        overlay_image=overlay_image,
        comparison_image=comparison_image,
    )

    print()
    for name, path in paths.items():
        print(f"{name.capitalize()} saved: {path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    predict_image(INPUT_IMAGE)


if __name__ == "__main__":
    main()

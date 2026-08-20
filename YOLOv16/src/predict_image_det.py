from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from utils.prediction import apply_global_nms
from utils.tiling import iter_image_tiles


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "yolo26s_scratch-4"
    / "weights"
    / "best.pt"
)

INPUT_IMAGE = PROJECT_DIR / "prediction_images" / "example.png"
OUTPUT_DIR = PROJECT_DIR / "prediction_results"


# ============================================================
# Configuration
# ============================================================

TILE_SIZE = 640
OVERLAP = 0.20
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.50
CLASS_NAME = "scratch"


# ============================================================
# Tiled detection
# ============================================================

def predict_tiled_image(
    model: YOLO,
    image: np.ndarray,
) -> list[dict]:

    detections: list[dict] = []
    tiles = list(iter_image_tiles(image, TILE_SIZE, OVERLAP))
    print(f"Number of tiles: {len(tiles)}")

    for tile_index, x1, y1, _, _, tile in tiles:
        result = model.predict(
            source=tile,
            imgsz=TILE_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False,
        )[0]

        if result.boxes is None:
            continue

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()

            detections.append(
                {
                    "box": [
                        float(xyxy[0] + x1),
                        float(xyxy[1] + y1),
                        float(xyxy[2] + x1),
                        float(xyxy[3] + y1),
                    ],
                    "confidence": float(box.conf[0].cpu().item()),
                    "class_id": int(box.cls[0].cpu().item()),
                    "tile_index": tile_index,
                }
            )

    return detections


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
) -> np.ndarray:

    result_image = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection["box"]]
        confidence = detection["confidence"]

        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            result_image,
            f"{CLASS_NAME} {confidence:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return result_image


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

    detections = predict_tiled_image(model=model, image=image)
    print(f"Detections before NMS: {len(detections)}")

    detections = apply_global_nms(
        detections=detections,
        iou_threshold=NMS_IOU_THRESHOLD,
    )
    print(f"Detections after NMS:  {len(detections)}")

    result_image = draw_detections(image, detections)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{image_path.stem}_prediction.png"
    cv2.imwrite(str(output_path), result_image)
    print(f"Result saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    predict_image(INPUT_IMAGE)


if __name__ == "__main__":
    main()

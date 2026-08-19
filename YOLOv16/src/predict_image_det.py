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
    / "yolo26s_scratch-4"
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

TILE_SIZE = 640
OVERLAP = 0.20

CONF_THRESHOLD = 0.25

NMS_IOU_THRESHOLD = 0.50

CLASS_NAME = "scratch"


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
# IoU
# ============================================================

def calculate_iou(
    box_a,
    box_b,
):

    x1 = max(
        box_a[0],
        box_b[0],
    )

    y1 = max(
        box_a[1],
        box_b[1],
    )

    x2 = min(
        box_a[2],
        box_b[2],
    )

    y2 = min(
        box_a[3],
        box_b[3],
    )

    intersection_width = max(
        0,
        x2 - x1,
    )

    intersection_height = max(
        0,
        y2 - y1,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0, box_a[2] - box_a[0])
        * max(0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0, box_b[2] - box_b[0])
        * max(0, box_b[3] - box_b[1])
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# Global NMS
# ============================================================

def apply_global_nms(
    detections: list[dict],
    iou_threshold: float,
) -> list[dict]:

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda detection:
            detection["confidence"],
        reverse=True,
    )

    keep = []

    while detections:

        best = detections.pop(0)

        keep.append(best)

        remaining = []

        for detection in detections:

            if (
                detection["class_id"]
                != best["class_id"]
            ):
                remaining.append(
                    detection
                )

                continue

            iou = calculate_iou(
                best["box"],
                detection["box"],
            )

            if iou < iou_threshold:
                remaining.append(
                    detection
                )

        detections = remaining

    return keep


# ============================================================
# Predict complete image using tiles
# ============================================================

def predict_tiled_image(
    model: YOLO,
    image: np.ndarray,
):

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

    detections = []

    for tile_index, (
        x_offset,
        y_offset,
    ) in enumerate(positions):

        # ----------------------------------------------------
        # Extract tile
        # ----------------------------------------------------

        tile = image[
            y_offset:
                y_offset + TILE_SIZE,

            x_offset:
                x_offset + TILE_SIZE,
        ]

        # ----------------------------------------------------
        # YOLO inference
        # ----------------------------------------------------

        results = model.predict(
            source=tile,

            imgsz=TILE_SIZE,

            conf=CONF_THRESHOLD,

            verbose=False,
        )

        result = results[0]

        # Detection models expose predictions via result.boxes.
        # ----------------------------------------------------

        if result.boxes is None:
            continue

        for box in result.boxes:

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            confidence = float(
                box.conf[0]
                .cpu()
                .item()
            )

            class_id = int(
                box.cls[0]
                .cpu()
                .item()
            )

            # ------------------------------------------------
            # Convert tile coordinates to full-image coordinates
            # ------------------------------------------------

            x1 = float(
                xyxy[0]
                + x_offset
            )

            y1 = float(
                xyxy[1]
                + y_offset
            )

            x2 = float(
                xyxy[2]
                + x_offset
            )

            y2 = float(
                xyxy[3]
                + y_offset
            )

            detections.append(
                {
                    "box": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],

                    "confidence":
                        confidence,

                    "class_id":
                        class_id,

                    "tile_index":
                        tile_index,
                }
            )

    return detections

def draw_detections(
    image: np.ndarray,
    detections: list[dict],
) -> np.ndarray:

    result_image = image.copy()

    for detection in detections:

        x1, y1, x2, y2 = (
            detection["box"]
        )

        confidence = (
            detection["confidence"]
        )

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        # ----------------------------------------------------
        # Bounding Box
        # ----------------------------------------------------

        cv2.rectangle(
            result_image,

            (x1, y1),
            (x2, y2),

            (0, 255, 0),

            2,
        )

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        text = (
            f"{CLASS_NAME} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            result_image,

            text,

            (
                x1,
                max(
                    y1 - 5,
                    15,
                ),
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (0, 255, 0),

            1,

            cv2.LINE_AA,
        )

    return result_image

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
    # Tiled prediction
    # --------------------------------------------------------

    detections = (
        predict_tiled_image(
            model=model,
            image=image,
        )
    )

    print(
        f"Detections before NMS: "
        f"{len(detections)}"
    )

    # --------------------------------------------------------
    # Remove duplicates caused by overlapping tiles
    # --------------------------------------------------------

    detections = apply_global_nms(
        detections=detections,
        iou_threshold=NMS_IOU_THRESHOLD,
    )

    print(
        f"Detections after NMS:  "
        f"{len(detections)}"
    )

    # --------------------------------------------------------
    # Draw detections
    # --------------------------------------------------------

    result_image = draw_detections(
        image=image,
        detections=detections,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_prediction.png"
    )

    cv2.imwrite(
        str(output_path),
        result_image,
    )

    print(
        f"Result saved: "
        f"{output_path}"
    )


def main():

    predict_image(
        INPUT_IMAGE
    )


if __name__ == "__main__":
    main()
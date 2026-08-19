from pathlib import Path

from ultralytics import YOLO


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_CONFIG = PROJECT_DIR / "config" / "dataset.yaml"

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "yolo26s_scratch"
    / "weights"
    / "best.pt"
)

TEST_OUTPUT_DIR = PROJECT_DIR / "models" / "yolo26s_scratch"


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 640
BATCH_SIZE = 8

RUN_NAME = "test"


# ============================================================
# Test model
# ============================================================

def test_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not DATA_CONFIG.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_CONFIG}"
        )

    print("Loading model...")
    print(f"Model:   {MODEL_PATH}")
    print(f"Dataset: {DATA_CONFIG}")

    model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------------
    # Evaluate on test split
    # --------------------------------------------------------

    metrics = model.val(
        data=str(DATA_CONFIG),

        # Important:
        # Use test split instead of validation split
        split="test",

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        project=str(TEST_OUTPUT_DIR),
        name=RUN_NAME,

        save=True,
        plots=True,

        verbose=True,
    )

    # --------------------------------------------------------
    # Print important metrics
    # --------------------------------------------------------

    print("\n========================================")
    print("TEST RESULTS")
    print("========================================")

    print(
        f"mAP50-95: {metrics.box.map:.4f}"
    )

    print(
        f"mAP50:    {metrics.box.map50:.4f}"
    )

    print(
        f"mAP75:    {metrics.box.map75:.4f}"
    )

    print("\nPer-class mAP50-95:")

    for class_id, class_map in enumerate(
        metrics.box.maps
    ):
        print(
            f"Class {class_id}: {class_map:.4f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    test_model()


if __name__ == "__main__":
    main()
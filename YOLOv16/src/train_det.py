from pathlib import Path

from ultralytics import YOLO


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_CONFIG = PROJECT_DIR / "config" / "dataset.yaml"
MODEL_OUTPUT_DIR = PROJECT_DIR / "models"


# ============================================================
# Training configuration
# ============================================================

MODEL_NAME = "yolo26n.pt"

IMAGE_SIZE = 320
EPOCHS = 150
PATIENCE = 20

BATCH_SIZE = 8

RUN_NAME = "yolo26n_scratch"


# ============================================================
# Train model
# ============================================================

def train_yolo():

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not DATA_CONFIG.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_CONFIG}"
        )

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load pretrained YOLO26n model
    # --------------------------------------------------------

    model = YOLO(MODEL_NAME)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    results = model.train(
        data=str(DATA_CONFIG),

        # Image size
        imgsz=IMAGE_SIZE,

        # Training duration
        epochs=EPOCHS,

        # Early stopping
        patience=PATIENCE,

        # Batch size
        batch=BATCH_SIZE,

        # Output directory
        project=str(MODEL_OUTPUT_DIR),
        name=RUN_NAME,

        # Save checkpoints
        save=True,

        # Hardware
        device=0,

        # Data loading
        workers=8,

        # Reproducibility
        seed=42,

        # Show training progress
        verbose=True,

        # ========================================================
        # Augmentation
        # ========================================================

        mosaic = 1.0,

        # Rotation
        degrees=15.0,

        # Translation
        translate=0.05,

        # Scaling
        scale=0.10,

        # Brightness variation
        hsv_v=0.10,
        )

    return results


# ============================================================
# Main
# ============================================================

def main():

    print("Starting YOLO26n training...")
    print(f"Dataset: {DATA_CONFIG}")
    print(f"Models:  {MODEL_OUTPUT_DIR}")

    train_yolo()


if __name__ == "__main__":
    main()
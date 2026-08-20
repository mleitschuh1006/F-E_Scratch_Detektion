from pathlib import Path

from utils.training import train_yolo_model


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_CONFIG = PROJECT_DIR / "dataset_tiled_sem" / "dataset_sem.yaml"
MODEL_OUTPUT_DIR = PROJECT_DIR / "models"


# ============================================================
# Training configuration
# ============================================================

MODEL_NAME = "yolo26n-sem.pt"
IMAGE_SIZE = 640
EPOCHS = 100
PATIENCE = 7
BATCH_SIZE = 8
RUN_NAME = "yolo26n_640_sem_scratch"


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("Starting YOLO26n semantic-segmentation training...")
    print(f"Dataset: {DATA_CONFIG}")
    print(f"Models:  {MODEL_OUTPUT_DIR}")

    train_yolo_model(
        data_config=DATA_CONFIG,
        model_output_dir=MODEL_OUTPUT_DIR,
        model_name=MODEL_NAME,
        image_size=IMAGE_SIZE,
        epochs=EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        run_name=RUN_NAME,
    )


if __name__ == "__main__":
    main()

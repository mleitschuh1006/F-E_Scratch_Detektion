"""Shared Ultralytics training wrapper."""

from pathlib import Path

from ultralytics import YOLO


DEFAULT_AUGMENTATION = {
    "mosaic": 1.0,
    "degrees": 15.0,
    "translate": 0.05,
    "scale": 0.10,
    "hsv_v": 0.10,
}


def train_yolo_model(
    *,
    data_config: Path,
    model_output_dir: Path,
    model_name: str,
    image_size: int,
    epochs: int,
    patience: int,
    batch_size: int,
    run_name: str,
    device=0,
    workers: int = 8,
    seed: int = 42,
    augmentation: dict | None = None,
):
    """Train a YOLO model using the common project settings."""

    if not data_config.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data_config}"
        )

    model_output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)

    train_args = dict(DEFAULT_AUGMENTATION)

    if augmentation is not None:
        train_args.update(augmentation)

    return model.train(
        data=str(data_config),
        imgsz=image_size,
        epochs=epochs,
        patience=patience,
        batch=batch_size,
        project=str(model_output_dir),
        name=run_name,
        save=True,
        device=device,
        workers=workers,
        seed=seed,
        verbose=True,
        **train_args,
    )

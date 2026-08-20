from pathlib import Path
import csv
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from utils.visualization import draw_mask_overlay, draw_title


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Adjust these paths if your run names differ.
DET_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "yolo26n_320_scratch"
    / "weights"
    / "best.pt"
)

SEG_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "yolo26n_320_seg_scratch"
    / "weights"
    / "best.pt"
)

SEM_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "yolo26n_320_sem_scratch"
    / "weights"
    / "best.pt"
)

# Common test set for ALL three models.
# Images and ground-truth masks come exclusively from
# dataset_tiled_sem. The masks already contain semantic class IDs:
# 0 = background, 1 = scratch.
TEST_IMAGE_DIR = (
    PROJECT_DIR
    / "dataset_tiled_sem"
    / "images"
    / "test"
)

TEST_MASK_DIR = (
    PROJECT_DIR
    / "dataset_tiled_sem"
    / "masks"
    / "test"
)

OUTPUT_DIR = PROJECT_DIR / "test_results"
PLOT_DIR = OUTPUT_DIR / "plots"
EXAMPLE_DIR = OUTPUT_DIR / "examples"


# ============================================================
# Configuration
# ============================================================

# Use the inference sizes that correspond to your trained models.
DET_IMAGE_SIZE = 320
SEG_IMAGE_SIZE = 320
SEM_IMAGE_SIZE = 320

DET_CONF_THRESHOLD = 0.25
SEG_CONF_THRESHOLD = 0.25

DET_SCRATCH_CLASS_ID = 0
SEG_SCRATCH_CLASS_ID = 0
SEM_SCRATCH_CLASS_ID = 1


# None = evaluate every image in the test set.
# Set e.g. 50 for a quick test run.
MAX_IMAGES = None

NUM_QUALITATIVE_EXAMPLES = 10
RANDOM_SEED = 42

# A clean / scratched image decision is based on at least one
# scratch pixel. Increase this value if single-pixel noise should
# not count as a positive prediction.
MIN_PIXELS_FOR_POSITIVE_IMAGE = 1


# ============================================================
# Qualitative comparison visualization
# ============================================================

TITLE_FONT_SCALE = 0.65
TITLE_THICKNESS = 2
TITLE_X = 10
TITLE_Y = 28

PANEL_GAP_PX = 5
PANEL_GAP_COLOR = (80, 80, 80)

LEGEND_FONT_SCALE = 0.50
LEGEND_HEIGHT = 38


# ============================================================
# Utility
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Return NaN when a metric is mathematically undefined."""

    if denominator == 0:
        return float("nan")

    return numerator / denominator


def format_value(
    value: float,
    digits: int = 4,
) -> str:
    """Format metric values for text output."""

    if np.isnan(value):
        return "n/a"

    return f"{value:.{digits}f}"


# ============================================================
# Dataset checks
# ============================================================

def validate_test_dataset():
    """
    Verify the semantic test dataset.

    The SAME images and 0/1 semantic masks from dataset_tiled_sem
    are used as the common ground truth for all three models.
    """

    if not TEST_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Test image directory not found: {TEST_IMAGE_DIR}"
        )

    if not TEST_MASK_DIR.exists():
        raise FileNotFoundError(
            f"Test mask directory not found: {TEST_MASK_DIR}"
        )

    image_paths = sorted(
        TEST_IMAGE_DIR.glob("*.png")
    )

    if not image_paths:
        raise ValueError(
            f"No test images found in: {TEST_IMAGE_DIR}"
        )

    missing_masks = []
    invalid_masks = []

    for image_path in image_paths:

        mask_path = (
            TEST_MASK_DIR
            / image_path.name
        )

        if not mask_path.exists():
            missing_masks.append(
                image_path.name
            )
            continue

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            invalid_masks.append(
                (
                    image_path.name,
                    "could not be loaded",
                )
            )
            continue

        unique_values = set(
            np.unique(mask).tolist()
        )

        if not unique_values.issubset(
            {0, 1}
        ):
            invalid_masks.append(
                (
                    image_path.name,
                    sorted(unique_values),
                )
            )

    if missing_masks:
        raise ValueError(
            "Missing masks for test images:\n"
            + "\n".join(
                missing_masks[:20]
            )
        )

    if invalid_masks:
        raise ValueError(
            "Semantic test masks must contain only class IDs "
            "0 = background and 1 = scratch. "
            f"Invalid examples: {invalid_masks[:10]}"
        )

    print(
        f"Test dataset validated: "
        f"{len(image_paths)} image-mask pairs."
    )

    print(
        "Ground-truth mask values: "
        "0 = background, 1 = scratch"
    )


# ============================================================
# Load ground truth
# ============================================================

def load_ground_truth_mask(
    mask_path: Path,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """
    Load a semantic ground-truth mask from dataset_tiled_sem.

    Expected values:
        0 = background
        1 = scratch

    No thresholding is applied because the pixel values are
    already semantic class IDs.
    """

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise FileNotFoundError(
            f"Could not load mask: {mask_path}"
        )

    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask size does not match image:\n"
            f"{mask_path}\n"
            f"Mask:  {mask.shape}\n"
            f"Image: {expected_shape}"
        )

    unique_values = set(
        np.unique(mask).tolist()
    )

    if not unique_values.issubset(
        {0, 1}
    ):
        raise ValueError(
            f"Invalid class IDs in {mask_path}: "
            f"{sorted(unique_values)}. "
            "Expected only 0 and 1."
        )

    return (
        mask == SEM_SCRATCH_CLASS_ID
    ).astype(np.uint8)


# ============================================================
# Detection -> binary pixel mask
# ============================================================

def predict_detection_mask(
    model: YOLO,
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert YOLO detection boxes to a binary scratch mask.

    Important:
    The complete area inside every predicted bounding box is
    treated as scratch. This intentionally evaluates how precise
    a detection model localizes scratches on a pixel basis.
    """

    height, width = image.shape[:2]

    prediction_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    result = model.predict(
        source=image,
        imgsz=DET_IMAGE_SIZE,
        conf=DET_CONF_THRESHOLD,
        verbose=False,
    )[0]

    if result.boxes is None:
        return prediction_mask

    for box in result.boxes:

        class_id = int(
            box.cls[0].cpu().item()
        )

        if class_id != DET_SCRATCH_CLASS_ID:
            continue

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        x1 = int(
            np.floor(
                np.clip(x1, 0, width - 1)
            )
        )

        y1 = int(
            np.floor(
                np.clip(y1, 0, height - 1)
            )
        )

        x2 = int(
            np.ceil(
                np.clip(x2, 0, width)
            )
        )

        y2 = int(
            np.ceil(
                np.clip(y2, 0, height)
            )
        )

        if x2 <= x1 or y2 <= y1:
            continue

        prediction_mask[
            y1:y2,
            x1:x2,
        ] = 1

    return prediction_mask


# ============================================================
# Instance segmentation -> binary pixel mask
# ============================================================

def predict_instance_segmentation_mask(
    model: YOLO,
    image: np.ndarray,
) -> np.ndarray:

    height, width = image.shape[:2]

    prediction_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    result = model.predict(
        source=image,
        imgsz=SEG_IMAGE_SIZE,
        conf=SEG_CONF_THRESHOLD,
        verbose=False,
    )[0]

    if (
        result.masks is None
        or result.boxes is None
        or len(result.boxes) == 0
    ):
        return prediction_mask

    masks = (
        result.masks.data
        .cpu()
        .numpy()
    )

    class_ids = (
        result.boxes.cls
        .cpu()
        .numpy()
        .astype(int)
    )

    confidences = (
        result.boxes.conf
        .cpu()
        .numpy()
    )

    for (
        instance_mask,
        class_id,
        confidence,
    ) in zip(
        masks,
        class_ids,
        confidences,
    ):

        if class_id != SEG_SCRATCH_CLASS_ID:
            continue

        if confidence < SEG_CONF_THRESHOLD:
            continue

        if instance_mask.shape != (
            height,
            width,
        ):

            instance_mask = cv2.resize(
                instance_mask.astype(
                    np.float32
                ),
                (
                    width,
                    height,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        binary_mask = (
            instance_mask >= 0.5
        ).astype(np.uint8)

        prediction_mask = np.maximum(
            prediction_mask,
            binary_mask,
        )

    return prediction_mask


# ============================================================
# Semantic segmentation -> binary pixel mask
# ============================================================

def predict_semantic_segmentation_mask(
    model: YOLO,
    image: np.ndarray,
) -> np.ndarray:

    height, width = image.shape[:2]

    result = model.predict(
        source=image,
        imgsz=SEM_IMAGE_SIZE,
        verbose=False,
    )[0]

    if (
        result.semantic_mask is None
        or result.semantic_mask.data is None
    ):
        return np.zeros(
            (height, width),
            dtype=np.uint8,
        )

    semantic_mask = (
        result.semantic_mask.data
        .cpu()
        .numpy()
    )

    semantic_mask = np.squeeze(
        semantic_mask
    )

    if semantic_mask.ndim != 2:
        raise ValueError(
            "Unexpected semantic mask shape: "
            f"{semantic_mask.shape}"
        )

    if semantic_mask.shape != (
        height,
        width,
    ):

        semantic_mask = cv2.resize(
            semantic_mask.astype(
                np.uint8
            ),
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    return (
        semantic_mask
        == SEM_SCRATCH_CLASS_ID
    ).astype(np.uint8)


# ============================================================
# Pixel metrics
# ============================================================

def calculate_pixel_counts(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> dict:

    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)

    tp = int(
        np.sum(gt & pred)
    )

    fp = int(
        np.sum(~gt & pred)
    )

    fn = int(
        np.sum(gt & ~pred)
    )

    tn = int(
        np.sum(~gt & ~pred)
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def calculate_metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> dict:
    """
    Calculate common pixel-level metrics.

    Most important for this project:

    recall:
        How many labeled scratch pixels were detected?

    extra_vs_gt:
        How many additional false-positive pixels were predicted,
        relative to the number of labeled scratch pixels?

    dice / IoU:
        Overall spatial agreement of prediction and annotation.
    """

    gt_pixels = tp + fn
    predicted_pixels = tp + fp

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    dice = safe_divide(
        2 * tp,
        2 * tp + fp + fn,
    )

    iou = safe_divide(
        tp,
        tp + fp + fn,
    )

    pixel_accuracy = safe_divide(
        tp + tn,
        tp + fp + fn + tn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    detected_gt_pct = (
        100.0
        * safe_divide(
            tp,
            gt_pixels,
        )
    )

    missed_gt_pct = (
        100.0
        * safe_divide(
            fn,
            gt_pixels,
        )
    )

    extra_vs_gt_pct = (
        100.0
        * safe_divide(
            fp,
            gt_pixels,
        )
    )

    predicted_gt_ratio = safe_divide(
        predicted_pixels,
        gt_pixels,
    )

    return {
        "gt_scratch_pixels":
            gt_pixels,

        "predicted_scratch_pixels":
            predicted_pixels,

        "precision":
            precision,

        "recall":
            recall,

        "dice":
            dice,

        "iou":
            iou,

        "pixel_accuracy":
            pixel_accuracy,

        "specificity":
            specificity,

        "detected_gt_pct":
            detected_gt_pct,

        "missed_gt_pct":
            missed_gt_pct,

        "extra_vs_gt_pct":
            extra_vs_gt_pct,

        "predicted_gt_ratio":
            predicted_gt_ratio,
    }


# ============================================================
# Error visualization
# ============================================================

def create_error_overlay(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.75,
) -> np.ndarray:
    """
    Overlay pixel errors on the original image.

    Green = true positive
    Red   = false positive

    Missed scratch pixels are not colored here.
    They remain visible in the separate Ground Truth Mask panel.
    """

    result = image.copy()

    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)

    tp = gt & pred
    fp = ~gt & pred
    overlays = [
        (
            tp,
            np.array(
                [0, 255, 0],
                dtype=np.float32,
            ),
        ),
        (
            fp,
            np.array(
                [0, 0, 255],
                dtype=np.float32,
            ),
        ),
    ]

    for pixels, color in overlays:

        result[pixels] = (
            (1.0 - alpha)
            * result[pixels]
            + alpha
            * color
        ).astype(np.uint8)

    return result


def create_qualitative_comparison(
    image: np.ndarray,
    ground_truth: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Create compact side-by-side visualization:

    Original | Ground Truth | Detection | Instance Seg. | Semantic Seg.

    Green = correctly detected scratch pixels
    Red   = false-positive scratch pixels
    """

    original_panel = image.copy()

    # --------------------------------------------------------
    # Ground-truth mask:
    # semantic class IDs 0/1 -> visualization 0/255
    # --------------------------------------------------------

    gt_visualization = (
        ground_truth.astype(np.uint8) * 255
    )

    gt_panel = cv2.cvtColor(
        gt_visualization,
        cv2.COLOR_GRAY2BGR,
    )

    det_panel = create_error_overlay(
        image,
        ground_truth,
        predictions["Detection"],
    )

    seg_panel = create_error_overlay(
        image,
        ground_truth,
        predictions["Instance Seg."],
    )

    sem_panel = create_error_overlay(
        image,
        ground_truth,
        predictions["Semantic Seg."],
    )

    # --------------------------------------------------------
    # Smaller titles
    # --------------------------------------------------------

    def draw_compact_title(
        panel: np.ndarray,
        text: str,
    ):

        # Black outline for readability
        cv2.putText(
            panel,
            text,
            (TITLE_X, TITLE_Y),
            cv2.FONT_HERSHEY_SIMPLEX,
            TITLE_FONT_SCALE,
            (0, 0, 0),
            TITLE_THICKNESS + 2,
            cv2.LINE_AA,
        )

        # Green title
        cv2.putText(
            panel,
            text,
            (TITLE_X, TITLE_Y),
            cv2.FONT_HERSHEY_SIMPLEX,
            TITLE_FONT_SCALE,
            (0, 255, 0),
            TITLE_THICKNESS,
            cv2.LINE_AA,
        )

    draw_compact_title(
        original_panel,
        "Original",
    )

    draw_compact_title(
        gt_panel,
        "Ground Truth",
    )

    draw_compact_title(
        det_panel,
        "Detection",
    )

    draw_compact_title(
        seg_panel,
        "Instance Seg.",
    )

    draw_compact_title(
        sem_panel,
        "Semantic Seg.",
    )

    # --------------------------------------------------------
    # Narrow separators between panels
    # --------------------------------------------------------

    panel_height = image.shape[0]

    separator = np.full(
        (
            panel_height,
            PANEL_GAP_PX,
            3,
        ),
        PANEL_GAP_COLOR,
        dtype=np.uint8,
    )

    comparison = cv2.hconcat(
        [
            original_panel,
            separator,
            gt_panel,
            separator,
            det_panel,
            separator,
            seg_panel,
            separator,
            sem_panel,
        ]
    )

    # --------------------------------------------------------
    # Compact legend
    # --------------------------------------------------------

    legend = np.zeros(
        (
            LEGEND_HEIGHT,
            comparison.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    legend_text = (
        "GREEN = correctly detected scratch  |  "
        "RED = false positive"
    )

    cv2.putText(
        legend,
        legend_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        LEGEND_FONT_SCALE,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return cv2.vconcat(
        [
            comparison,
            legend,
        ]
    )


# ============================================================
# CSV
# ============================================================

def write_csv(
    path: Path,
    rows: list[dict],
):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Plots
# ============================================================

def add_bar_labels(
    axis,
    bars,
    digits: int = 1,
):
    """Add value labels above bars."""

    for bar in bars:

        height = bar.get_height()

        if np.isnan(height):
            continue

        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            height,
            f"{height:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_quality_metrics(
    summary: dict,
    output_path: Path,
):

    models = list(
        summary.keys()
    )

    metric_keys = [
        "precision",
        "recall",
        "dice",
        "iou",
    ]

    metric_labels = [
        "Precision",
        "Recall",
        "Dice / F1",
        "IoU",
    ]

    x = np.arange(
        len(metric_labels)
    )

    width = 0.24

    fig, axis = plt.subplots(
        figsize=(11, 6)
    )

    for index, model_name in enumerate(models):

        values = [
            summary[model_name][key]
            for key in metric_keys
        ]

        bars = axis.bar(
            x
            + (
                index
                - (len(models) - 1) / 2
            )
            * width,
            values,
            width,
            label=model_name,
        )

        add_bar_labels(
            axis,
            bars,
            digits=2,
        )

    axis.set_ylabel(
        "Score"
    )

    axis.set_ylim(
        0,
        1.08,
    )

    axis.set_title(
        "Pixel-level model quality"
    )

    axis.set_xticks(
        x,
        metric_labels,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


def plot_gt_pixel_coverage(
    summary: dict,
    output_path: Path,
):

    models = list(
        summary.keys()
    )

    detected = [
        summary[model][
            "detected_gt_pct"
        ]
        for model in models
    ]

    missed = [
        summary[model][
            "missed_gt_pct"
        ]
        for model in models
    ]

    x = np.arange(
        len(models)
    )

    fig, axis = plt.subplots(
        figsize=(9, 6)
    )

    bars_detected = axis.bar(
        x,
        detected,
        label="Detected labeled pixels",
    )

    bars_missed = axis.bar(
        x,
        missed,
        bottom=detected,
        label="Missed labeled pixels",
    )

    add_bar_labels(
        axis,
        bars_detected,
        digits=1,
    )

    for index, bar in enumerate(
        bars_missed
    ):

        if np.isnan(missed[index]):
            continue

        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            detected[index]
            + missed[index] / 2,
            f"{missed[index]:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
        )

    axis.set_ylabel(
        "Ground-truth scratch pixels [%]"
    )

    axis.set_ylim(
        0,
        105,
    )

    axis.set_title(
        "How many labeled scratch pixels are found?"
    )

    axis.set_xticks(
        x,
        models,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


def plot_extra_pixels(
    summary: dict,
    output_path: Path,
):

    models = list(
        summary.keys()
    )

    values = [
        summary[model][
            "extra_vs_gt_pct"
        ]
        for model in models
    ]

    fig, axis = plt.subplots(
        figsize=(9, 6)
    )

    bars = axis.bar(
        models,
        values,
    )

    add_bar_labels(
        axis,
        bars,
        digits=1,
    )

    axis.set_ylabel(
        "False-positive pixels / GT scratch pixels [%]"
    )

    axis.set_title(
        "How many additional pixels are predicted as scratch?"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


def plot_prediction_area_ratio(
    summary: dict,
    output_path: Path,
):

    models = list(
        summary.keys()
    )

    values = [
        summary[model][
            "predicted_gt_ratio"
        ]
        for model in models
    ]

    fig, axis = plt.subplots(
        figsize=(9, 6)
    )

    bars = axis.bar(
        models,
        values,
    )

    add_bar_labels(
        axis,
        bars,
        digits=2,
    )

    axis.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label="Ideal ratio = 1",
    )

    axis.set_ylabel(
        "Predicted scratch pixels / labeled scratch pixels"
    )

    axis.set_title(
        "Predicted scratch area compared with ground truth"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


def plot_inference_time(
    summary: dict,
    output_path: Path,
):

    models = list(
        summary.keys()
    )

    values = [
        summary[model][
            "avg_inference_ms"
        ]
        for model in models
    ]

    fig, axis = plt.subplots(
        figsize=(9, 6)
    )

    bars = axis.bar(
        models,
        values,
    )

    add_bar_labels(
        axis,
        bars,
        digits=1,
    )

    axis.set_ylabel(
        "Average inference time per test tile [ms]"
    )

    axis.set_title(
        "Inference speed"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


def plot_confusion_matrix(
    model_name: str,
    counts: dict,
    output_path: Path,
):

    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    tp = counts["tp"]

    background_total = (
        tn + fp
    )

    scratch_total = (
        fn + tp
    )

    matrix = np.array(
        [
            [
                safe_divide(
                    tn,
                    background_total,
                ),
                safe_divide(
                    fp,
                    background_total,
                ),
            ],
            [
                safe_divide(
                    fn,
                    scratch_total,
                ),
                safe_divide(
                    tp,
                    scratch_total,
                ),
            ],
        ],
        dtype=float,
    )

    fig, axis = plt.subplots(
        figsize=(6, 5)
    )

    image = axis.imshow(
        matrix,
        vmin=0,
        vmax=1,
    )

    axis.set_xticks(
        [0, 1],
        [
            "Pred. background",
            "Pred. scratch",
        ],
    )

    axis.set_yticks(
        [0, 1],
        [
            "GT background",
            "GT scratch",
        ],
    )

    axis.set_title(
        f"{model_name} - normalized pixel confusion matrix"
    )

    for row in range(2):
        for column in range(2):

            value = matrix[
                row,
                column,
            ]

            axis.text(
                column,
                row,
                (
                    "n/a"
                    if np.isnan(value)
                    else f"{value * 100:.1f}%"
                ),
                ha="center",
                va="center",
                fontsize=12,
            )

    fig.colorbar(
        image,
        ax=axis,
        label="Fraction",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# Text summary
# ============================================================

def create_text_summary(
    summary: dict,
    output_path: Path,
):

    model_names = list(
        summary.keys()
    )

    best_dice = max(
        model_names,
        key=lambda model:
            summary[model]["dice"],
    )

    best_iou = max(
        model_names,
        key=lambda model:
            summary[model]["iou"],
    )

    best_recall = max(
        model_names,
        key=lambda model:
            summary[model]["recall"],
    )

    best_precision = max(
        model_names,
        key=lambda model:
            summary[model]["precision"],
    )

    lowest_extra = min(
        model_names,
        key=lambda model:
            summary[model][
                "extra_vs_gt_pct"
            ],
    )

    fastest = min(
        model_names,
        key=lambda model:
            summary[model][
                "avg_inference_ms"
            ],
    )

    lines = [
        "YOLO26 Scratch Detection - Test Summary",
        "=" * 44,
        "",
        "All three models are evaluated on the SAME tiled test images",
        "from dataset_tiled_sem/images/test.",
        "Ground truth is taken directly from dataset_tiled_sem/masks/test",
        "with class IDs 0 = background and 1 = scratch.",
        "No threshold conversion of the ground-truth masks is performed.",
        "",
        "Important interpretation:",
        "- Recall = fraction of labeled scratch pixels that were found.",
        "- Precision = fraction of predicted scratch pixels that are correct.",
        "- Extra pixels = false-positive pixels relative to the labeled scratch area.",
        "- Dice and IoU summarize spatial overlap.",
        "- Pixel accuracy is less informative because background dominates.",
        "- Detection boxes are filled completely for this pixel-level comparison.",
        "",
    ]

    for model_name in model_names:

        metrics = summary[
            model_name
        ]

        lines.extend(
            [
                model_name,
                "-" * len(model_name),
                (
                    "Detected labeled pixels: "
                    f"{format_value(metrics['detected_gt_pct'], 1)} %"
                ),
                (
                    "Missed labeled pixels:   "
                    f"{format_value(metrics['missed_gt_pct'], 1)} %"
                ),
                (
                    "Extra predicted pixels:  "
                    f"{format_value(metrics['extra_vs_gt_pct'], 1)} % of GT scratch area"
                ),
                (
                    "Precision:               "
                    f"{format_value(metrics['precision'])}"
                ),
                (
                    "Recall:                  "
                    f"{format_value(metrics['recall'])}"
                ),
                (
                    "Dice / F1:               "
                    f"{format_value(metrics['dice'])}"
                ),
                (
                    "IoU:                     "
                    f"{format_value(metrics['iou'])}"
                ),
                (
                    "Pixel accuracy:          "
                    f"{format_value(metrics['pixel_accuracy'])}"
                ),
                (
                    "Positive-image hit rate: "
                    f"{format_value(metrics['positive_image_hit_rate_pct'], 1)} %"
                ),
                (
                    "Clean-image false alarm: "
                    f"{format_value(metrics['clean_image_false_alarm_rate_pct'], 1)} %"
                ),
                (
                    "Average inference time:  "
                    f"{format_value(metrics['avg_inference_ms'], 1)} ms"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "Best values",
            "-----------",
            f"Best Dice:       {best_dice}",
            f"Best IoU:        {best_iou}",
            f"Best Recall:     {best_recall}",
            f"Best Precision:  {best_precision}",
            f"Least extra px:  {lowest_extra}",
            f"Fastest:         {fastest}",
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# Main evaluation
# ============================================================

def main():

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    model_paths = {
        "Detection":
            DET_MODEL_PATH,

        "Instance Seg.":
            SEG_MODEL_PATH,

        "Semantic Seg.":
            SEM_MODEL_PATH,
    }

    for model_name, model_path in model_paths.items():

        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_name} model not found: "
                f"{model_path}"
            )

    validate_test_dataset()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load models once
    # --------------------------------------------------------

    print("\nLoading models...")

    models = {
        "Detection":
            YOLO(
                str(
                    DET_MODEL_PATH
                )
            ),

        "Instance Seg.":
            YOLO(
                str(
                    SEG_MODEL_PATH
                )
            ),

        "Semantic Seg.":
            YOLO(
                str(
                    SEM_MODEL_PATH
                )
            ),
    }

    prediction_functions = {
        "Detection":
            predict_detection_mask,

        "Instance Seg.":
            predict_instance_segmentation_mask,

        "Semantic Seg.":
            predict_semantic_segmentation_mask,
    }

    # --------------------------------------------------------
    # Test images
    # --------------------------------------------------------

    image_paths = sorted(
        TEST_IMAGE_DIR.glob(
            "*.png"
        )
    )

    if MAX_IMAGES is not None:
        image_paths = image_paths[
            :MAX_IMAGES
        ]

    if not image_paths:
        raise ValueError(
            "No test images available."
        )

    print(
        f"Evaluating {len(image_paths)} "
        f"test images..."
    )

    # --------------------------------------------------------
    # Choose qualitative samples
    # Prefer test tiles containing labeled scratches.
    # --------------------------------------------------------

    positive_image_names = []

    for image_path in image_paths:

        mask_path = (
            TEST_MASK_DIR
            / image_path.name
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise FileNotFoundError(
                f"Could not load mask: {mask_path}"
            )

        # Semantic ground-truth masks already contain:
        # 0 = background
        # 1 = scratch
        if np.any(
            mask == SEM_SCRATCH_CLASS_ID
        ):
            positive_image_names.append(
                image_path.name
            )

    random_generator = random.Random(
        RANDOM_SEED
    )

    all_image_names = [
        image_path.name
        for image_path in image_paths
    ]

    # First choose scratch-containing tiles.
    if len(positive_image_names) >= NUM_QUALITATIVE_EXAMPLES:

        selected_example_names = (
            random_generator.sample(
                positive_image_names,
                NUM_QUALITATIVE_EXAMPLES,
            )
        )

    else:

        selected_example_names = list(
            positive_image_names
        )

        remaining_names = [
            image_name
            for image_name in all_image_names
            if image_name
            not in selected_example_names
        ]

        number_needed = (
            NUM_QUALITATIVE_EXAMPLES
            - len(
                selected_example_names
            )
        )

        if (
            number_needed > 0
            and remaining_names
        ):

            selected_example_names.extend(
                random_generator.sample(
                    remaining_names,
                    min(
                        number_needed,
                        len(remaining_names),
                    ),
                )
            )

    qualitative_names = set(
        selected_example_names
    )

    print(
        f"Selected {len(qualitative_names)} "
        f"qualitative examples."
    )

    print(
        f"Scratch-containing test tiles: "
        f"{len(positive_image_names)}"
    )

    saved_example_count = 0

    # --------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------

    aggregate_counts = {
        model_name: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
        for model_name
        in models
    }

    inference_times = {
        model_name: []
        for model_name
        in models
    }

    image_statistics = {
        model_name: {
            "positive_images": 0,
            "positive_image_hits": 0,
            "clean_images": 0,
            "clean_image_false_alarms": 0,
        }
        for model_name
        in models
    }

    per_image_rows = []

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for image_index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            f"[{image_index:>3}/"
            f"{len(image_paths)}] "
            f"{image_path.name}"
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                f"Could not load image: "
                f"{image_path}"
            )

        ground_truth = load_ground_truth_mask(
            mask_path=(
                TEST_MASK_DIR
                / image_path.name
            ),
            expected_shape=
                image.shape[:2],
        )

        predictions = {}

        gt_positive_pixels = int(
            np.sum(
                ground_truth == 1
            )
        )

        for model_name, model in models.items():

            predict_function = (
                prediction_functions[
                    model_name
                ]
            )

            start_time = (
                time.perf_counter()
            )

            prediction = predict_function(
                model,
                image,
            )

            elapsed_ms = (
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000.0
            )

            inference_times[
                model_name
            ].append(
                elapsed_ms
            )

            predictions[
                model_name
            ] = prediction

            counts = (
                calculate_pixel_counts(
                    ground_truth,
                    prediction,
                )
            )

            for key in [
                "tp",
                "fp",
                "fn",
                "tn",
            ]:

                aggregate_counts[
                    model_name
                ][key] += counts[key]

            metrics = (
                calculate_metrics_from_counts(
                    **counts
                )
            )

            predicted_positive_pixels = int(
                np.sum(
                    prediction == 1
                )
            )

            statistics = image_statistics[
                model_name
            ]

            if (
                gt_positive_pixels
                >= MIN_PIXELS_FOR_POSITIVE_IMAGE
            ):

                statistics[
                    "positive_images"
                ] += 1

                if counts["tp"] > 0:
                    statistics[
                        "positive_image_hits"
                    ] += 1

            else:

                statistics[
                    "clean_images"
                ] += 1

                if (
                    predicted_positive_pixels
                    >= MIN_PIXELS_FOR_POSITIVE_IMAGE
                ):

                    statistics[
                        "clean_image_false_alarms"
                    ] += 1

            per_image_rows.append(
                {
                    "image":
                        image_path.name,

                    "model":
                        model_name,

                    "tp_pixels":
                        counts["tp"],

                    "fp_pixels":
                        counts["fp"],

                    "fn_pixels":
                        counts["fn"],

                    "tn_pixels":
                        counts["tn"],

                    "gt_scratch_pixels":
                        metrics[
                            "gt_scratch_pixels"
                        ],

                    "predicted_scratch_pixels":
                        metrics[
                            "predicted_scratch_pixels"
                        ],

                    "detected_gt_pct":
                        metrics[
                            "detected_gt_pct"
                        ],

                    "missed_gt_pct":
                        metrics[
                            "missed_gt_pct"
                        ],

                    "extra_vs_gt_pct":
                        metrics[
                            "extra_vs_gt_pct"
                        ],

                    "precision":
                        metrics[
                            "precision"
                        ],

                    "recall":
                        metrics[
                            "recall"
                        ],

                    "dice":
                        metrics[
                            "dice"
                        ],

                    "iou":
                        metrics[
                            "iou"
                        ],

                    "pixel_accuracy":
                        metrics[
                            "pixel_accuracy"
                        ],

                    "inference_ms":
                        elapsed_ms,
                }
            )

        # ----------------------------------------------------
        # Save selected qualitative comparisons
        # ----------------------------------------------------

        if image_path.name in qualitative_names:

            qualitative_image = (
                create_qualitative_comparison(
                    image=image,
                    ground_truth=
                        ground_truth,
                    predictions=
                        predictions,
                )
            )

            example_output_path = (
                EXAMPLE_DIR
                / (
                    f"{image_path.stem}"
                    "_comparison.png"
                )
            )

            write_success = cv2.imwrite(
                str(example_output_path),
                qualitative_image,
            )

            if not write_success:
                raise IOError(
                    f"Could not save example image: "
                    f"{example_output_path}"
                )

            saved_example_count += 1

    print(
        f"Saved {saved_example_count} "
        f"qualitative examples to: "
        f"{EXAMPLE_DIR}"
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    summary = {}

    for model_name in models:

        counts = aggregate_counts[
            model_name
        ]

        metrics = (
            calculate_metrics_from_counts(
                **counts
            )
        )

        statistics = image_statistics[
            model_name
        ]

        positive_hit_rate = (
            100.0
            * safe_divide(
                statistics[
                    "positive_image_hits"
                ],
                statistics[
                    "positive_images"
                ],
            )
        )

        clean_false_alarm_rate = (
            100.0
            * safe_divide(
                statistics[
                    "clean_image_false_alarms"
                ],
                statistics[
                    "clean_images"
                ],
            )
        )

        metrics.update(
            {
                "tp_pixels":
                    counts["tp"],

                "fp_pixels":
                    counts["fp"],

                "fn_pixels":
                    counts["fn"],

                "tn_pixels":
                    counts["tn"],

                "images":
                    len(image_paths),

                "positive_images":
                    statistics[
                        "positive_images"
                    ],

                "clean_images":
                    statistics[
                        "clean_images"
                    ],

                "positive_image_hit_rate_pct":
                    positive_hit_rate,

                "clean_image_false_alarm_rate_pct":
                    clean_false_alarm_rate,

                "avg_inference_ms":
                    float(
                        np.mean(
                            inference_times[
                                model_name
                            ]
                        )
                    ),
            }
        )

        summary[
            model_name
        ] = metrics

    # --------------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------------

    summary_rows = []

    for model_name, metrics in summary.items():

        summary_rows.append(
            {
                "model":
                    model_name,

                **metrics,
            }
        )

    write_csv(
        OUTPUT_DIR
        / "model_metrics.csv",
        summary_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "per_image_metrics.csv",
        per_image_rows,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_quality_metrics(
        summary,
        PLOT_DIR
        / "01_quality_metrics.png",
    )

    plot_gt_pixel_coverage(
        summary,
        PLOT_DIR
        / "02_labeled_pixel_coverage.png",
    )

    plot_extra_pixels(
        summary,
        PLOT_DIR
        / "03_extra_predicted_pixels.png",
    )

    plot_prediction_area_ratio(
        summary,
        PLOT_DIR
        / "04_prediction_area_ratio.png",
    )

    plot_inference_time(
        summary,
        PLOT_DIR
        / "05_inference_time.png",
    )

    for model_name in models:

        safe_name = (
            model_name
            .lower()
            .replace(
                " ",
                "_",
            )
            .replace(
                ".",
                "",
            )
        )

        plot_confusion_matrix(
            model_name=
                model_name,

            counts=
                aggregate_counts[
                    model_name
                ],

            output_path=(
                PLOT_DIR
                / (
                    "confusion_matrix_"
                    f"{safe_name}.png"
                )
            ),
        )

    # --------------------------------------------------------
    # Text report
    # --------------------------------------------------------

    create_text_summary(
        summary=summary,
        output_path=(
            OUTPUT_DIR
            / "summary.txt"
        ),
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("FINAL PIXEL-LEVEL RESULTS")
    print("=" * 75)

    for model_name, metrics in summary.items():

        print()
        print(model_name)
        print("-" * len(model_name))

        print(
            "Labeled scratch pixels found: "
            f"{format_value(metrics['detected_gt_pct'], 1)} %"
        )

        print(
            "Labeled scratch pixels missed: "
            f"{format_value(metrics['missed_gt_pct'], 1)} %"
        )

        print(
            "Additional false-positive pixels: "
            f"{format_value(metrics['extra_vs_gt_pct'], 1)} % "
            "of GT scratch area"
        )

        print(
            "Precision: "
            f"{format_value(metrics['precision'])}"
        )

        print(
            "Recall:    "
            f"{format_value(metrics['recall'])}"
        )

        print(
            "Dice/F1:   "
            f"{format_value(metrics['dice'])}"
        )

        print(
            "IoU:       "
            f"{format_value(metrics['iou'])}"
        )

        print(
            "Positive image hit rate: "
            f"{format_value(metrics['positive_image_hit_rate_pct'], 1)} %"
        )

        print(
            "Clean image false-alarm rate: "
            f"{format_value(metrics['clean_image_false_alarm_rate_pct'], 1)} %"
        )

    print()
    print(
        f"Results saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()

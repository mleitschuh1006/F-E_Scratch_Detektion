"""Shared visualization and result-saving helpers."""

from pathlib import Path

import cv2
import numpy as np


def draw_title(
    image: np.ndarray,
    text: str,
) -> None:
    """Draw a scalable green title with a black outline in-place."""

    font_scale = max(1.5, image.shape[0] / 180)
    thickness = max(2, int(font_scale * 2))
    text_x = 25
    text_y = int(30 + 20 * font_scale)

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


def draw_mask_overlay(
    image: np.ndarray,
    scratch_mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay binary scratch pixels in green."""

    result_image = image.copy()
    scratch_pixels = scratch_mask.astype(bool)

    result_image[scratch_pixels] = (
        (1.0 - alpha) * result_image[scratch_pixels]
        + alpha * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    return result_image


def create_comparison_image(
    original_image: np.ndarray,
    mask_visualization: np.ndarray,
    overlay_image: np.ndarray,
    titles: tuple[str, str, str] = (
        "Original",
        "Predicted Mask",
        "Scratch Overlay",
    ),
) -> np.ndarray:
    """Create Original | Mask | Overlay comparison image."""

    if mask_visualization.ndim == 2:
        mask_panel = cv2.cvtColor(mask_visualization, cv2.COLOR_GRAY2BGR)
    else:
        mask_panel = mask_visualization.copy()

    original_panel = original_image.copy()
    overlay_panel = overlay_image.copy()

    draw_title(original_panel, titles[0])
    draw_title(mask_panel, titles[1])
    draw_title(overlay_panel, titles[2])

    return cv2.hconcat([original_panel, mask_panel, overlay_panel])


def save_mask_prediction_outputs(
    output_dir: Path,
    image_stem: str,
    original_image: np.ndarray,
    scratch_mask: np.ndarray,
    overlay_image: np.ndarray,
    comparison_image: np.ndarray,
) -> dict[str, Path]:
    """Save original, binary mask, overlay and comparison images."""

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_visualization = (scratch_mask.astype(np.uint8) * 255)

    paths = {
        "original": output_dir / f"{image_stem}_original.png",
        "mask": output_dir / f"{image_stem}_mask.png",
        "overlay": output_dir / f"{image_stem}_overlay.png",
        "comparison": output_dir / f"{image_stem}_comparison.png",
    }

    cv2.imwrite(str(paths["original"]), original_image)
    cv2.imwrite(str(paths["mask"]), mask_visualization)
    cv2.imwrite(str(paths["overlay"]), overlay_image)
    cv2.imwrite(str(paths["comparison"]), comparison_image)

    return paths

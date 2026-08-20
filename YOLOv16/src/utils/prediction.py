"""Shared post-processing helpers for tiled predictions."""

import numpy as np


def calculate_iou(box_a, box_b) -> float:
    """Calculate intersection over union of two xyxy boxes."""

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection = intersection_width * intersection_height

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def apply_global_nms(
    detections: list[dict],
    iou_threshold: float,
) -> list[dict]:
    """Apply class-aware NMS after merging detections from overlapping tiles."""

    if not detections:
        return []

    pending = sorted(
        detections,
        key=lambda detection: detection["confidence"],
        reverse=True,
    )

    keep: list[dict] = []

    while pending:
        best = pending.pop(0)
        keep.append(best)
        remaining = []

        for detection in pending:
            if detection["class_id"] != best["class_id"]:
                remaining.append(detection)
                continue

            if calculate_iou(best["box"], detection["box"]) < iou_threshold:
                remaining.append(detection)

        pending = remaining

    return keep


def create_mask_vote_maps(
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create scratch-vote and coverage maps for tiled mask predictions."""

    scratch_votes = np.zeros(
        (image_height, image_width),
        dtype=np.uint16,
    )
    coverage = np.zeros(
        (image_height, image_width),
        dtype=np.uint16,
    )

    return scratch_votes, coverage


def add_tile_mask_vote(
    scratch_votes: np.ndarray,
    coverage: np.ndarray,
    tile_mask: np.ndarray,
    x1: int,
    y1: int,
) -> None:
    """Add one binary tile mask to full-image vote maps."""

    tile_height, tile_width = tile_mask.shape[:2]
    x2 = x1 + tile_width
    y2 = y1 + tile_height

    scratch_votes[y1:y2, x1:x2] += tile_mask.astype(np.uint16)
    coverage[y1:y2, x1:x2] += 1


def finalize_mask_votes(
    scratch_votes: np.ndarray,
    coverage: np.ndarray,
    vote_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert accumulated tile votes into a binary full-image mask."""

    scratch_ratio = np.zeros(scratch_votes.shape, dtype=np.float32)
    valid_pixels = coverage > 0

    scratch_ratio[valid_pixels] = (
        scratch_votes[valid_pixels] / coverage[valid_pixels]
    )

    final_mask = (scratch_ratio >= vote_threshold).astype(np.uint8)
    return final_mask, scratch_ratio

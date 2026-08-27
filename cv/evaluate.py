from __future__ import annotations

from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
EVAL = CFG["evaluation"]


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    return mask > 0


def strict_counts(gt: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.logical_and(gt, pred).sum()),
        "fp": int(np.logical_and(~gt, pred).sum()),
        "fn": int(np.logical_and(gt, ~pred).sum()),
        "tn": int(np.logical_and(~gt, ~pred).sum()),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "iou": safe_div(tp, tp + fp + fn),
        "specificity": safe_div(tn, tn + fp),
    }


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def tolerant_metrics(gt: np.ndarray, pred: np.ndarray, radius: int) -> dict[str, float | int]:
    if radius <= 0:
        return metrics_from_counts(strict_counts(gt, pred))

    gt_dilated = dilate(gt, radius)
    pred_dilated = dilate(pred, radius)

    matched_pred_pixels = int(np.logical_and(pred, gt_dilated).sum())
    matched_gt_pixels = int(np.logical_and(gt, pred_dilated).sum())
    pred_pixels = int(pred.sum())
    gt_pixels = int(gt.sum())

    precision = safe_div(matched_pred_pixels, pred_pixels)
    recall = safe_div(matched_gt_pixels, gt_pixels)

    return {
        "tolerance_px": radius,
        "matched_prediction_pixels": matched_pred_pixels,
        "prediction_pixels": pred_pixels,
        "matched_gt_pixels": matched_gt_pixels,
        "gt_pixels": gt_pixels,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


def pca_length_width(component: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(component)
    area = float(len(xs))
    if len(xs) < 2:
        return area, area

    pts = np.column_stack((xs, ys)).astype(np.float64)
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        length = float(max(np.ptp(xs) + 1, np.ptp(ys) + 1))
    else:
        values, vectors = np.linalg.eigh(cov)
        major = vectors[:, int(np.argmax(values))]
        projection = centered @ major
        length = float(projection.max() - projection.min() + 1.0)

    length = max(length, 1.0)
    width = area / length
    return length, width


def gt_scratch_rows(filename: str, gt: np.ndarray, pred: np.ndarray) -> list[dict]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
    rows: list[dict] = []
    thresholds = [float(v) for v in EVAL["scratch_overlap_thresholds"]]

    for label in range(1, n):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        overlap_pixels = int(np.logical_and(component, pred).sum())
        overlap_fraction = safe_div(overlap_pixels, area)
        length, width = pca_length_width(component)

        row = {
            "filename": filename,
            "scratch_id": label,
            "area_px": area,
            "estimated_length_px": length,
            "estimated_width_px": width,
            "overlap_pixels": overlap_pixels,
            "overlap_fraction": overlap_fraction,
        }
        for threshold in thresholds:
            key = f"detected_{int(round(threshold * 100)):02d}pct"
            row[key] = int(overlap_fraction >= threshold)
        rows.append(row)

    return rows


def prediction_component_stats(gt: np.ndarray, pred: np.ndarray, threshold: float) -> tuple[int, int]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), 8)
    total = max(0, n - 1)
    hits = 0

    for label in range(1, n):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        overlap = int(np.logical_and(component, gt).sum())
        if safe_div(overlap, area) >= threshold:
            hits += 1

    return hits, total


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) / total) + z2 / (4.0 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def make_size_bins(
    scratch_rows: list[dict],
    dimension: str,
    edges: list[float],
    detected_key: str,
) -> list[dict]:
    result: list[dict] = []
    sorted_edges = [float(v) for v in edges]

    for i, low in enumerate(sorted_edges):
        high = sorted_edges[i + 1] if i + 1 < len(sorted_edges) else math.inf
        selected = [
            row for row in scratch_rows
            if float(row[dimension]) >= low and float(row[dimension]) < high
        ]
        total = len(selected)
        detected = sum(int(row[detected_key]) for row in selected)
        rate = safe_div(detected, total)
        ci_low, ci_high = wilson_interval(detected, total)

        result.append(
            {
                "dimension": dimension,
                "bin_low": low,
                "bin_high": "inf" if math.isinf(high) else high,
                "n_scratches": total,
                "n_detected": detected,
                "detection_rate": rate,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )

    return result


def reliable_threshold(
    scratch_rows: list[dict],
    dimension: str,
    detected_key: str,
    target: float,
    min_samples: int,
) -> dict | None:
    values = sorted({float(row[dimension]) for row in scratch_rows})
    for threshold in values:
        selected = [row for row in scratch_rows if float(row[dimension]) >= threshold]
        if len(selected) < min_samples:
            continue
        detected = sum(int(row[detected_key]) for row in selected)
        rate = safe_div(detected, len(selected))
        if rate >= target:
            ci_low, ci_high = wilson_interval(detected, len(selected))
            return {
                "threshold": threshold,
                "n_scratches_at_or_above": len(selected),
                "n_detected": detected,
                "detection_rate": rate,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    mask_dir = ROOT / CFG["mask_dir"]
    pred_dir = ROOT / CFG["output_dir"] / "prediction_masks"
    out_dir = ROOT / CFG["output_dir"] / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_paths = sorted(mask_dir.glob("*.png"))
    if not gt_paths:
        raise RuntimeError(f"No ground-truth masks found in {mask_dir}")

    missing = [path.name for path in gt_paths if not (pred_dir / path.name).exists()]
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            f"Missing {len(missing)} prediction masks ({preview}). "
            "Run 'uv run python run_detection.py' first."
        )

    strict_total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    tolerance_radii = [int(v) for v in EVAL["pixel_tolerances_px"]]
    tolerance_acc = {
        radius: {
            "matched_prediction_pixels": 0,
            "prediction_pixels": 0,
            "matched_gt_pixels": 0,
            "gt_pixels": 0,
        }
        for radius in tolerance_radii
        if radius > 0
    }

    scratch_rows: list[dict] = []
    per_image_rows: list[dict] = []
    pred_component_hits_total = 0
    pred_component_total = 0
    pred_component_threshold = float(EVAL["prediction_component_overlap"])
    primary_overlap = float(EVAL["primary_scratch_overlap"])
    primary_key = f"detected_{int(round(primary_overlap * 100)):02d}pct"

    for index, gt_path in enumerate(gt_paths, 1):
        gt = read_mask(gt_path)
        pred = read_mask(pred_dir / gt_path.name)
        if gt.shape != pred.shape:
            raise RuntimeError(f"Shape mismatch for {gt_path.name}: GT={gt.shape}, prediction={pred.shape}")

        counts = strict_counts(gt, pred)
        for key in strict_total:
            strict_total[key] += counts[key]
        strict = metrics_from_counts(counts)

        for radius in tolerance_acc:
            tm = tolerant_metrics(gt, pred, radius)
            for key in tolerance_acc[radius]:
                tolerance_acc[radius][key] += int(tm[key])

        image_scratches = gt_scratch_rows(gt_path.name, gt, pred)
        scratch_rows.extend(image_scratches)

        pred_hits, pred_total = prediction_component_stats(gt, pred, pred_component_threshold)
        pred_component_hits_total += pred_hits
        pred_component_total += pred_total

        scratch_total = len(image_scratches)
        scratch_detected = sum(int(row[primary_key]) for row in image_scratches)

        per_image_rows.append(
            {
                "filename": gt_path.name,
                "pixel_precision": strict["precision"],
                "pixel_recall": strict["recall"],
                "pixel_f1": strict["f1"],
                "pixel_iou": strict["iou"],
                "gt_scratch_components": scratch_total,
                "detected_gt_scratches": scratch_detected,
                "scratch_detection_rate": safe_div(scratch_detected, scratch_total),
                "prediction_components": pred_total,
                "prediction_components_with_gt_overlap": pred_hits,
                "prediction_component_precision": safe_div(pred_hits, pred_total),
            }
        )

        print(f"[{index:02d}/{len(gt_paths):02d}] {gt_path.name}")

    strict_summary = metrics_from_counts(strict_total)

    tolerance_summary: dict[str, dict] = {"0": strict_summary}
    for radius, acc in tolerance_acc.items():
        precision = safe_div(acc["matched_prediction_pixels"], acc["prediction_pixels"])
        recall = safe_div(acc["matched_gt_pixels"], acc["gt_pixels"])
        tolerance_summary[str(radius)] = {
            "tolerance_px": radius,
            **acc,
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
        }

    scratch_detection_summary: dict[str, dict] = {}
    for threshold in [float(v) for v in EVAL["scratch_overlap_thresholds"]]:
        key = f"detected_{int(round(threshold * 100)):02d}pct"
        total = len(scratch_rows)
        detected = sum(int(row[key]) for row in scratch_rows)
        ci_low, ci_high = wilson_interval(detected, total)
        scratch_detection_summary[f"{threshold:.2f}"] = {
            "overlap_threshold": threshold,
            "n_scratches": total,
            "n_detected": detected,
            "detection_rate": safe_div(detected, total),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        }

    size_rows: list[dict] = []
    size_rows += make_size_bins(scratch_rows, "area_px", EVAL["area_bins_px"], primary_key)
    size_rows += make_size_bins(scratch_rows, "estimated_length_px", EVAL["length_bins_px"], primary_key)
    size_rows += make_size_bins(scratch_rows, "estimated_width_px", EVAL["width_bins_px"], primary_key)

    reliable_target = float(EVAL["reliable_recall_target"])
    reliable_min_samples = int(EVAL["reliable_min_samples"])
    reliable = {
        "area_px": reliable_threshold(
            scratch_rows, "area_px", primary_key, reliable_target, reliable_min_samples
        ),
        "estimated_length_px": reliable_threshold(
            scratch_rows, "estimated_length_px", primary_key, reliable_target, reliable_min_samples
        ),
        "estimated_width_px": reliable_threshold(
            scratch_rows, "estimated_width_px", primary_key, reliable_target, reliable_min_samples
        ),
    }

    pred_precision = safe_div(pred_component_hits_total, pred_component_total)
    pred_ci_low, pred_ci_high = wilson_interval(pred_component_hits_total, pred_component_total)

    summary = {
        "method": "cv_opt classical-CV scratch detector",
        "evaluation_scope": (
            "Descriptive evaluation on the supplied dataset. If this dataset was used to tune "
            "parameters manually or automatically, these values are not an independent test-set estimate."
        ),
        "n_image_mask_pairs": len(gt_paths),
        "definitions": {
            "strict_pixel_metrics": "Prediction and ground truth must overlap at the same pixel.",
            "tolerant_pixel_metrics": (
                "Precision: predicted pixel is correct when a GT pixel lies within the tolerance radius. "
                "Recall: GT pixel is found when a prediction lies within the tolerance radius."
            ),
            "scratch_detection": (
                "Each connected component in the GT mask is treated as one scratch. It counts as detected "
                "when at least the configured fraction of its pixels overlaps the prediction."
            ),
            "prediction_component_precision": (
                "A connected prediction counts as GT-supported when at least the configured fraction of "
                "its own pixels overlaps any GT scratch pixel. This is not one-to-one object matching."
            ),
            "estimated_length_px": "PCA major-axis span of the GT scratch component.",
            "estimated_width_px": "GT scratch area divided by its estimated PCA length.",
            "confidence_interval": "95% Wilson interval for binomial detection rates.",
        },
        "pixel_metrics_strict": strict_summary,
        "pixel_metrics_by_tolerance_px": tolerance_summary,
        "scratch_detection_by_gt_overlap": scratch_detection_summary,
        "prediction_component_precision": {
            "overlap_threshold": pred_component_threshold,
            "n_prediction_components": pred_component_total,
            "n_gt_supported_components": pred_component_hits_total,
            "precision": pred_precision,
            "ci95_low": pred_ci_low,
            "ci95_high": pred_ci_high,
        },
        "primary_scratch_overlap": primary_overlap,
        "reliable_size_thresholds": {
            "target_detection_rate": reliable_target,
            "minimum_samples": reliable_min_samples,
            "meaning": (
                "Smallest observed size for which scratches at or above that size reach the target "
                "empirical detection rate, subject to the minimum sample count."
            ),
            **reliable,
        },
    }

    write_csv(out_dir / "per_image.csv", per_image_rows)
    write_csv(out_dir / "scratch_instances.csv", scratch_rows)
    write_csv(out_dir / "size_bins.csv", size_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nEvaluation finished -> {out_dir}")
    print("Generated: summary.json, per_image.csv, scratch_instances.csv, size_bins.csv")


if __name__ == "__main__":
    main()

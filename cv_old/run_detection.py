from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import shutil

import cv2
import numpy as np

from cv_pipeline import segment_scratches


ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def overlay_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    circles: list[tuple[int, int, int]],
    edge_rejected_mask: np.ndarray,
    params: dict,
    alpha: float,
) -> np.ndarray:
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    result = base.copy()
    pred = prediction.astype(bool)

    if np.any(pred):
        red = np.zeros_like(base)
        red[:, :, 2] = 255
        blended = cv2.addWeighted(base, 1.0 - alpha, red, alpha, 0)
        result[pred] = blended[pred]

    if CFG.get("show_edge_rejections_in_overlay", True):
        result[edge_rejected_mask.astype(bool)] = (0, 255, 255)

    if CFG.get("show_hole_circles_in_overlay", True):
        margin = int(params.get("hole_detection", {}).get("exclusion_margin_px", 0))
        for x, y, radius in circles:
            cv2.circle(result, (x, y), radius + margin, (255, 0, 0), 2)

    return result


def save_debug_stages(output_dir: Path, image: np.ndarray, debug: dict, params: dict) -> None:
    debug_dir = output_dir / "debug"
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True)

    def save_bool(name: str, value: np.ndarray) -> None:
        cv2.imwrite(str(debug_dir / name), value.astype(np.uint8) * 255)

    cv2.imwrite(str(debug_dir / "01_gray.png"), debug["gray"])
    save_bool("02_roi.png", debug["roi"])
    save_bool("03_hole_mask.png", debug["hole_mask"])
    save_bool("04_scratch_roi.png", debug["scratch_roi"])
    cv2.imwrite(str(debug_dir / "05_contrast_stretched.png"), debug["normalized"])
    cv2.imwrite(str(debug_dir / "06_local_residual.png"), debug["response"])
    save_bool("07_threshold.png", debug["thresholded"])
    save_bool("08_morphology.png", debug["after_morphology"])
    save_bool("09_component_filter.png", debug["after_component_filter"])
    save_bool("10_edge_parallel_rejected.png", debug["edge_rejected_mask"])
    save_bool("11_final_mask.png", debug["final"])
    cv2.imwrite(
        str(debug_dir / "12_overlay.png"),
        overlay_prediction(
            image,
            debug["final"],
            debug["circles"],
            debug["edge_rejected_mask"],
            params,
            float(CFG["overlay_alpha"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen classical-CV scratch detector."
    )
    parser.add_argument(
        "--image",
        help="Optional single image filename, e.g. 13_max_flat.png. Without this option all images are processed.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="When used with --image, also save the important intermediate stages.",
    )
    args = parser.parse_args()

    input_dir = ROOT / CFG["input_dir"]
    output_dir = ROOT / CFG["output_dir"]
    overlay_dir = output_dir / "overlays"
    mask_dir = output_dir / "prediction_masks"

    params = CFG["pipeline"]

    if args.image:
        paths = [input_dir / args.image]
        if not paths[0].exists():
            raise FileNotFoundError(paths[0])
    else:
        paths = sorted(input_dir.glob("*.png"))
        if not paths:
            raise RuntimeError(f"No PNG images found in {input_dir}")
        # Full run = clean old detection results first so no stale masks survive.
        for folder in (overlay_dir, mask_dir):
            if folder.exists():
                shutil.rmtree(folder)

    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    print(f"Images: {len(paths)}")

    for i, path in enumerate(paths, 1):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")

        prediction, debug = segment_scratches(image, params, return_debug=True)

        cv2.imwrite(str(mask_dir / path.name), prediction.astype(np.uint8) * 255)
        cv2.imwrite(
            str(overlay_dir / f"{path.stem}_detected.png"),
            overlay_prediction(
                image,
                prediction,
                debug["circles"],
                debug["edge_rejected_mask"],
                params,
                float(CFG["overlay_alpha"]),
            ),
        )

        n, _, stats, _ = cv2.connectedComponentsWithStats(
            prediction.astype(np.uint8), 8
        )
        areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([], dtype=int)

        rows.append(
            {
                "filename": path.name,
                "detected_holes": len(debug["circles"]),
                "edge_parallel_rejected_components": len(debug["edge_rejections"]),
                "edge_parallel_rejected_pixels": int(debug["edge_rejected_mask"].sum()),
                "prediction_components": max(0, n - 1),
                "prediction_pixels": int(prediction.sum()),
                "median_component_area_px": float(np.median(areas)) if len(areas) else 0.0,
            }
        )

        if args.image and args.debug:
            save_debug_stages(output_dir, image, debug, params)

        print(
            f"[{i:02d}/{len(paths):02d}] {path.name:<22} "
            f"holes={len(debug['circles']):3d} | "
            f"edge_rejected={len(debug['edge_rejections']):3d} | "
            f"components={max(0, n - 1):4d}"
        )

    summary_path = output_dir / "detection_summary.csv"
    if not args.image:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        (output_dir / "used_config.json").write_text(
            json.dumps(CFG, indent=2), encoding="utf-8"
        )

    print(f"\nDone -> {output_dir}")
    print("Overlay: red=scratch | blue=excluded hole | yellow=rejected outer-edge component")
    if args.image and args.debug:
        print(f"Debug stages -> {output_dir / 'debug'}")


if __name__ == "__main__":
    main()

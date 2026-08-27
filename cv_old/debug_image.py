from pathlib import Path
import argparse
import json
import cv2
import numpy as np

from cv_pipeline import segment_scratches

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def save_bool(path, value):
    cv2.imwrite(str(path), value.astype(np.uint8) * 255)


def overlay(image, prediction, circles, edge_rejected_mask, params):
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    out = base.copy()
    mask = prediction.astype(bool)
    out[mask] = (0, 0, 255)

    # Yellow = component removed by edge + parallelity rule.
    out[edge_rejected_mask.astype(bool)] = (0, 255, 255)

    margin = int(params.get("hole_detection", {}).get("exclusion_margin_px", 0))
    for x, y, radius in circles:
        cv2.circle(out, (x, y), radius + margin, (255, 0, 0), 2)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Show the important stages of the CV pipeline for ONE image."
    )
    parser.add_argument("filename", help="Example: 13_max_flat.png")
    args = parser.parse_args()

    image_path = ROOT / CFG["input_dir"] / args.filename
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    final, d = segment_scratches(image, CFG["pipeline"], return_debug=True)

    out = ROOT / CFG["debug_dir"] / image_path.stem
    if out.exists():
        for p in out.iterdir():
            if p.is_file():
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out / "01_gray.png"), d["gray"])
    save_bool(out / "02_roi_before_holes.png", d["roi"])
    save_bool(out / "03_detected_holes.png", d["hole_mask"])
    save_bool(out / "04_scratch_roi.png", d["scratch_roi"])
    cv2.imwrite(str(out / "05_contrast_stretched.png"), d["normalized"])
    cv2.imwrite(str(out / "06_local_residual.png"), d["response"])
    save_bool(out / "07_after_threshold.png", d["thresholded"])
    save_bool(out / "08_after_morphology.png", d["after_morphology"])
    save_bool(out / "09_after_component_filter.png", d["after_component_filter"])
    save_bool(out / "10_edge_parallel_rejected.png", d["edge_rejected_mask"])
    save_bool(out / "11_final_mask.png", d["final"])
    cv2.imwrite(
        str(out / "12_overlay.png"),
        overlay(
            image,
            d["final"],
            d["circles"],
            d["edge_rejected_mask"],
            CFG["pipeline"],
        ),
    )

    print(f"Detected holes: {len(d['circles'])}")
    print(f"Rejected edge-parallel components: {len(d['edge_rejections'])}")
    print(f"Debug stages saved to: {out}")
    print("Red = detected scratch | Blue = excluded hole | Yellow = rejected edge-parallel detection")


if __name__ == "__main__":
    main()

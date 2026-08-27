from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "results" / "evaluation" / "summary.json"
NEW = ROOT / "results_opt" / "evaluation" / "summary.json"


def pct(v: float) -> str:
    return f"{100.0 * float(v):.2f}%"


def main() -> None:
    if not OLD.exists():
        raise FileNotFoundError(f"Old evaluation not found: {OLD}")
    if not NEW.exists():
        raise FileNotFoundError(
            f"New evaluation not found: {NEW}\n"
            "Run 'uv run python run_detection.py' and then 'uv run python evaluate.py' first."
        )

    old = json.loads(OLD.read_text(encoding="utf-8"))
    new = json.loads(NEW.read_text(encoding="utf-8"))

    rows = [
        ("Pixel Precision", old["pixel_metrics_strict"]["precision"], new["pixel_metrics_strict"]["precision"]),
        ("Pixel Recall", old["pixel_metrics_strict"]["recall"], new["pixel_metrics_strict"]["recall"]),
        ("Pixel F1", old["pixel_metrics_strict"]["f1"], new["pixel_metrics_strict"]["f1"]),
        ("Pixel IoU", old["pixel_metrics_strict"]["iou"], new["pixel_metrics_strict"]["iou"]),
        ("Scratch detection @15%", old["scratch_detection_by_gt_overlap"]["0.15"]["detection_rate"], new["scratch_detection_by_gt_overlap"]["0.15"]["detection_rate"]),
        ("Prediction component precision", old["prediction_component_precision"]["precision"], new["prediction_component_precision"]["precision"]),
    ]

    print("\nOld vs. cv_opt\n")
    print(f"{'Metric':34s} {'OLD':>10s} {'CV_OPT':>10s} {'DELTA':>10s}")
    print("-" * 68)
    for name, a, b in rows:
        print(f"{name:34s} {pct(a):>10s} {pct(b):>10s} {100*(b-a):+9.2f} pp")

    old_103 = ROOT / "results" / "evaluation" / "per_image.csv"
    new_103 = ROOT / "results_opt" / "evaluation" / "per_image.csv"
    if old_103.exists() and new_103.exists():
        import csv
        def row103(path: Path):
            with path.open(newline="", encoding="utf-8") as f:
                return next((r for r in csv.DictReader(f) if r["filename"] == "103_max_flat.png"), None)
        a, b = row103(old_103), row103(new_103)
        if a and b:
            print("\n103_max_flat.png (GT contains no scratch pixels)")
            print(f"Prediction components: {a['prediction_components']} -> {b['prediction_components']}")


if __name__ == "__main__":
    main()

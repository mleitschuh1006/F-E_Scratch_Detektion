"""Regenerate all available PNG masks from saved annotation JSON files.

Run from the repository root:

    uv run python annotation/export_all_masks.py

Only annotations that are actually present in a series JSON file are exported:
- the master image is always exported when its JSON exists;
- a slave is exported only when that slave has a saved state in the JSON.

Original images are never modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from annotation.scratch_core import (  # noqa: E402
    discover_series,
    load_config,
    load_project,
    render_mask,
    write_mask,
)


def resolve_config_dir(annotation_dir: Path, config: dict, key: str) -> Path:
    value = Path(str(config[key]))
    return value if value.is_absolute() else annotation_dir / value


def main() -> int:
    annotation_dir = Path(__file__).resolve().parent
    config = load_config(annotation_dir / "config.yaml")
    images_dir = resolve_config_dir(annotation_dir, config, "images_dir")
    annotations_dir = resolve_config_dir(annotation_dir, config, "annotations_dir")
    masks_dir = resolve_config_dir(annotation_dir, config, "masks_dir")

    series_map = discover_series(images_dir)
    exported = 0
    skipped_series = 0

    for series_id, series in series_map.items():
        project_path = annotations_dir / f"{series_id}.json"
        if not project_path.exists():
            skipped_series += 1
            print(f"Übersprungen: Bildreihe {series_id} besitzt keine JSON-Annotation.")
            continue

        project = load_project(project_path, series, config)
        filenames = [project["master_file"]]
        saved_slaves = set(project.get("slaves", {}).keys())
        filenames.extend(path.name for path in series.slaves if path.name in saved_slaves)

        for filename in filenames:
            mask = render_mask(project, filename)
            target = masks_dir / f"{Path(filename).stem}.png"
            write_mask(target, mask, (series.width, series.height))
            exported += 1
            print(f"Exportiert: {filename} -> {target.name}")

    print()
    print(f"Fertig. {exported} Masken wurden erzeugt.")
    if skipped_series:
        print(f"{skipped_series} Bildreihen ohne JSON-Annotation wurden übersprungen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

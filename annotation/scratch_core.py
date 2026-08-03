"""Core data model and mask generation for the scratch annotation tool.

The module deliberately has no GUI dependency. It stores all coordinates in the
original image coordinate system and always creates masks at the original image
resolution.
"""

from __future__ import annotations

import copy
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
PROJECT_VERSION = 1


@dataclass(frozen=True)
class SeriesFiles:
    """Files belonging to one image series."""

    series_id: str
    master: Path
    slaves: tuple[Path, ...]
    width: int
    height: int

    @property
    def all_files(self) -> tuple[Path, ...]:
        return (self.master, *self.slaves)


DEFAULT_CONFIG: dict[str, Any] = {
    "images_dir": "images",
    "annotations_dir": "annotations",
    "masks_dir": "masks",
    "max_annotation_zoom": 5.0,
    "min_scratch_length_px": 35.0,
    "default_width_px": 5,
    "min_width_px": 1,
    "max_width_px": 60,
    "overlay_opacity": 0.45,
    "overlay_rgb": [255, 40, 40],
}


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration and fill missing keys with defaults."""

    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must contain a mapping: {config_path}")
        config.update(loaded)

    if float(config["max_annotation_zoom"]) <= 0:
        raise ValueError("max_annotation_zoom must be greater than zero")
    if float(config["min_scratch_length_px"]) < 0:
        raise ValueError("min_scratch_length_px must not be negative")
    if int(config["min_width_px"]) < 1:
        raise ValueError("min_width_px must be at least one")
    if int(config["max_width_px"]) < int(config["min_width_px"]):
        raise ValueError("max_width_px must be >= min_width_px")
    return config


def _read_image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Image could not be read: {path}")
    height, width = image.shape[:2]
    return width, height


def discover_series(images_dir: Path) -> dict[str, SeriesFiles]:
    """Discover image series using ``<series>_max_flat.<ext>`` as master naming rule.

    A series may contain any number of slaves. Every image in the series must
    have exactly the same dimensions as its master.
    """

    images_dir = images_dir.resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"Image folder does not exist: {images_dir}")

    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )

    masters: dict[str, Path] = {}
    for path in images:
        if path.stem.lower().endswith("_max_flat"):
            series_id = path.stem[:-9]
            if series_id in masters:
                raise ValueError(f"Multiple _max_flat master images found for series '{series_id}'")
            masters[series_id] = path

    grouped: dict[str, list[Path]] = {series_id: [] for series_id in masters}
    series_ids_by_length = sorted(masters, key=len, reverse=True)
    for path in images:
        if path.stem.lower().endswith("_max_flat"):
            continue
        matching_series = next(
            (series_id for series_id in series_ids_by_length if path.stem.startswith(series_id + "_")),
            None,
        )
        if matching_series is not None:
            grouped[matching_series].append(path)

    result: dict[str, SeriesFiles] = {}
    for series_id, master in sorted(masters.items()):
        master_width, master_height = _read_image_size(master)
        slaves = sorted(grouped.get(series_id, []))
        for slave in slaves:
            width, height = _read_image_size(slave)
            if (width, height) != (master_width, master_height):
                raise ValueError(
                    f"Resolution mismatch in series '{series_id}': "
                    f"{slave.name} has {width}x{height}, master has "
                    f"{master_width}x{master_height}"
                )
        result[series_id] = SeriesFiles(
            series_id=series_id,
            master=master,
            slaves=tuple(slaves),
            width=master_width,
            height=master_height,
        )

    if not result:
        raise ValueError(
            f"No master image matching '<series>_max_flat.<ext>' was found in {images_dir}"
        )
    return result


def new_stroke(
    points: Iterable[tuple[float, float]],
    width_px: int,
    *,
    source: str,
    zoom_violation: bool = False,
) -> dict[str, Any]:
    """Create a serializable scratch stroke."""

    clean_points = [[round(float(x), 3), round(float(y), 3)] for x, y in points]
    return {
        "id": uuid.uuid4().hex,
        "points": clean_points,
        "width_px": int(width_px),
        "source": source,
        "zoom_violation": bool(zoom_violation),
        "accepted_short": False,
        "accepted_zoom": False,
    }


def stroke_length_px(stroke: dict[str, Any]) -> float:
    """Return polyline length in original image pixels."""

    points = stroke.get("points", [])
    if len(points) < 2:
        return 0.0
    return float(
        sum(
            math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
            for a, b in zip(points, points[1:])
        )
    )


def stroke_warnings(stroke: dict[str, Any], config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if stroke_length_px(stroke) < float(config["min_scratch_length_px"]):
        if not bool(stroke.get("accepted_short", False)):
            warnings.append("short")
    if bool(stroke.get("zoom_violation", False)) and not bool(
        stroke.get("accepted_zoom", False)
    ):
        warnings.append("zoom")
    return warnings


def create_project(series: SeriesFiles, config: dict[str, Any]) -> dict[str, Any]:
    """Create an empty project dictionary for one image series."""

    return {
        "version": PROJECT_VERSION,
        "series_id": series.series_id,
        "image_size": [series.width, series.height],
        "master_file": series.master.name,
        "slave_files": [path.name for path in series.slaves],
        "settings": {
            "max_annotation_zoom": float(config["max_annotation_zoom"]),
            "min_scratch_length_px": float(config["min_scratch_length_px"]),
        },
        "master": {
            "strokes": [],
            "status": "not_started",
            "locked": False,
            "zoom_edit_violation": False,
            "accepted_zoom_edit": False,
        },
        "slaves": {},
    }


def normalize_project(project: dict[str, Any], series: SeriesFiles) -> dict[str, Any]:
    """Add fields introduced by newer versions without discarding user data."""

    project.setdefault("version", PROJECT_VERSION)
    project.setdefault("series_id", series.series_id)
    project["image_size"] = [series.width, series.height]
    project["master_file"] = series.master.name
    project["slave_files"] = [path.name for path in series.slaves]
    master = project.setdefault("master", {})
    master.setdefault("strokes", [])
    master.setdefault("status", "not_started")
    master.setdefault("locked", False)
    master.setdefault("zoom_edit_violation", False)
    master.setdefault("accepted_zoom_edit", False)
    project.setdefault("slaves", {})
    return project


def load_project(path: Path, series: SeriesFiles, config: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return create_project(series, config)
    with path.open("r", encoding="utf-8") as handle:
        project = json.load(handle)
    if not isinstance(project, dict):
        raise ValueError(f"Invalid project file: {path}")
    return normalize_project(project, series)


def save_project_atomic(path: Path, project: dict[str, Any]) -> None:
    """Atomically save JSON to reduce the risk of corruption after a crash."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(project, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def ensure_slave(project: dict[str, Any], slave_filename: str) -> dict[str, Any]:
    """Initialize a slave from the current master, without locking the master yet."""

    slaves = project.setdefault("slaves", {})
    state = slaves.get(slave_filename)
    if state is None:
        state = {
            "base_strokes": copy.deepcopy(project["master"]["strokes"]),
            "added_strokes": [],
            "hidden_base_ids": [],
            "erase_rects": [],
            "edit_operations": [],
            "clear_base": False,
            "zoom_erase_violation": False,
            "accepted_zoom_erase": False,
            "status": "not_started",
            "modified": False,
        }
        slaves[slave_filename] = state
    elif not state.get("modified", False) and state.get("status") == "not_started":
        # An untouched slave should follow later master edits until the first real edit.
        state["base_strokes"] = copy.deepcopy(project["master"]["strokes"])
    state.setdefault("base_strokes", [])
    state.setdefault("added_strokes", [])
    state.setdefault("hidden_base_ids", [])
    state.setdefault("erase_rects", [])

    # Older project files stored added strokes and erase rectangles separately.
    # Their effective order was: all strokes first, then all erase rectangles.
    # Preserve that exact result once, while allowing future draw/erase actions
    # to be replayed chronologically. This prevents a historic erase rectangle
    # from clipping a new scratch drawn afterwards.
    if "edit_operations" not in state:
        operations: list[dict[str, Any]] = [
            {"type": "add_stroke", "stroke_id": str(stroke.get("id", ""))}
            for stroke in state.get("added_strokes", [])
            if stroke.get("id")
        ]
        operations.extend(
            {"type": "erase_rect", "rect": list(rect)}
            for rect in state.get("erase_rects", [])
            if isinstance(rect, (list, tuple)) and len(rect) == 4
        )
        state["edit_operations"] = operations
    elif not isinstance(state.get("edit_operations"), list):
        state["edit_operations"] = []

    # Keep manually edited or partially migrated JSON files usable. Missing
    # stroke operations are inserted first and missing erase operations last,
    # which reproduces the rendering semantics of the original project format.
    operations = state["edit_operations"]
    referenced_stroke_ids = {
        str(operation.get("stroke_id", ""))
        for operation in operations
        if isinstance(operation, dict) and operation.get("type") == "add_stroke"
    }
    for stroke in state.get("added_strokes", []):
        stroke_id = str(stroke.get("id", ""))
        if stroke_id and stroke_id not in referenced_stroke_ids:
            operations.append({"type": "add_stroke", "stroke_id": stroke_id})
            referenced_stroke_ids.add(stroke_id)

    unmatched_operation_rects = [
        tuple(operation.get("rect", []))
        for operation in operations
        if isinstance(operation, dict)
        and operation.get("type") == "erase_rect"
        and len(operation.get("rect", [])) == 4
    ]
    for rect in state.get("erase_rects", []):
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            continue
        rect_tuple = tuple(rect)
        if rect_tuple in unmatched_operation_rects:
            unmatched_operation_rects.remove(rect_tuple)
        else:
            operations.append({"type": "erase_rect", "rect": list(rect)})

    state.setdefault("clear_base", False)
    state.setdefault("zoom_erase_violation", False)
    state.setdefault("accepted_zoom_erase", False)
    state.setdefault("status", "not_started")
    state.setdefault("modified", False)
    return state


def mark_slave_modified(project: dict[str, Any], state: dict[str, Any]) -> None:
    state["modified"] = True
    if state.get("status") in {"not_started", "finished", "finished_with_exceptions"}:
        state["status"] = "in_progress"
    project["master"]["locked"] = True


def current_strokes(
    project: dict[str, Any], image_filename: str
) -> list[tuple[str, dict[str, Any]]]:
    """Return visible strokes as ``(kind, stroke)`` pairs.

    kind is ``master``, ``base`` or ``added`` and is used by the GUI to apply
    edits to the correct collection.
    """

    if image_filename == project["master_file"]:
        return [("master", stroke) for stroke in project["master"]["strokes"]]

    state = ensure_slave(project, image_filename)
    hidden = set(state.get("hidden_base_ids", []))
    visible: list[tuple[str, dict[str, Any]]] = []
    if not state.get("clear_base", False):
        visible.extend(
            ("base", stroke)
            for stroke in state.get("base_strokes", [])
            if stroke.get("id") not in hidden
        )
    visible.extend(("added", stroke) for stroke in state.get("added_strokes", []))
    return visible


def find_stroke(
    project: dict[str, Any], image_filename: str, stroke_id: str
) -> tuple[str, dict[str, Any]] | None:
    for kind, stroke in current_strokes(project, image_filename):
        if stroke.get("id") == stroke_id:
            return kind, stroke
    return None


def remove_stroke(project: dict[str, Any], image_filename: str, stroke_id: str) -> bool:
    """Remove a master stroke, hide a slave base stroke, or delete an added stroke."""

    if image_filename == project["master_file"]:
        strokes = project["master"]["strokes"]
        old_length = len(strokes)
        strokes[:] = [stroke for stroke in strokes if stroke.get("id") != stroke_id]
        return len(strokes) != old_length

    state = ensure_slave(project, image_filename)
    for stroke in state["base_strokes"]:
        if stroke.get("id") == stroke_id:
            hidden = state.setdefault("hidden_base_ids", [])
            if stroke_id not in hidden:
                hidden.append(stroke_id)
                mark_slave_modified(project, state)
                return True
            return False
    old_length = len(state["added_strokes"])
    state["added_strokes"] = [
        stroke for stroke in state["added_strokes"] if stroke.get("id") != stroke_id
    ]
    if len(state["added_strokes"]) != old_length:
        mark_slave_modified(project, state)
        return True
    return False


def _draw_stroke(mask: np.ndarray, stroke: dict[str, Any]) -> None:
    points = np.asarray(stroke.get("points", []), dtype=np.float64)
    if len(points) < 2:
        return
    height, width = mask.shape
    points[:, 0] = np.clip(np.rint(points[:, 0]), 0, width - 1)
    points[:, 1] = np.clip(np.rint(points[:, 1]), 0, height - 1)
    points_i32 = points.astype(np.int32).reshape((-1, 1, 2))
    thickness = max(1, int(stroke.get("width_px", 1)))
    cv2.polylines(
        mask,
        [points_i32],
        isClosed=False,
        color=255,
        thickness=thickness,
        lineType=cv2.LINE_8,
    )


def _erase_mask_rectangle(
    mask: np.ndarray, rect: Iterable[float], width: int, height: int
) -> None:
    """Erase one rectangular image-coordinate area from ``mask`` in-place."""

    values = list(rect)
    if len(values) != 4:
        return
    x1, y1, x2, y2 = map(float, values)
    left = max(0, min(width - 1, int(math.floor(min(x1, x2)))))
    right = max(0, min(width - 1, int(math.ceil(max(x1, x2)))))
    top = max(0, min(height - 1, int(math.floor(min(y1, y2)))))
    bottom = max(0, min(height - 1, int(math.ceil(max(y1, y2)))))
    mask[top : bottom + 1, left : right + 1] = 0


def render_mask(project: dict[str, Any], image_filename: str) -> np.ndarray:
    """Rasterize the current image annotation to a binary uint8 mask.

    Slave edits are replayed in their original order. Therefore, a rectangle
    erases only annotations that already existed when the rectangle was drawn;
    a new scratch added afterwards remains visible inside that former area.
    """

    width, height = map(int, project["image_size"])
    mask = np.zeros((height, width), dtype=np.uint8)

    if image_filename == project["master_file"]:
        for stroke in project["master"].get("strokes", []):
            _draw_stroke(mask, stroke)
    else:
        state = ensure_slave(project, image_filename)
        hidden = set(state.get("hidden_base_ids", []))

        if not state.get("clear_base", False):
            for stroke in state.get("base_strokes", []):
                if stroke.get("id") not in hidden:
                    _draw_stroke(mask, stroke)

        added_by_id = {
            str(stroke.get("id")): stroke
            for stroke in state.get("added_strokes", [])
            if stroke.get("id")
        }
        drawn_added_ids: set[str] = set()

        for operation in state.get("edit_operations", []):
            if not isinstance(operation, dict):
                continue
            operation_type = operation.get("type")
            if operation_type == "add_stroke":
                stroke_id = str(operation.get("stroke_id", ""))
                stroke = added_by_id.get(stroke_id)
                if stroke is not None:
                    _draw_stroke(mask, stroke)
                    drawn_added_ids.add(stroke_id)
            elif operation_type == "erase_rect":
                _erase_mask_rectangle(
                    mask, operation.get("rect", []), width, height
                )

        # Defensive compatibility for manually edited or partially migrated
        # project JSON files: do not silently lose an unreferenced added stroke.
        for stroke_id, stroke in added_by_id.items():
            if stroke_id not in drawn_added_ids:
                _draw_stroke(mask, stroke)

    # Defensive final binarization: no antialiasing or intermediate grey values.
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def image_open_warnings(
    project: dict[str, Any], image_filename: str, config: dict[str, Any]
) -> list[str]:
    warnings: list[str] = []
    for _, stroke in current_strokes(project, image_filename):
        warnings.extend(
            f"{stroke.get('id')}:{warning}" for warning in stroke_warnings(stroke, config)
        )
    if image_filename == project["master_file"]:
        master = project["master"]
        if master.get("zoom_edit_violation", False) and not master.get(
            "accepted_zoom_edit", False
        ):
            warnings.append("image:zoom_edit")
    else:
        state = ensure_slave(project, image_filename)
        if state.get("zoom_erase_violation", False) and not state.get(
            "accepted_zoom_erase", False
        ):
            warnings.append("image:zoom_erase")
    return warnings


def completion_status(
    project: dict[str, Any], image_filename: str, config: dict[str, Any]
) -> tuple[bool, str, list[str]]:
    """Validate and return ``(can_finish, final_status, warnings)``."""

    warnings = image_open_warnings(project, image_filename, config)
    if warnings:
        return False, "in_progress", warnings

    has_accepted_exception = False
    for _, stroke in current_strokes(project, image_filename):
        if stroke.get("accepted_short") or stroke.get("accepted_zoom"):
            has_accepted_exception = True
    if image_filename == project["master_file"]:
        has_accepted_exception = has_accepted_exception or bool(
            project["master"].get("accepted_zoom_edit", False)
        )
    else:
        state = ensure_slave(project, image_filename)
        has_accepted_exception = has_accepted_exception or bool(
            state.get("accepted_zoom_erase", False)
        )
    return (
        True,
        "finished_with_exceptions" if has_accepted_exception else "finished",
        [],
    )


def write_mask(path: Path, mask: np.ndarray, expected_size: tuple[int, int]) -> None:
    """Write a validated lossless binary PNG mask."""

    expected_width, expected_height = expected_size
    if mask.dtype != np.uint8:
        raise ValueError("Mask dtype must be uint8")
    if mask.shape != (expected_height, expected_width):
        raise ValueError(
            f"Mask size {mask.shape[1]}x{mask.shape[0]} does not match "
            f"image size {expected_width}x{expected_height}"
        )
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 255}):
        raise ValueError(f"Mask contains non-binary values: {sorted(values)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".png":
        raise ValueError("Masks must be written as PNG")
    if not cv2.imwrite(str(path), mask):
        raise OSError(f"Mask could not be written: {path}")


def deep_snapshot(value: Any) -> Any:
    return copy.deepcopy(value)
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from annotation.scratch_core import (
    completion_status,
    create_project,
    discover_series,
    ensure_slave,
    load_config,
    load_project,
    mark_slave_modified,
    new_stroke,
    render_mask,
    save_project_atomic,
    stroke_length_px,
    write_mask,
)


class ScratchCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(
            "max_annotation_zoom: 5.0\nmin_scratch_length_px: 35\n",
            encoding="utf-8",
        )
        self.config = load_config(self.config_path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_image(self, name: str, width: int = 100, height: int = 80) -> None:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.images / name), image))

    def test_variable_slave_count_and_resolution(self) -> None:
        self.write_image("01_all.bmp")
        for index in range(13):
            self.write_image(f"01_slave_{index:02d}.bmp")
        series = discover_series(self.images)["01"]
        self.assertEqual(len(series.slaves), 13)
        self.assertEqual((series.width, series.height), (100, 80))

    def test_series_id_may_contain_underscores(self) -> None:
        self.write_image("part_01_all.bmp")
        self.write_image("part_01_left.bmp")
        series = discover_series(self.images)["part_01"]
        self.assertEqual([path.name for path in series.slaves], ["part_01_left.bmp"])

    def test_resolution_mismatch_is_rejected(self) -> None:
        self.write_image("02_all.bmp")
        self.write_image("02_left.bmp", width=99)
        with self.assertRaises(ValueError):
            discover_series(self.images)

    def test_mask_is_binary_original_size_and_individual_widths(self) -> None:
        self.write_image("03_all.bmp")
        self.write_image("03_left.bmp")
        series = discover_series(self.images)["03"]
        project = create_project(series, self.config)
        thin = new_stroke([(10, 10), (90, 10)], 3, source="master")
        thick = new_stroke([(10, 30), (90, 30)], 11, source="master")
        project["master"]["strokes"] = [thin, thick]
        mask = render_mask(project, series.master.name)
        self.assertEqual(mask.shape, (80, 100))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(set(np.unique(mask)).issubset({0, 255}))
        self.assertGreater(np.count_nonzero(mask[25:36]), np.count_nonzero(mask[8:13]))

    def test_slave_copy_is_independent_and_rectangle_erases_only_area(self) -> None:
        self.write_image("04_all.bmp")
        self.write_image("04_left.bmp")
        series = discover_series(self.images)["04"]
        project = create_project(series, self.config)
        stroke = new_stroke([(5, 40), (95, 40)], 7, source="master")
        project["master"]["strokes"] = [stroke]
        slave = ensure_slave(project, "04_left.bmp")
        slave["erase_rects"].append([40, 0, 60, 79])
        mark_slave_modified(project, slave)
        slave_mask = render_mask(project, "04_left.bmp")
        master_mask = render_mask(project, "04_all.bmp")
        self.assertGreater(np.count_nonzero(master_mask[:, 45:56]), 0)
        self.assertEqual(np.count_nonzero(slave_mask[:, 45:56]), 0)
        self.assertGreater(np.count_nonzero(slave_mask[:, 5:30]), 0)
        self.assertEqual(len(project["master"]["strokes"]), 1)

    def test_short_and_zoom_warnings_require_acceptance(self) -> None:
        self.write_image("05_all.bmp")
        series = discover_series(self.images)["05"]
        project = create_project(series, self.config)
        stroke = new_stroke([(10, 10), (20, 10)], 5, source="master", zoom_violation=True)
        project["master"]["strokes"] = [stroke]
        can_finish, _, warnings = completion_status(project, "05_all.bmp", self.config)
        self.assertFalse(can_finish)
        self.assertEqual(len(warnings), 2)
        stroke["accepted_short"] = True
        stroke["accepted_zoom"] = True
        can_finish, status, warnings = completion_status(project, "05_all.bmp", self.config)
        self.assertTrue(can_finish)
        self.assertEqual(status, "finished_with_exceptions")
        self.assertEqual(warnings, [])

    def test_polyline_length_uses_original_pixels(self) -> None:
        stroke = new_stroke([(0, 0), (3, 4), (6, 8)], 5, source="master")
        self.assertAlmostEqual(stroke_length_px(stroke), 10.0)

    def test_master_image_level_zoom_edit_requires_acceptance(self) -> None:
        self.write_image("07_all.bmp")
        series = discover_series(self.images)["07"]
        project = create_project(series, self.config)
        project["master"]["zoom_edit_violation"] = True
        can_finish, _, warnings = completion_status(project, "07_all.bmp", self.config)
        self.assertFalse(can_finish)
        self.assertIn("image:zoom_edit", warnings)
        project["master"]["accepted_zoom_edit"] = True
        can_finish, status, _ = completion_status(project, "07_all.bmp", self.config)
        self.assertTrue(can_finish)
        self.assertEqual(status, "finished_with_exceptions")

    def test_atomic_json_and_png_roundtrip(self) -> None:
        self.write_image("06_all.bmp")
        series = discover_series(self.images)["06"]
        project = create_project(series, self.config)
        project["master"]["strokes"].append(
            new_stroke([(10, 10), (70, 10)], 5, source="master")
        )
        json_path = self.root / "annotations" / "06.json"
        save_project_atomic(json_path, project)
        loaded = load_project(json_path, series, self.config)
        self.assertEqual(loaded["series_id"], "06")
        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["version"], 1)
        mask = render_mask(loaded, "06_all.bmp")
        mask_path = self.root / "masks" / "06_all.png"
        write_mask(mask_path, mask, (100, 80))
        read_back = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(read_back)
        self.assertEqual(read_back.shape, (80, 100))
        self.assertTrue(set(np.unique(read_back)).issubset({0, 255}))


if __name__ == "__main__":
    unittest.main()

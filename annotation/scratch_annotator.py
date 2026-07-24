"""Local desktop tool for pixel-accurate scratch annotation.

Run from the repository root with:

    uv run python annotation/scratch_annotator.py

The GUI uses Tkinter so no heavy UI framework is required. Images are only
rescaled for screen display. Annotation coordinates and exported masks always
use the untouched original image resolution.
"""

from __future__ import annotations

import copy
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:  # pragma: no cover - depends on OS package
    raise SystemExit(
        "Tkinter is not installed. On Ubuntu run: sudo apt install python3-tk"
    ) from exc

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from annotation.scratch_core import (  # noqa: E402
    SeriesFiles,
    completion_status,
    current_strokes,
    deep_snapshot,
    discover_series,
    ensure_slave,
    find_stroke,
    image_open_warnings,
    load_config,
    load_project,
    mark_slave_modified,
    new_stroke,
    remove_stroke,
    render_mask,
    save_project_atomic,
    stroke_length_px,
    stroke_warnings,
    write_mask,
)

APP_TITLE = "Scratch Annotator"
STATUS_SYMBOLS = {
    "not_started": "○",
    "in_progress": "◐",
    "finished": "✓",
    "finished_with_exceptions": "⚠",
}
STATUS_LABELS = {
    "not_started": "Nicht begonnen",
    "in_progress": "In Bearbeitung",
    "finished": "Fertig",
    "finished_with_exceptions": "Fertig mit Ausnahmen",
}


@dataclass
class SharedView:
    scale: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0


class ImageCanvas(tk.Canvas):
    """Viewport-rendered image canvas using shared zoom and center coordinates."""

    def __init__(self, master: tk.Widget, app: "ScratchAnnotatorApp", *, editable: bool):
        super().__init__(
            master,
            background="#202124",
            highlightthickness=1,
            highlightbackground="#5f6368",
            cursor="crosshair" if editable else "arrow",
        )
        self.app = app
        self.editable = editable
        self._photo: ImageTk.PhotoImage | None = None
        self._pan_start: tuple[int, int, float, float] | None = None
        self._erase_start: tuple[float, float] | None = None
        self._drag_point_index: int | None = None
        self._dragging_point = False
        self._resize_job: str | None = None

        self.bind("<Configure>", self._on_configure)
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Button-4>", lambda event: self._zoom_linux(event, 1))
        self.bind("<Button-5>", lambda event: self._zoom_linux(event, -1))
        self.bind("<ButtonPress-2>", self._start_pan)
        self.bind("<B2-Motion>", self._pan)
        self.bind("<ButtonRelease-2>", self._end_pan)
        self.bind("<Shift-ButtonPress-1>", self._start_pan)
        self.bind("<Shift-B1-Motion>", self._pan)
        self.bind("<Shift-ButtonRelease-1>", self._end_pan)
        self.bind("<ButtonPress-1>", self._left_press)
        self.bind("<B1-Motion>", self._left_motion)
        self.bind("<ButtonRelease-1>", self._left_release)
        self.bind("<ButtonPress-3>", self._right_press)

    def _on_configure(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(35, self.app.render_all)

    def _on_mousewheel(self, event: tk.Event) -> None:
        direction = 1 if event.delta > 0 else -1
        self.app.zoom_at(self, event.x, event.y, direction)

    def _zoom_linux(self, event: tk.Event, direction: int) -> str:
        self.app.zoom_at(self, event.x, event.y, direction)
        return "break"

    def _start_pan(self, event: tk.Event) -> str:
        self._pan_start = (
            event.x,
            event.y,
            self.app.view.center_x,
            self.app.view.center_y,
        )
        self.configure(cursor="fleur")
        return "break"

    def _pan(self, event: tk.Event) -> str:
        if self._pan_start is None:
            return "break"
        start_x, start_y, center_x, center_y = self._pan_start
        scale = max(self.app.view.scale, 1e-9)
        self.app.view.center_x = center_x - (event.x - start_x) / scale
        self.app.view.center_y = center_y - (event.y - start_y) / scale
        self.app.clamp_view()
        self.app.render_all()
        return "break"

    def _end_pan(self, _event: tk.Event) -> str:
        self._pan_start = None
        self.configure(cursor="crosshair" if self.editable else "arrow")
        return "break"

    def _left_press(self, event: tk.Event) -> str | None:
        if event.state & 0x0001:  # Shift is handled by the pan bindings.
            return "break"
        if not self.editable or not self.app.current_image:
            return None
        if not self.app.editing_allowed():
            self.app.show_locked_master_message()
            return "break"

        x, y = self.screen_to_image(event.x, event.y)
        if not self.app.point_inside_image(x, y):
            return "break"

        mode = self.app.mode_var.get()
        if mode == "draw":
            self.app.add_polyline_point(x, y)
        elif mode == "select":
            self.app.select_nearest_stroke(x, y)
            point_index = self.app.nearest_selected_point(x, y, tolerance_screen_px=10)
            if point_index is not None:
                self._drag_point_index = point_index
                self._dragging_point = True
                self.app.begin_geometry_edit()
        elif mode == "erase_rect":
            if self.app.is_master_image():
                messagebox.showinfo(
                    APP_TITLE,
                    "Der Bereichslöscher steht nur für Slave-Bilder zur Verfügung.",
                    parent=self.app.root,
                )
            else:
                self._erase_start = (x, y)
                self.app.preview_erase_rect = (x, y, x, y)
                self.app.render_all()
        elif mode == "pan":
            return self._start_pan(event)
        return "break"

    def _left_motion(self, event: tk.Event) -> str | None:
        if not self.editable:
            return None
        x, y = self.screen_to_image(event.x, event.y)
        x, y = self.app.clamp_point(x, y)
        if self._dragging_point and self._drag_point_index is not None:
            self.app.move_selected_point(self._drag_point_index, x, y)
            return "break"
        if self._erase_start is not None:
            sx, sy = self._erase_start
            self.app.preview_erase_rect = (sx, sy, x, y)
            self.app.render_all()
            return "break"
        return None

    def _left_release(self, event: tk.Event) -> str | None:
        if self._dragging_point:
            self._dragging_point = False
            self._drag_point_index = None
            self.app.end_geometry_edit()
            return "break"
        if self._erase_start is not None:
            x, y = self.screen_to_image(event.x, event.y)
            x, y = self.app.clamp_point(x, y)
            sx, sy = self._erase_start
            self._erase_start = None
            self.app.preview_erase_rect = None
            if abs(x - sx) >= 1 or abs(y - sy) >= 1:
                self.app.apply_erase_rectangle(sx, sy, x, y)
            else:
                self.app.render_all()
            return "break"
        return None

    def _right_press(self, _event: tk.Event) -> str:
        if self.editable and self.app.mode_var.get() == "draw":
            self.app.finish_polyline()
        return "break"

    def screen_to_image(self, screen_x: float, screen_y: float) -> tuple[float, float]:
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        scale = max(self.app.view.scale, 1e-9)
        return (
            self.app.view.center_x + (screen_x - width / 2) / scale,
            self.app.view.center_y + (screen_y - height / 2) / scale,
        )

    def image_to_screen(self, image_x: float, image_y: float) -> tuple[float, float]:
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        return (
            (image_x - self.app.view.center_x) * self.app.view.scale + width / 2,
            (image_y - self.app.view.center_y) * self.app.view.scale + height / 2,
        )

    def render(self) -> None:
        self.delete("all")
        image = self.app.current_image
        if image is None:
            self.create_text(
                max(1, self.winfo_width()) / 2,
                max(1, self.winfo_height()) / 2,
                text="Kein Bild geladen",
                fill="#e8eaed",
                font=("TkDefaultFont", 13),
            )
            return

        canvas_width = max(2, self.winfo_width())
        canvas_height = max(2, self.winfo_height())
        scale = max(self.app.view.scale, 1e-6)
        left = self.app.view.center_x - canvas_width / (2 * scale)
        top = self.app.view.center_y - canvas_height / (2 * scale)
        right = self.app.view.center_x + canvas_width / (2 * scale)
        bottom = self.app.view.center_y + canvas_height / (2 * scale)

        clip_left = max(0.0, left)
        clip_top = max(0.0, top)
        clip_right = min(float(image.width), right)
        clip_bottom = min(float(image.height), bottom)

        frame = Image.new("RGB", (canvas_width, canvas_height), "#202124")
        if clip_right > clip_left and clip_bottom > clip_top:
            crop_box = (
                int(math.floor(clip_left)),
                int(math.floor(clip_top)),
                int(math.ceil(clip_right)),
                int(math.ceil(clip_bottom)),
            )
            crop = image.crop(crop_box)
            display_width = max(1, int(round((crop_box[2] - crop_box[0]) * scale)))
            display_height = max(1, int(round((crop_box[3] - crop_box[1]) * scale)))
            resample = Image.Resampling.NEAREST if scale >= 1 else Image.Resampling.LANCZOS
            crop = crop.resize((display_width, display_height), resample=resample)

            paste_x = int(round((crop_box[0] - left) * scale))
            paste_y = int(round((crop_box[1] - top) * scale))
            frame.paste(crop, (paste_x, paste_y))

            if self.editable and not self.app.overlay_hidden:
                overlay_crop = self.app.create_overlay_crop(crop_box)
                overlay_crop = overlay_crop.resize(
                    (display_width, display_height), resample=Image.Resampling.NEAREST
                )
                rgba = frame.convert("RGBA")
                rgba.alpha_composite(overlay_crop, (paste_x, paste_y))
                frame = rgba.convert("RGB")

        self._photo = ImageTk.PhotoImage(frame)
        self.create_image(0, 0, image=self._photo, anchor="nw")

        if self.editable:
            self._draw_vector_guides()

        zoom_text = f"{self.app.view.scale:.2f}×"
        self.create_rectangle(8, 8, 80, 34, fill="#202124", outline="#5f6368")
        self.create_text(44, 21, text=zoom_text, fill="#e8eaed")

    def _draw_vector_guides(self) -> None:
        for _, stroke in self.app.visible_strokes():
            warnings = stroke_warnings(stroke, self.app.config)
            if not warnings:
                continue
            coords: list[float] = []
            for x, y in stroke.get("points", []):
                sx, sy = self.image_to_screen(float(x), float(y))
                coords.extend((sx, sy))
            if len(coords) >= 4:
                line_width = max(2, min(12, int(stroke.get("width_px", 1) * self.app.view.scale)))
                self.create_line(
                    *coords,
                    fill="#ff9800",
                    width=line_width,
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                )

        selected = self.app.get_selected_stroke()
        if selected is not None:
            _, stroke = selected
            coords = []
            for x, y in stroke.get("points", []):
                sx, sy = self.image_to_screen(float(x), float(y))
                coords.extend((sx, sy))
            if len(coords) >= 4:
                self.create_line(
                    *coords,
                    fill="#ffd54f",
                    width=max(2, min(8, int(stroke.get("width_px", 1) * self.app.view.scale))),
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                )
            for index, (x, y) in enumerate(stroke.get("points", [])):
                sx, sy = self.image_to_screen(float(x), float(y))
                radius = 5 if index not in {0, len(stroke.get("points", [])) - 1} else 6
                self.create_oval(
                    sx - radius,
                    sy - radius,
                    sx + radius,
                    sy + radius,
                    fill="#ffd54f",
                    outline="#3c4043",
                    width=1,
                )

        if self.app.current_polyline:
            coords = []
            for x, y in self.app.current_polyline:
                sx, sy = self.image_to_screen(x, y)
                coords.extend((sx, sy))
            if len(coords) >= 4:
                self.create_line(
                    *coords,
                    fill="#00e5ff",
                    width=max(2, min(10, int(self.app.width_var.get() * self.app.view.scale))),
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                )
            for x, y in self.app.current_polyline:
                sx, sy = self.image_to_screen(x, y)
                self.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, fill="#00e5ff", outline="")

        if self.app.preview_erase_rect is not None:
            x1, y1, x2, y2 = self.app.preview_erase_rect
            sx1, sy1 = self.image_to_screen(x1, y1)
            sx2, sy2 = self.image_to_screen(x2, y2)
            self.create_rectangle(
                sx1,
                sy1,
                sx2,
                sy2,
                outline="#ff5252",
                width=2,
                dash=(6, 4),
            )


class ScratchAnnotatorApp:
    def __init__(self, root: tk.Tk, annotation_dir: Path):
        self.root = root
        self.annotation_dir = annotation_dir.resolve()
        self.config_path = self.annotation_dir / "config.yaml"
        self.config = load_config(self.config_path)
        self.images_dir = self._resolve_config_dir("images_dir")
        self.annotations_dir = self._resolve_config_dir("annotations_dir")
        self.masks_dir = self._resolve_config_dir("masks_dir")

        self.series_map: dict[str, SeriesFiles] = {}
        self.project: dict[str, Any] | None = None
        self.current_series: SeriesFiles | None = None
        self.current_filename: str | None = None
        self.current_image: Image.Image | None = None
        self.current_mask: np.ndarray | None = None
        self.mask_dirty = True
        self.selected_stroke_id: str | None = None
        self.current_polyline: list[tuple[float, float]] = []
        self.current_polyline_zoom_violation = False
        self.preview_erase_rect: tuple[float, float, float, float] | None = None
        self.overlay_hidden = False
        self.view = SharedView()
        self.undo_stack: list[Any] = []
        self.redo_stack: list[Any] = []
        self._geometry_snapshot_active = False
        self._suppress_width_update = False
        self._image_filenames: list[str] = []
        self._listbox_update_active = False

        self.mode_var = tk.StringVar(value="draw")
        self.series_var = tk.StringVar()
        self.width_var = tk.IntVar(value=int(self.config["default_width_px"]))
        self.opacity_var = tk.DoubleVar(value=float(self.config["overlay_opacity"]))
        self.zoom_status_var = tk.StringVar(value="Zoomrichtlinie: ✓")
        self.selection_info_var = tk.StringVar(value="Kein Kratzer ausgewählt")
        self.image_status_var = tk.StringVar(value="Kein Bild geladen")
        self.master_lock_var = tk.StringVar(value="")

        self._build_ui()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._initial_load)

    def _resolve_config_dir(self, key: str) -> Path:
        value = Path(str(self.config[key]))
        return value if value.is_absolute() else self.annotation_dir / value

    def _build_ui(self) -> None:
        self.root.title(APP_TITLE)
        self.root.geometry("1500x900")
        self.root.minsize(1050, 650)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(side="top", fill="x")
        ttk.Button(toolbar, text="Bilderordner öffnen", command=self.choose_images_folder).pack(
            side="left", padx=(0, 6)
        )
        ttk.Label(toolbar, text="Bildreihe:").pack(side="left", padx=(8, 4))
        self.series_combo = ttk.Combobox(
            toolbar, textvariable=self.series_var, state="readonly", width=16
        )
        self.series_combo.pack(side="left")
        self.series_combo.bind("<<ComboboxSelected>>", self._on_series_selected)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(toolbar, text="Speichern", command=self.save_current).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Bild als fertig markieren", command=self.finish_current_image).pack(
            side="left", padx=3
        )
        ttk.Button(toolbar, text="Rückgängig", command=self.undo).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Wiederholen", command=self.redo).pack(side="left", padx=3)
        ttk.Label(toolbar, textvariable=self.master_lock_var).pack(side="right", padx=8)

        main = ttk.Panedwindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True)

        sidebar = ttk.Frame(main, padding=6)
        main.add(sidebar, weight=0)
        ttk.Label(sidebar, text="Bilder der Reihe", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", pady=(0, 5)
        )
        list_frame = ttk.Frame(sidebar)
        list_frame.pack(fill="both", expand=True)
        self.image_list = tk.Listbox(
            list_frame,
            width=32,
            activestyle="dotbox",
            exportselection=False,
            font=("TkFixedFont", 10),
        )
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.image_list.yview)
        self.image_list.configure(yscrollcommand=list_scroll.set)
        self.image_list.pack(side="left", fill="both", expand=True)
        list_scroll.pack(side="right", fill="y")
        self.image_list.bind("<<ListboxSelect>>", self._on_image_selected)

        center = ttk.Frame(main, padding=4)
        main.add(center, weight=1)
        view_pane = ttk.Panedwindow(center, orient="horizontal")
        view_pane.pack(fill="both", expand=True)

        left_frame = ttk.Labelframe(view_pane, text="Bearbeitung: Slave/Master mit Maske")
        right_frame = ttk.Labelframe(view_pane, text="Referenz: unverändertes Bild")
        view_pane.add(left_frame, weight=1)
        view_pane.add(right_frame, weight=1)
        self.left_canvas = ImageCanvas(left_frame, self, editable=True)
        self.right_canvas = ImageCanvas(right_frame, self, editable=False)
        self.left_canvas.pack(fill="both", expand=True)
        self.right_canvas.pack(fill="both", expand=True)

        controls = ttk.Frame(main, padding=8, width=280)
        main.add(controls, weight=0)
        self._build_controls(controls)

        statusbar = ttk.Frame(self.root, padding=(8, 4))
        statusbar.pack(side="bottom", fill="x")
        self.status_label = ttk.Label(statusbar, textvariable=self.image_status_var)
        self.status_label.pack(side="left")
        ttk.Label(
            statusbar,
            text="Mausrad: Zoom · Mittlere Taste/Shift+Ziehen: Verschieben · Rechtsklick/Enter: Linie beenden",
        ).pack(side="right")

    def _build_controls(self, controls: ttk.Frame) -> None:
        ttk.Label(controls, text="Werkzeug", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w"
        )
        modes = [
            ("Polyline zeichnen", "draw"),
            ("Auswählen / Punkte verschieben", "select"),
            ("Bereichslöscher", "erase_rect"),
            ("Ansicht verschieben", "pan"),
        ]
        for label, value in modes:
            ttk.Radiobutton(
                controls,
                text=label,
                value=value,
                variable=self.mode_var,
                command=self._on_mode_changed,
            ).pack(anchor="w", pady=1)

        ttk.Separator(controls).pack(fill="x", pady=8)
        ttk.Label(controls, text="Finale Kratzerbreite").pack(anchor="w")
        width_scale = tk.Scale(
            controls,
            from_=int(self.config["min_width_px"]),
            to=int(self.config["max_width_px"]),
            orient="horizontal",
            variable=self.width_var,
            resolution=1,
            showvalue=True,
            length=240,
        )
        width_scale.pack(fill="x")
        ttk.Button(
            controls,
            text="Breite auf Auswahl anwenden",
            command=self.apply_width_to_selected,
        ).pack(fill="x", pady=(2, 6))

        ttk.Label(controls, text="Overlay-Deckkraft").pack(anchor="w")
        opacity_scale = ttk.Scale(
            controls,
            from_=0.05,
            to=0.9,
            variable=self.opacity_var,
            command=lambda _value: self.render_all(),
        )
        opacity_scale.pack(fill="x", pady=(2, 8))

        ttk.Separator(controls).pack(fill="x", pady=6)
        ttk.Label(controls, text="Auswahl", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            controls,
            textvariable=self.selection_info_var,
            wraplength=255,
            justify="left",
        ).pack(anchor="w", fill="x", pady=(2, 4))
        ttk.Button(controls, text="Ausgewählten Kratzer löschen", command=self.delete_selected).pack(
            fill="x", pady=2
        )
        ttk.Button(controls, text="Offene Ausnahme akzeptieren", command=self.accept_exception).pack(
            fill="x", pady=2
        )

        ttk.Separator(controls).pack(fill="x", pady=8)
        ttk.Label(controls, text="Slave-Funktionen", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w"
        )
        ttk.Button(controls, text="Slave auf Master zurücksetzen", command=self.reset_slave).pack(
            fill="x", pady=2
        )
        ttk.Button(controls, text="Slave-Maske vollständig leeren", command=self.clear_slave).pack(
            fill="x", pady=2
        )

        ttk.Separator(controls).pack(fill="x", pady=8)
        ttk.Label(controls, text="Leitlinien", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w"
        )
        self.zoom_status_label = tk.Label(
            controls,
            textvariable=self.zoom_status_var,
            anchor="w",
            padx=5,
            pady=4,
            background="#d7f4dd",
        )
        self.zoom_status_label.pack(fill="x", pady=2)
        ttk.Label(
            controls,
            text=f"Mindestlänge: {float(self.config['min_scratch_length_px']):g} Originalpixel",
        ).pack(anchor="w", pady=2)
        ttk.Label(
            controls,
            text=f"Max. Annotationszoom: {float(self.config['max_annotation_zoom']):g}×",
        ).pack(anchor="w", pady=2)

        ttk.Separator(controls).pack(fill="x", pady=8)
        self.unlock_button = ttk.Button(
            controls, text="Master entsperren", command=self.unlock_master
        )
        self.unlock_button.pack(fill="x", pady=2)
        ttk.Label(
            controls,
            text="Orange Markierungen müssen vor dem Abschluss explizit akzeptiert werden.",
            wraplength=255,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-s>", lambda _event: self.save_current())
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Control-y>", lambda _event: self.redo())
        self.root.bind("<Return>", lambda _event: self.finish_polyline())
        self.root.bind("<Escape>", lambda _event: self.cancel_current_action())
        self.root.bind("<Delete>", lambda _event: self.delete_selected())
        self.root.bind("<KeyPress-space>", self._hide_overlay)
        self.root.bind("<KeyRelease-space>", self._show_overlay)
        self.root.bind("n", lambda _event: self.mode_var.set("draw") or self._on_mode_changed())
        self.root.bind("e", lambda _event: self.mode_var.set("erase_rect") or self._on_mode_changed())
        self.root.bind("v", lambda _event: self.mode_var.set("select") or self._on_mode_changed())
        self.root.bind("a", lambda _event: self.navigate_image(-1))
        self.root.bind("d", lambda _event: self.navigate_image(1))

    def _initial_load(self) -> None:
        if self.project is not None:
            return
        try:
            self.reload_series()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc), parent=self.root)

    def choose_images_folder(self) -> None:
        folder = filedialog.askdirectory(
            title="Ordner mit Bildreihen auswählen",
            initialdir=str(self.images_dir if self.images_dir.exists() else self.annotation_dir),
            parent=self.root,
        )
        if not folder:
            return
        self.images_dir = Path(folder).resolve()
        try:
            self.reload_series()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc), parent=self.root)

    def reload_series(self) -> None:
        self.series_map = discover_series(self.images_dir)
        series_ids = list(self.series_map.keys())
        self.series_combo["values"] = series_ids
        desired = self.series_var.get() if self.series_var.get() in self.series_map else series_ids[0]
        self.series_var.set(desired)
        self.load_series(desired)

    def _on_series_selected(self, _event: tk.Event) -> None:
        self.load_series(self.series_var.get())

    def load_series(self, series_id: str) -> None:
        if not series_id:
            return
        self.save_project_only(silent=True)
        series = self.series_map[series_id]
        project_path = self.annotations_dir / f"{series.series_id}.json"
        self.project = load_project(project_path, series, self.config)
        self.current_series = series
        self._image_filenames = [series.master.name, *[path.name for path in series.slaves]]
        self.refresh_image_list()
        self.load_image(series.master.name)

    def _on_image_selected(self, _event: tk.Event) -> None:
        if self._listbox_update_active:
            return
        selection = self.image_list.curselection()
        if not selection:
            return
        index = int(selection[0])
        if 0 <= index < len(self._image_filenames):
            self.load_image(self._image_filenames[index])

    def load_image(self, filename: str) -> None:
        if self.current_series is None or self.project is None:
            return
        self.save_project_only(silent=True)
        self.cancel_current_action()
        path = self.image_path(filename)
        with Image.open(path) as loaded:
            image = loaded.convert("RGB")
        expected = tuple(map(int, self.project["image_size"]))
        if image.size != expected:
            raise ValueError(
                f"Bildauflösung von {filename} ist {image.width}x{image.height}, "
                f"erwartet wird {expected[0]}x{expected[1]}."
            )
        if filename != self.project["master_file"]:
            ensure_slave(self.project, filename)
        self.current_filename = filename
        self.current_image = image
        self.selected_stroke_id = None
        self.current_mask = None
        self.mask_dirty = True
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.root.after(60, self.fit_image)
        self.refresh_image_list(select_filename=filename)
        self.update_ui_state()
        self.render_all()

    def image_path(self, filename: str) -> Path:
        return self.images_dir / filename

    def project_path(self) -> Path:
        assert self.current_series is not None
        return self.annotations_dir / f"{self.current_series.series_id}.json"

    def mask_path(self, filename: str) -> Path:
        return self.masks_dir / f"{Path(filename).stem}.png"

    def is_master_image(self) -> bool:
        return bool(
            self.project
            and self.current_filename
            and self.current_filename == self.project["master_file"]
        )

    def editing_allowed(self) -> bool:
        if self.project is None or self.current_filename is None:
            return False
        return not (self.is_master_image() and bool(self.project["master"].get("locked", False)))

    def show_locked_master_message(self) -> None:
        messagebox.showinfo(
            APP_TITLE,
            "Der Master ist gesperrt, weil bereits ein Slave bearbeitet wurde. "
            "Bestehende Slave-Kopien bleiben bei einer Entsperrung unverändert.",
            parent=self.root,
        )

    def point_inside_image(self, x: float, y: float) -> bool:
        return bool(
            self.current_image
            and 0 <= x < self.current_image.width
            and 0 <= y < self.current_image.height
        )

    def clamp_point(self, x: float, y: float) -> tuple[float, float]:
        if self.current_image is None:
            return x, y
        return (
            min(max(x, 0.0), self.current_image.width - 1.0),
            min(max(y, 0.0), self.current_image.height - 1.0),
        )

    def fit_image(self) -> None:
        if self.current_image is None:
            return
        width = max(100, self.left_canvas.winfo_width())
        height = max(100, self.left_canvas.winfo_height())
        self.view.scale = min(width / self.current_image.width, height / self.current_image.height)
        self.view.center_x = self.current_image.width / 2
        self.view.center_y = self.current_image.height / 2
        self.update_zoom_status()
        self.render_all()

    def zoom_at(self, canvas: ImageCanvas, screen_x: float, screen_y: float, direction: int) -> None:
        if self.current_image is None:
            return
        before_x, before_y = canvas.screen_to_image(screen_x, screen_y)
        factor = 1.25 if direction > 0 else 1 / 1.25
        minimum = min(
            max(0.03, self.left_canvas.winfo_width() / max(1, self.current_image.width) * 0.25),
            1.0,
        )
        self.view.scale = min(20.0, max(minimum, self.view.scale * factor))
        after_x, after_y = canvas.screen_to_image(screen_x, screen_y)
        self.view.center_x += before_x - after_x
        self.view.center_y += before_y - after_y
        self.clamp_view()
        self.update_zoom_status()
        self.render_all()

    def clamp_view(self) -> None:
        if self.current_image is None:
            return
        margin_x = min(self.current_image.width / 2, 200 / max(self.view.scale, 1e-6))
        margin_y = min(self.current_image.height / 2, 200 / max(self.view.scale, 1e-6))
        self.view.center_x = min(
            max(self.view.center_x, -margin_x), self.current_image.width + margin_x
        )
        self.view.center_y = min(
            max(self.view.center_y, -margin_y), self.current_image.height + margin_y
        )

    def update_zoom_status(self) -> None:
        max_zoom = float(self.config["max_annotation_zoom"])
        if self.view.scale <= max_zoom:
            self.zoom_status_var.set(f"Zoomrichtlinie: ✓ {self.view.scale:.2f}× / {max_zoom:g}×")
            self.zoom_status_label.configure(background="#d7f4dd", foreground="#1b5e20")
        else:
            self.zoom_status_var.set(f"Zoomrichtlinie: ⚠ {self.view.scale:.2f}× / {max_zoom:g}×")
            self.zoom_status_label.configure(background="#ffe0b2", foreground="#8a4b00")

    def annotation_above_zoom_limit(self) -> bool:
        return self.view.scale > float(self.config["max_annotation_zoom"])

    def current_state(self) -> dict[str, Any] | None:
        if self.project is None or self.current_filename is None:
            return None
        if self.is_master_image():
            return self.project["master"]
        return ensure_slave(self.project, self.current_filename)

    def snapshot_current_state(self) -> Any:
        return deep_snapshot(self.current_state())

    def restore_current_state(self, snapshot: Any) -> None:
        if self.project is None or self.current_filename is None:
            return
        if self.is_master_image():
            self.project["master"] = deep_snapshot(snapshot)
        else:
            self.project["slaves"][self.current_filename] = deep_snapshot(snapshot)
        self.selected_stroke_id = None
        self.mark_dirty_and_save()

    def record_undo(self) -> None:
        snapshot = self.snapshot_current_state()
        if snapshot is not None:
            self.undo_stack.append(snapshot)
            if len(self.undo_stack) > 100:
                self.undo_stack.pop(0)
            self.redo_stack.clear()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        current = self.snapshot_current_state()
        snapshot = self.undo_stack.pop()
        if current is not None:
            self.redo_stack.append(current)
        self.restore_current_state(snapshot)

    def redo(self) -> None:
        if not self.redo_stack:
            return
        current = self.snapshot_current_state()
        snapshot = self.redo_stack.pop()
        if current is not None:
            self.undo_stack.append(current)
        self.restore_current_state(snapshot)

    def mark_current_in_progress(self) -> None:
        if self.project is None or self.current_filename is None:
            return
        if self.is_master_image():
            self.project["master"]["status"] = "in_progress"
        else:
            state = ensure_slave(self.project, self.current_filename)
            mark_slave_modified(self.project, state)

    def mark_dirty_and_save(self) -> None:
        self.mask_dirty = True
        self.current_mask = None
        self.save_project_only(silent=True)
        self.refresh_image_list(select_filename=self.current_filename)
        self.update_ui_state()
        self.render_all()

    def add_polyline_point(self, x: float, y: float) -> None:
        if self.mode_var.get() != "draw":
            return
        self.selected_stroke_id = None
        self.current_polyline.append(self.clamp_point(x, y))
        if self.annotation_above_zoom_limit():
            self.current_polyline_zoom_violation = True
        self.update_ui_state()
        self.render_all()

    def finish_polyline(self) -> None:
        if not self.current_polyline:
            return
        if len(self.current_polyline) < 2:
            messagebox.showwarning(
                APP_TITLE,
                "Ein Kratzer benötigt mindestens zwei miteinander verbundene Punkte.",
                parent=self.root,
            )
            return
        if not self.editing_allowed():
            return
        assert self.project is not None and self.current_filename is not None
        self.record_undo()
        stroke = new_stroke(
            self.current_polyline,
            int(self.width_var.get()),
            source="master" if self.is_master_image() else "slave_added",
            zoom_violation=self.current_polyline_zoom_violation,
        )
        if self.is_master_image():
            self.project["master"]["strokes"].append(stroke)
        else:
            state = ensure_slave(self.project, self.current_filename)
            state["added_strokes"].append(stroke)
            mark_slave_modified(self.project, state)
        self.mark_current_in_progress()
        self.current_polyline = []
        self.current_polyline_zoom_violation = False
        self.selected_stroke_id = stroke["id"]
        self.mark_dirty_and_save()

    def cancel_current_action(self) -> None:
        self.current_polyline = []
        self.current_polyline_zoom_violation = False
        self.preview_erase_rect = None
        self.render_all()

    def visible_strokes(self) -> list[tuple[str, dict[str, Any]]]:
        if self.project is None or self.current_filename is None:
            return []
        return current_strokes(self.project, self.current_filename)

    def get_selected_stroke(self) -> tuple[str, dict[str, Any]] | None:
        if self.project is None or self.current_filename is None or not self.selected_stroke_id:
            return None
        return find_stroke(self.project, self.current_filename, self.selected_stroke_id)

    @staticmethod
    def _distance_to_segment(
        px: float, py: float, ax: float, ay: float, bx: float, by: float
    ) -> float:
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def select_nearest_stroke(self, x: float, y: float) -> None:
        tolerance = max(3.0, 12.0 / max(self.view.scale, 1e-6))
        best: tuple[float, str] | None = None
        for _, stroke in self.visible_strokes():
            points = stroke.get("points", [])
            distance = min(
                (
                    self._distance_to_segment(
                        x,
                        y,
                        float(a[0]),
                        float(a[1]),
                        float(b[0]),
                        float(b[1]),
                    )
                    for a, b in zip(points, points[1:])
                ),
                default=float("inf"),
            )
            hit_distance = distance - max(0.0, float(stroke.get("width_px", 1)) / 2)
            if hit_distance <= tolerance and (best is None or hit_distance < best[0]):
                best = (hit_distance, str(stroke.get("id")))
        self.selected_stroke_id = best[1] if best else None
        selected = self.get_selected_stroke()
        if selected is not None:
            self._suppress_width_update = True
            self.width_var.set(int(selected[1].get("width_px", self.width_var.get())))
            self._suppress_width_update = False
        self.update_ui_state()
        self.render_all()

    def nearest_selected_point(
        self, x: float, y: float, *, tolerance_screen_px: float
    ) -> int | None:
        selected = self.get_selected_stroke()
        if selected is None:
            return None
        tolerance = tolerance_screen_px / max(self.view.scale, 1e-6)
        best: tuple[float, int] | None = None
        for index, point in enumerate(selected[1].get("points", [])):
            distance = math.hypot(x - float(point[0]), y - float(point[1]))
            if distance <= tolerance and (best is None or distance < best[0]):
                best = (distance, index)
        return best[1] if best else None

    def begin_geometry_edit(self) -> None:
        if not self._geometry_snapshot_active:
            self.record_undo()
            self._geometry_snapshot_active = True

    def move_selected_point(self, point_index: int, x: float, y: float) -> None:
        selected = self.get_selected_stroke()
        if selected is None:
            return
        _, stroke = selected
        if not (0 <= point_index < len(stroke.get("points", []))):
            return
        stroke["points"][point_index] = [round(x, 3), round(y, 3)]
        stroke["accepted_short"] = False
        if self.annotation_above_zoom_limit():
            stroke["zoom_violation"] = True
            stroke["accepted_zoom"] = False
        self.mark_current_in_progress()
        if not self.is_master_image() and self.project and self.current_filename:
            mark_slave_modified(self.project, ensure_slave(self.project, self.current_filename))
        self.mask_dirty = True
        self.current_mask = None
        self.update_ui_state()
        self.render_all()

    def end_geometry_edit(self) -> None:
        if not self._geometry_snapshot_active:
            return
        self._geometry_snapshot_active = False
        self.mark_dirty_and_save()

    def apply_width_to_selected(self) -> None:
        selected = self.get_selected_stroke()
        if selected is None:
            messagebox.showinfo(APP_TITLE, "Zuerst einen Kratzer auswählen.", parent=self.root)
            return
        if not self.editing_allowed():
            self.show_locked_master_message()
            return
        self.record_undo()
        _, stroke = selected
        stroke["width_px"] = int(self.width_var.get())
        if self.annotation_above_zoom_limit():
            stroke["zoom_violation"] = True
            stroke["accepted_zoom"] = False
        self.mark_current_in_progress()
        if not self.is_master_image() and self.project and self.current_filename:
            mark_slave_modified(self.project, ensure_slave(self.project, self.current_filename))
        self.mark_dirty_and_save()

    def delete_selected(self) -> None:
        if self.project is None or self.current_filename is None or not self.selected_stroke_id:
            return
        if not self.editing_allowed():
            self.show_locked_master_message()
            return
        self.record_undo()
        removed = remove_stroke(self.project, self.current_filename, self.selected_stroke_id)
        if not removed:
            self.undo_stack.pop()
            return
        if self.annotation_above_zoom_limit():
            if self.is_master_image():
                self.project["master"]["zoom_edit_violation"] = True
                self.project["master"]["accepted_zoom_edit"] = False
            else:
                state = ensure_slave(self.project, self.current_filename)
                state["zoom_erase_violation"] = True
                state["accepted_zoom_erase"] = False
        self.mark_current_in_progress()
        self.selected_stroke_id = None
        self.mark_dirty_and_save()

    def apply_erase_rectangle(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if self.project is None or self.current_filename is None or self.is_master_image():
            return
        self.record_undo()
        state = ensure_slave(self.project, self.current_filename)
        state["erase_rects"].append(
            [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]
        )
        if self.annotation_above_zoom_limit():
            state["zoom_erase_violation"] = True
            state["accepted_zoom_erase"] = False
        mark_slave_modified(self.project, state)
        self.mark_dirty_and_save()

    def clear_slave(self) -> None:
        if self.project is None or self.current_filename is None or self.is_master_image():
            messagebox.showinfo(
                APP_TITLE, "Diese Funktion steht nur für Slave-Bilder zur Verfügung.", parent=self.root
            )
            return
        if not messagebox.askyesno(
            APP_TITLE,
            "Die komplette aktuelle Slave-Maske leeren? Der Master bleibt unverändert.",
            parent=self.root,
        ):
            return
        self.record_undo()
        state = ensure_slave(self.project, self.current_filename)
        state["clear_base"] = True
        state["hidden_base_ids"] = []
        state["added_strokes"] = []
        state["erase_rects"] = []
        if self.annotation_above_zoom_limit():
            state["zoom_erase_violation"] = True
            state["accepted_zoom_erase"] = False
        mark_slave_modified(self.project, state)
        self.selected_stroke_id = None
        self.mark_dirty_and_save()

    def reset_slave(self) -> None:
        if self.project is None or self.current_filename is None or self.is_master_image():
            messagebox.showinfo(
                APP_TITLE, "Diese Funktion steht nur für Slave-Bilder zur Verfügung.", parent=self.root
            )
            return
        if not messagebox.askyesno(
            APP_TITLE,
            "Alle Slave-Änderungen verwerfen und erneut vom aktuellen Master starten?",
            parent=self.root,
        ):
            return
        self.record_undo()
        state = ensure_slave(self.project, self.current_filename)
        state.clear()
        state.update(
            {
                "base_strokes": copy.deepcopy(self.project["master"]["strokes"]),
                "added_strokes": [],
                "hidden_base_ids": [],
                "erase_rects": [],
                "clear_base": False,
                "zoom_erase_violation": False,
                "accepted_zoom_erase": False,
                "status": "in_progress",
                "modified": True,
            }
        )
        self.project["master"]["locked"] = True
        if self.annotation_above_zoom_limit():
            state["zoom_erase_violation"] = True
            state["accepted_zoom_erase"] = False
        self.selected_stroke_id = None
        self.mark_dirty_and_save()

    def accept_exception(self) -> None:
        if self.project is None or self.current_filename is None:
            return
        selected = self.get_selected_stroke()
        if selected is not None:
            _, stroke = selected
            warnings = stroke_warnings(stroke, self.config)
            if not warnings:
                messagebox.showinfo(
                    APP_TITLE, "Der ausgewählte Kratzer hat keine offene Ausnahme.", parent=self.root
                )
                return
            self.record_undo()
            if "short" in warnings:
                stroke["accepted_short"] = True
            if "zoom" in warnings:
                stroke["accepted_zoom"] = True
            self.mark_current_in_progress()
            if not self.is_master_image():
                mark_slave_modified(self.project, ensure_slave(self.project, self.current_filename))
            self.mark_dirty_and_save()
            return

        if self.is_master_image():
            master = self.project["master"]
            if master.get("zoom_edit_violation") and not master.get("accepted_zoom_edit"):
                self.record_undo()
                master["accepted_zoom_edit"] = True
                master["status"] = "in_progress"
                self.mark_dirty_and_save()
                return
        else:
            state = ensure_slave(self.project, self.current_filename)
            if state.get("zoom_erase_violation") and not state.get("accepted_zoom_erase"):
                self.record_undo()
                state["accepted_zoom_erase"] = True
                mark_slave_modified(self.project, state)
                self.mark_dirty_and_save()
                return
        messagebox.showinfo(
            APP_TITLE,
            "Zuerst einen orange markierten Kratzer auswählen. Bei einer Bildwarnung darf keine Auswahl aktiv sein.",
            parent=self.root,
        )

    def unlock_master(self) -> None:
        if self.project is None:
            return
        if not self.project["master"].get("locked", False):
            messagebox.showinfo(APP_TITLE, "Der Master ist bereits entsperrt.", parent=self.root)
            return
        answer = messagebox.askyesno(
            APP_TITLE,
            "Master entsperren? Bereits bearbeitete Slave-Kopien bleiben unverändert. "
            "Nur noch unbearbeitete Slaves übernehmen spätere Masteränderungen.",
            parent=self.root,
        )
        if not answer:
            return
        self.project["master"]["locked"] = False
        self.save_project_only(silent=True)
        self.update_ui_state()

    def get_mask(self) -> np.ndarray | None:
        if self.project is None or self.current_filename is None:
            return None
        if self.mask_dirty or self.current_mask is None:
            self.current_mask = render_mask(self.project, self.current_filename)
            self.mask_dirty = False
        return self.current_mask

    def create_overlay_crop(self, crop_box: tuple[int, int, int, int]) -> Image.Image:
        mask = self.get_mask()
        if mask is None:
            width = max(1, crop_box[2] - crop_box[0])
            height = max(1, crop_box[3] - crop_box[1])
            return Image.new("RGBA", (width, height), (0, 0, 0, 0))
        left, top, right, bottom = crop_box
        mask_crop = mask[top:bottom, left:right]
        rgb = tuple(int(value) for value in self.config.get("overlay_rgb", [255, 40, 40]))
        alpha = np.where(
            mask_crop > 0,
            int(max(0.0, min(1.0, float(self.opacity_var.get()))) * 255),
            0,
        ).astype(np.uint8)
        rgba = np.zeros((mask_crop.shape[0], mask_crop.shape[1], 4), dtype=np.uint8)
        rgba[..., 0] = rgb[0]
        rgba[..., 1] = rgb[1]
        rgba[..., 2] = rgb[2]
        rgba[..., 3] = alpha
        return Image.fromarray(rgba, mode="RGBA")

    def save_project_only(self, *, silent: bool = False) -> bool:
        if self.project is None or self.current_series is None:
            return False
        try:
            save_project_atomic(self.project_path(), self.project)
            return True
        except Exception as exc:
            if not silent:
                messagebox.showerror(APP_TITLE, f"Annotation konnte nicht gespeichert werden:\n{exc}", parent=self.root)
            return False

    def save_current(self) -> bool:
        if self.project is None or self.current_filename is None or self.current_series is None:
            return False
        try:
            self.save_project_only(silent=False)
            mask = self.get_mask()
            assert mask is not None
            write_mask(
                self.mask_path(self.current_filename),
                mask,
                (self.current_series.width, self.current_series.height),
            )
            self.image_status_var.set(
                f"Gespeichert: {self.current_filename} → {self.mask_path(self.current_filename).name}"
            )
            return True
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Speichern fehlgeschlagen:\n{exc}", parent=self.root)
            return False

    def finish_current_image(self) -> None:
        if self.project is None or self.current_filename is None:
            return
        if self.current_polyline:
            messagebox.showwarning(
                APP_TITLE,
                "Die aktuelle Polyline zuerst mit Enter/Rechtsklick abschließen oder mit Esc abbrechen.",
                parent=self.root,
            )
            return
        can_finish, final_status, warnings = completion_status(
            self.project, self.current_filename, self.config
        )
        if not can_finish:
            short_count = sum(warning.endswith(":short") for warning in warnings)
            zoom_count = sum(
                warning.endswith(":zoom")
                or warning in {"image:zoom_erase", "image:zoom_edit"}
                for warning in warnings
            )
            messagebox.showwarning(
                APP_TITLE,
                "Das Bild kann noch nicht abgeschlossen werden.\n\n"
                f"Offene Mindestlängen-Ausnahmen: {short_count}\n"
                f"Offene Zoom-Ausnahmen: {zoom_count}\n\n"
                "Orange Kratzer auswählen und 'Offene Ausnahme akzeptieren' verwenden. "
                "Für eine orange Bildwarnung die Auswahl aufheben und die Ausnahme akzeptieren.",
                parent=self.root,
            )
            return
        if self.is_master_image():
            self.project["master"]["status"] = final_status
        else:
            state = ensure_slave(self.project, self.current_filename)
            state["status"] = final_status
            state["modified"] = True
            self.project["master"]["locked"] = True
        if self.save_current():
            self.refresh_image_list(select_filename=self.current_filename)
            self.update_ui_state()

    def refresh_image_list(self, select_filename: str | None = None) -> None:
        if self.project is None:
            return
        selected = select_filename or self.current_filename
        self._listbox_update_active = True
        self.image_list.delete(0, tk.END)
        selected_index = 0
        for index, filename in enumerate(self._image_filenames):
            status = self.image_status(filename)
            role = "MASTER" if filename == self.project["master_file"] else "SLAVE "
            display = f"{STATUS_SYMBOLS.get(status, '○')} {role}  {filename}"
            self.image_list.insert(tk.END, display)
            if filename == selected:
                selected_index = index
        if self._image_filenames:
            self.image_list.selection_clear(0, tk.END)
            self.image_list.selection_set(selected_index)
            self.image_list.see(selected_index)
        self._listbox_update_active = False

    def image_status(self, filename: str) -> str:
        if self.project is None:
            return "not_started"
        if filename == self.project["master_file"]:
            return str(self.project["master"].get("status", "not_started"))
        state = self.project.get("slaves", {}).get(filename)
        return str(state.get("status", "not_started")) if state else "not_started"

    def update_ui_state(self) -> None:
        if self.project is None or self.current_filename is None:
            self.selection_info_var.set("Kein Kratzer ausgewählt")
            self.image_status_var.set("Kein Bild geladen")
            return
        selected = self.get_selected_stroke()
        if selected is None:
            self.selection_info_var.set("Kein Kratzer ausgewählt")
        else:
            kind, stroke = selected
            length = stroke_length_px(stroke)
            warnings = stroke_warnings(stroke, self.config)
            warning_text = ", ".join(warnings) if warnings else "keine"
            self.selection_info_var.set(
                f"Typ: {kind}\nLänge: {length:.1f} px\nBreite: {int(stroke.get('width_px', 1))} px\n"
                f"Offene Warnungen: {warning_text}"
            )
        status = self.image_status(self.current_filename)
        open_warnings = image_open_warnings(self.project, self.current_filename, self.config)
        self.image_status_var.set(
            f"{self.current_filename} · {STATUS_LABELS.get(status, status)} · "
            f"{len(self.visible_strokes())} sichtbare Kratzer · {len(open_warnings)} offene Warnungen"
        )
        locked = bool(self.project["master"].get("locked", False))
        self.master_lock_var.set("🔒 Master gesperrt" if locked else "🔓 Master entsperrt")
        self.unlock_button.configure(state="normal" if locked else "disabled")
        self.update_zoom_status()

    def render_all(self) -> None:
        self.left_canvas.render()
        self.right_canvas.render()

    def navigate_image(self, offset: int) -> None:
        if not self.current_filename or self.current_filename not in self._image_filenames:
            return
        index = self._image_filenames.index(self.current_filename)
        new_index = max(0, min(len(self._image_filenames) - 1, index + offset))
        if new_index != index:
            self.load_image(self._image_filenames[new_index])

    def _on_mode_changed(self) -> None:
        if self.mode_var.get() != "draw" and self.current_polyline:
            if messagebox.askyesno(
                APP_TITLE,
                "Die noch nicht abgeschlossene Polyline verwerfen?",
                parent=self.root,
            ):
                self.cancel_current_action()
            else:
                self.mode_var.set("draw")
        self.left_canvas.configure(
            cursor="fleur" if self.mode_var.get() == "pan" else "crosshair"
        )

    def _hide_overlay(self, _event: tk.Event) -> str:
        self.overlay_hidden = True
        self.render_all()
        return "break"

    def _show_overlay(self, _event: tk.Event) -> str:
        self.overlay_hidden = False
        self.render_all()
        return "break"

    def on_close(self) -> None:
        self.save_project_only(silent=True)
        self.root.destroy()


def main() -> None:
    annotation_dir = Path(__file__).resolve().parent
    root = tk.Tk()
    try:
        ScratchAnnotatorApp(root, annotation_dir)
        root.mainloop()
    except Exception as exc:  # pragma: no cover - top-level GUI safety net
        traceback.print_exc()
        try:
            messagebox.showerror(APP_TITLE, f"Unerwarteter Fehler:\n{exc}", parent=root)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()

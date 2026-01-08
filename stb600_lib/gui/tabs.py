"""
GUI Tabs Module - Tab Classes for Pipeline Stages
=================================================

Provides individual tab classes for each processing step in the pipeline.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog

from .base import BaseTab
from ..io import load_source_cv2
from ..color import detect_piece_color_and_check_size
from ..contours import crop_and_align_vertical, extract_contour_features
from ..display import show_image
from ..transforms import rotate_if_marker_bottom, rotate_if_blue_piece_upside_down


class InputTab(BaseTab):
    """Tab for loading images and toggling camera."""

    def __init__(self, parent, app):
        super().__init__(parent, app, title="Original image")

        btn_load = ttk.Button(self.controls, text="Load image", command=self.load_image)
        btn_load.grid(row=0, column=0, sticky="we", pady=(0, 5))

        self.camera_enabled = tk.BooleanVar(value=False)
        btn_cam = ttk.Button(
            self.controls, text="Toggle camera", command=self.toggle_camera
        )
        btn_cam.grid(row=1, column=0, sticky="we")

        ttk.Label(self.controls, text="Piece color").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.color_var = tk.StringVar(value=self.app.piece_color)
        color_menu = ttk.OptionMenu(
            self.controls, self.color_var, self.app.piece_color,
            "blue", "yellow", "red", command=self.on_color_change
        )
        color_menu.grid(row=3, column=0, sticky="we")

    def on_color_change(self, value):
        """Handle piece color selection change - load ROI config for new color."""
        self.app.piece_color = value
        self.app.SECOND_COLOR = value
        # Only ROIs differ per color, shared settings stay the same
        self.app.apply_rois_for_color(value)
        self.app.initial_update()

    def load_image(self):
        """Open file dialog and load selected image."""
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            print("Could not read image:", path)
            return

        self.camera_enabled.set(False)
        if self.app.cap is not None:
            self.app.cap.release()
            self.app.cap = None

        self.app.set_base_image(img)
        self.update_image()

    def toggle_camera(self):
        """Toggle camera on/off."""
        new_state = not self.camera_enabled.get()
        self.camera_enabled.set(new_state)
        self.app.set_camera_enabled(new_state)
        print("Camera enabled:", new_state)

    def update_image(self):
        """Update displayed image."""
        frame = self.app.get_base_image()
        if frame is not None:
            self.set_image(frame)


class RemoveGreenTab(BaseTab):
    """Tab for green background removal with adjustable HSV thresholds."""

    def __init__(self, parent, app):
        super().__init__(parent, app, title="1) Remove green (HSV)")
        self.h_low = tk.IntVar(value=35)
        self.h_high = tk.IntVar(value=85)

        ttk.Label(self.controls, text="H low").grid(row=0, column=0, sticky="w")
        tk.Scale(self.controls, from_=0, to=179, orient="horizontal", length=180,
                 variable=self.h_low, command=lambda v: self.update_image()
                 ).grid(row=1, column=0, sticky="we")

        ttk.Label(self.controls, text="H high").grid(row=2, column=0, sticky="w")
        tk.Scale(self.controls, from_=0, to=179, orient="horizontal", length=180,
                 variable=self.h_high, command=lambda v: self.update_image()
                 ).grid(row=3, column=0, sticky="we")

    def update_image(self):
        """Apply green removal and display result."""
        img = self.app.get_base_image()
        if img is None:
            return

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_low.get(), 40, 40])
        upper = np.array([self.h_high.get(), 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        no_green = img.copy()
        no_green[mask > 0] = 0

        self.set_image(no_green)
        self.app.no_green = no_green
        self.app.green_mask = mask

        # Save config
        self.app.save_config_section("remove_green", {
            "h_low": self.h_low.get(),
            "h_high": self.h_high.get(),
        })


class BinarizeTab(BaseTab):
    """Tab for binarization with adjustable threshold."""

    def __init__(self, parent, app):
        super().__init__(parent, app, title="2) Binarize")
        self.thresh_value = tk.IntVar(value=127)

        ttk.Label(self.controls, text="Threshold").grid(row=0, column=0, sticky="w")
        tk.Scale(self.controls, from_=0, to=255, orient="horizontal", length=180,
                 variable=self.thresh_value, command=lambda v: self.update_image()
                 ).grid(row=1, column=0, sticky="we")

    def update_image(self):
        """Apply binarization and display result."""
        base = self.app.get_base_image()
        no_green = self.app.no_green
        if base is None or no_green is None:
            return

        gray = cv2.cvtColor(no_green, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.thresh_value.get(), 255, cv2.THRESH_BINARY)

        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.set_image(binary_bgr)
        self.app.gray = gray
        self.app.binary = binary

        # Save config
        self.app.save_config_section("binarize", {
            "thresh_value": self.thresh_value.get(),
        })


class OpeningTab(BaseTab):
    """Tab for morphological opening with adjustable kernel and iterations."""

    def __init__(self, parent, app):
        super().__init__(parent, app, title="3) Morphological opening")

        self.kernel_size = tk.IntVar(value=5)
        self.iterations = tk.IntVar(value=1)

        ttk.Label(self.controls, text="Kernel size").grid(row=0, column=0, sticky="w")
        tk.Scale(self.controls, from_=1, to=15, orient="horizontal", length=180,
                 variable=self.kernel_size, resolution=2,
                 command=lambda v: self.update_image()
                 ).grid(row=1, column=0, sticky="we")

        ttk.Label(self.controls, text="Iterations").grid(row=2, column=0, sticky="w")
        tk.Scale(self.controls, from_=1, to=5, orient="horizontal", length=180,
                 variable=self.iterations, command=lambda v: self.update_image()
                 ).grid(row=3, column=0, sticky="we")

    def update_image(self):
        """Apply morphological opening and display result."""
        binary = self.app.binary
        if binary is None:
            return

        k = self.kernel_size.get()
        if k % 2 == 0:
            k += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=self.iterations.get())

        opened_bgr = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
        self.set_image(opened_bgr)
        self.app.gray_closed = opened

        # Save config
        self.app.save_config_section("opening", {
            "kernel_size": self.kernel_size.get(),
            "iterations": self.iterations.get(),
        })


class BigContoursTab(BaseTab):
    """Tab for detecting and filtering large contours by area."""

    def __init__(self, parent, app):
        super().__init__(parent, app, title="4) Large contours")

        self.min_area = tk.IntVar(value=10000)
        self.max_area = tk.IntVar(value=200000)
        self.small_max = tk.IntVar(value=30000)
        self.medium_max = tk.IntVar(value=90000)

        ttk.Label(self.controls, text="Min area").grid(row=0, column=0, sticky="w")
        tk.Scale(self.controls, from_=0, to=200000, orient="horizontal", length=180,
                 variable=self.min_area, resolution=1000,
                 command=lambda v: self.update_image()
                 ).grid(row=1, column=0, sticky="we")

        ttk.Label(self.controls, text="Max area").grid(row=2, column=0, sticky="w")
        tk.Scale(self.controls, from_=0, to=500000, orient="horizontal", length=180,
                 variable=self.max_area, resolution=1000,
                 command=lambda v: self.update_image()
                 ).grid(row=3, column=0, sticky="we")

        ttk.Label(self.controls, text="Small max").grid(row=4, column=0, sticky="w")
        tk.Scale(self.controls, from_=0, to=200000, orient="horizontal", length=180,
                 variable=self.small_max, resolution=1000,
                 command=lambda v: self.update_image()
                 ).grid(row=5, column=0, sticky="we")

        ttk.Label(self.controls, text="Medium max").grid(row=6, column=0, sticky="w")
        tk.Scale(self.controls, from_=0, to=400000, orient="horizontal", length=180,
                 variable=self.medium_max, resolution=1000,
                 command=lambda v: self.update_image()
                 ).grid(row=7, column=0, sticky="we")

    def update_image(self):
        """Find contours, filter by area, and display with labels."""
        img = self.app.get_base_image()
        gray_closed = self.app.gray_closed
        if img is None or gray_closed is None:
            return

        contours, _ = cv2.findContours(gray_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_a = self.min_area.get()
        max_a = self.max_area.get()
        small_max = self.small_max.get()
        medium_max = self.medium_max.get()

        filtered = []
        labeled_info = []

        for c in contours:
            area = cv2.contourArea(c)
            if not (min_a <= area <= max_a):
                continue
            filtered.append(c)

            if area <= small_max:
                label = "small"
                color = (0, 255, 0)
            elif area <= medium_max:
                label = "medium"
                color = (255, 0, 0)
            else:
                label = "large"
                color = (0, 255, 255)

            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            labeled_info.append((c, label, color, (cx, cy), area))

        main_cnt = None
        main_label = None
        if filtered:
            main_cnt = max(filtered, key=cv2.contourArea)

        out = img.copy()
        for c, label, color, (cx, cy), area in labeled_info:
            label_to_draw = label
            draw_color = color

            if main_cnt is not None and np.array_equal(c, main_cnt):
                main_label = label
                if label == "small":
                    expected_color_from_size = "red"
                elif label == "medium":
                    expected_color_from_size = "yellow"
                else:
                    expected_color_from_size = "blue"

                expected_piece_color = self.app.piece_color
                if expected_piece_color != expected_color_from_size:
                    label_to_draw = "fake"
                    draw_color = (0, 0, 0)

            cv2.drawContours(out, [c], -1, draw_color, 3)
            cv2.putText(out, label_to_draw, (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2, cv2.LINE_AA)

        self.app.big_contours = filtered
        self.app.all_contours = contours
        self.app.contour_img = out
        self.app.main_cnt = main_cnt
        self.app.main_size_label = main_label if main_label is not None else "unknown"

        self.set_image(out)

        # Save config
        self.app.save_config_section("big_contours", {
            "min_area": self.min_area.get(),
            "max_area": self.max_area.get(),
            "small_max": self.small_max.get(),
            "medium_max": self.medium_max.get(),
        })


class CroppedPartTab(BaseTab):
    """Tab for cropping and aligning the main detected part."""

    def __init__(self, parent, app):
        super().__init__(parent, app, title="4b) Cropped main part")

    def update_image(self):
        """Crop main contour and detect piece color."""
        img = self.app.get_base_image()
        contours = self.app.big_contours

        if img is None:
            return

        if not contours:
            contours = getattr(self.app, "all_contours", [])
            if not contours:
                return

        main_cnt = max(contours, key=cv2.contourArea)
        cropped_aligned = crop_and_align_vertical(img, main_cnt)

        if cropped_aligned is None or cropped_aligned.size == 0:
            x, y, w, h = cv2.boundingRect(main_cnt)
            H, W = img.shape[:2]
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))
            cropped_aligned = img[y:y + h, x:x + w]

        try:
            piece_color, size_label, is_consistent, color_counts = detect_piece_color_and_check_size(
                img_bgr=img, contour=main_cnt,
                small_area_max=90_000, medium_area_max=130_000,
                color_area_fraction_thresh=0
            )

            self.app.piece_color = piece_color
            self.app.main_size_label = size_label

            if is_consistent:
                self.app.SECOND_COLOR = piece_color

        except ValueError as e:
            print(f"[WARN] Could not determine piece color: {e}")

        current_part = cropped_aligned
        self.set_image(current_part)

        self.app.main_cnt = main_cnt
        self.app.cropped_aligned = current_part
        self.app.current_part = current_part


class InnerColorPartsTab(BaseTab):
    """Tab for detecting internal colored parts within the main piece."""

    def __init__(self, parent, app):
        super().__init__(parent, app, title="5) Internal parts by color")

        self.min_area = tk.IntVar(value=50)
        self.max_area = tk.IntVar(value=10000)

        ttk.Label(self.controls, text="Min area").grid(row=0, column=0, sticky="w")
        tk.Scale(self.controls, from_=0, to=5000, resolution=10, orient="horizontal",
                 length=180, variable=self.min_area,
                 command=lambda v: self.update_image()
                 ).grid(row=1, column=0, sticky="we")

        ttk.Label(self.controls, text="Max area").grid(row=2, column=0, sticky="w")
        tk.Scale(self.controls, from_=0, to=50000, resolution=50, orient="horizontal",
                 length=180, variable=self.max_area,
                 command=lambda v: self.update_image()
                 ).grid(row=3, column=0, sticky="we")

    def update_image(self):
        """Detect inner color parts and display contours."""
        current_part = self.app.current_part
        if current_part is None:
            return

        second_color = self.app.SECOND_COLOR

        hsv = cv2.cvtColor(current_part, cv2.COLOR_BGR2HSV)

        if second_color == "blue":
            lower = np.array([90, 50, 50])
            upper = np.array([130, 255, 255])
        elif second_color == "red":
            lower = np.array([0, 50, 50])
            upper = np.array([10, 255, 255])
        else:
            lower = np.array([20, 50, 50])
            upper = np.array([35, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_a = self.min_area.get()
        max_a = self.max_area.get()

        part_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_a <= area <= max_a:
                part_contours.append(c)

        out = current_part.copy()
        cv2.drawContours(out, part_contours, -1, (0, 255, 0), 3)
        self.set_image(out)
        self.app.part_contours = part_contours
        self.app.part_contour_img = out
        self.app.mask_closed = mask_closed

        # Save config
        self.app.save_config_section("inner_parts", {
            "min_area": self.min_area.get(),
            "max_area": self.max_area.get(),
        })


class LabeledPartsTab(BaseTab):
    """Tab for labeling detected parts as marker/small/medium/large."""

    def __init__(self, parent, app):
        super().__init__(parent, app, title="6) Labeled small/medium/large")

        self.extent_threshold = tk.DoubleVar(value=0.5)
        self.small_area_max = tk.IntVar(value=1000)
        self.medium_area_max = tk.IntVar(value=2100)

        ttk.Label(self.controls, text="Extent threshold").grid(row=0, column=0, sticky="w")
        tk.Scale(self.controls, from_=0, to=1, resolution=0.05, orient="horizontal",
                 length=180, variable=self.extent_threshold,
                 command=lambda v: self.update_image()
                 ).grid(row=1, column=0, sticky="we")

        ttk.Label(self.controls, text="Small area max").grid(row=2, column=0, sticky="w")
        tk.Scale(self.controls, from_=0, to=5000, resolution=50, orient="horizontal",
                 length=180, variable=self.small_area_max,
                 command=lambda v: self.update_image()
                 ).grid(row=3, column=0, sticky="we")

        ttk.Label(self.controls, text="Medium area max").grid(row=4, column=0, sticky="w")
        tk.Scale(self.controls, from_=0, to=10000, resolution=50, orient="horizontal",
                 length=180, variable=self.medium_area_max,
                 command=lambda v: self.update_image()
                 ).grid(row=5, column=0, sticky="we")

        legend = "Legend:\nmarker: red\nsmall: green\nmedium: blue\nlarge: yellow"
        self.legend_box = tk.Text(self.controls, height=6, width=28)
        self.legend_box.grid(row=6, column=0, sticky="we", pady=(10, 0))
        self.legend_box.insert(tk.END, legend)
        self.legend_box.config(state="disabled")

    def _label_contours(self, part_contours):
        """Label contours by extent and area, return features list and kept contours."""
        features_list = []
        kept_contours = []

        extent_threshold = self.extent_threshold.get()
        small_area_max = self.small_area_max.get()
        medium_area_max = self.medium_area_max.get()

        for idx, c in enumerate(part_contours):
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            x, y, w, h = cv2.boundingRect(c)
            rect_area = w * h
            extent = area / float(rect_area) if rect_area != 0 else 0

            f = {
                "index": idx,
                "area": area,
                "perimeter": peri,
                "centroid": (cx, cy),
                "bounding_rect": (x, y, w, h),
                "rect_area": rect_area,
                "extent": extent,
            }

            if extent < extent_threshold:
                size_label = "marker"
            elif area <= small_area_max:
                size_label = "small"
            elif area <= medium_area_max:
                size_label = "medium"
            else:
                size_label = "large"

            f["label"] = size_label
            features_list.append(f)
            kept_contours.append(c)

        return features_list, kept_contours

    def _detect_inner_contours(self, img):
        """Detect inner color contours in an image."""
        second_color = self.app.SECOND_COLOR
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if second_color == "blue":
            lower = np.array([90, 50, 50])
            upper = np.array([130, 255, 255])
        elif second_color == "red":
            lower = np.array([0, 50, 50])
            upper = np.array([10, 255, 255])
        else:
            lower = np.array([20, 50, 50])
            upper = np.array([35, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area (use InnerColorPartsTab settings if available)
        min_a = getattr(self.app.tab_inner, 'min_area', None)
        max_a = getattr(self.app.tab_inner, 'max_area', None)
        min_a = min_a.get() if min_a else 50
        max_a = max_a.get() if max_a else 10000

        part_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_a <= area <= max_a:
                part_contours.append(c)

        return part_contours

    def _draw_labeled_contours(self, img, features_list, kept_contours):
        """Draw labeled contours on image and return result."""
        colored_part = img.copy()
        for f, c in zip(features_list, kept_contours):
            label = f["label"]
            cx, cy = f["centroid"]

            color = (255, 255, 255)
            if label == "marker":
                color = (0, 0, 255)
            elif label == "small":
                color = (0, 255, 0)
            elif label == "medium":
                color = (255, 0, 0)
            elif label == "large":
                color = (0, 255, 255)

            cv2.drawContours(colored_part, [c], -1, color, 3)
            cv2.putText(colored_part, label, (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return colored_part

    def update_image(self):
        """Label contours, check orientation, rotate if needed, and re-label."""
        current_part = self.app.current_part
        part_contours = self.app.part_contours
        if current_part is None or part_contours is None:
            return

        # Initial labeling
        features_list, kept_contours = self._label_contours(part_contours)

        # Check if rotation is needed
        rotated = False

        # 1) Rotate if marker is at bottom
        rotated_part, did_rotate = rotate_if_marker_bottom(
            current_part, features_list, bottom_fraction=0.33
        )
        if did_rotate:
            rotated = True
            current_part = rotated_part

        # 2) For blue pieces, check if upside down (only if not already rotated)
        if not rotated and self.app.piece_color == "blue":
            h = current_part.shape[0]
            rotated_part, did_rotate = rotate_if_blue_piece_upside_down(
                current_part, features_list, main_contour_height=h, bottom_fraction=0.33
            )
            if did_rotate:
                rotated = True
                current_part = rotated_part

        # If rotated, re-detect and re-label contours
        if rotated:
            print("Image rotated 180° - re-detecting contours...")
            # Update current_part in app
            self.app.current_part = current_part

            # Re-detect inner contours on rotated image
            part_contours = self._detect_inner_contours(current_part)
            self.app.part_contours = part_contours

            # Re-label
            features_list, kept_contours = self._label_contours(part_contours)

        # Draw labeled contours
        colored_part = self._draw_labeled_contours(current_part, features_list, kept_contours)

        self.set_image(colored_part)
        self.app.features_list = features_list
        self.app.part_contours = kept_contours
        self.app.colored_part = colored_part

        # Save config
        self.app.save_config_section("labeled", {
            "extent_threshold": self.extent_threshold.get(),
            "small_area_max": self.small_area_max.get(),
            "medium_area_max": self.medium_area_max.get(),
        })


class ROIsTab(BaseTab):
    """Tab for defining ROIs and counting labeled parts per ROI."""

    def __init__(self, parent, app):
        super().__init__(parent, app, title="7) ROIs and counting")

        self.num_rois = tk.IntVar(value=3)
        self.offset_pct = tk.IntVar(value=0)
        self.height_pct = tk.IntVar(value=100)

        ttk.Label(self.controls, text="Num ROIs").grid(row=0, column=0, sticky="w")
        tk.Scale(self.controls, from_=1, to=6, resolution=1, orient="horizontal",
                 length=180, variable=self.num_rois,
                 command=lambda v: self.update_image()
                 ).grid(row=1, column=0, sticky="we")

        ttk.Label(self.controls, text="Offset % (from top)").grid(row=2, column=0, sticky="w")
        tk.Scale(self.controls, from_=0, to=80, resolution=5, orient="horizontal",
                 length=180, variable=self.offset_pct,
                 command=lambda v: self.update_image()
                 ).grid(row=3, column=0, sticky="we")

        ttk.Label(self.controls, text="Height % (ROI region)").grid(row=4, column=0, sticky="w")
        tk.Scale(self.controls, from_=20, to=100, resolution=1, orient="horizontal",
                 length=180, variable=self.height_pct,
                 command=lambda v: self.update_image()
                 ).grid(row=5, column=0, sticky="we")

        self.count_text = tk.Text(self.controls, height=10, width=32)
        self.count_text.grid(row=6, column=0, sticky="we", pady=(10, 0))

    def compute_boundaries(self, h):
        """Compute ROI boundaries based on current settings."""
        num = max(1, self.num_rois.get())
        offset_frac = self.offset_pct.get() / 100.0
        height_frac = self.height_pct.get() / 100.0

        offset_frac = max(0.0, min(offset_frac, 1.0))
        height_frac = max(0.05, min(height_frac, 1.0 - offset_frac))

        roi_y_start = int(h * offset_frac)
        roi_y_end = int(h * (offset_frac + height_frac))
        roi_y_start = max(0, min(roi_y_start, h - 1))
        roi_y_end = max(roi_y_start + 1, min(roi_y_end, h))

        total_roi_height = roi_y_end - roi_y_start
        step = total_roi_height // num if num > 0 else total_roi_height

        boundaries = []
        for i in range(num):
            y_start = roi_y_start + i * step
            if i == num - 1:
                y_end = roi_y_end
            else:
                y_end = roi_y_start + (i + 1) * step
            boundaries.append((y_start, y_end))

        return boundaries

    def update_image(self):
        """Compute ROIs, count parts per ROI, and display."""
        current_part = self.app.current_part
        features_list = self.app.features_list
        part_contours = self.app.part_contours
        if current_part is None or features_list is None or part_contours is None:
            return

        h, w, _ = current_part.shape
        boundaries = self.compute_boundaries(h)

        roi_counts = [
            {"marker": 0, "small": 0, "medium": 0, "large": 0}
            for _ in range(len(boundaries))
        ]

        for f in features_list:
            cx, cy = f["centroid"]
            label = f["label"]

            roi_index = 0
            for i, (y_start, y_end) in enumerate(boundaries):
                if y_start <= cy < y_end:
                    roi_index = i
                    break

            if label not in roi_counts[roi_index]:
                roi_counts[roi_index][label] = 0
            roi_counts[roi_index][label] += 1

        # Draw ROIs
        roi_vis = current_part.copy()
        overlay = roi_vis.copy()

        for i, (y_start, y_end) in enumerate(boundaries):
            color = (0, 255, 255) if i % 2 == 0 else (0, 128, 255)
            cv2.rectangle(overlay, (0, y_start), (w - 1, y_end), color, thickness=-1)

            counts = roi_counts[i]
            text = f"R{i+1} S:{counts['small']} M:{counts['medium']} L:{counts['large']}"
            cv2.putText(overlay, text, (10, y_start + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.3, roi_vis, 0.7, 0, dst=roi_vis)

        self.set_image(roi_vis)
        self.app.boundaries = boundaries
        self.app.roi_vis = roi_vis
        self.app.roi_counts = roi_counts

        self.count_text.delete("1.0", tk.END)
        for i, c in enumerate(roi_counts):
            self.count_text.insert(
                tk.END,
                f"ROI {i+1} -> marker:{c['marker']} small:{c['small']} "
                f"medium:{c['medium']} large:{c['large']}\n"
            )

        # Save config
        self.app.save_config_section("rois", {
            "num_rois": self.num_rois.get(),
            "offset_pct": self.offset_pct.get(),
            "height_pct": self.height_pct.get(),
        })

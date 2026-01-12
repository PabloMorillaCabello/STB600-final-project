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
from ..color import detect_piece_color_and_check_size, classify_by_color_ranges
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
        # Update ROI tab button text to show new color
        if hasattr(self.app, 'tab_rois') and hasattr(self.app.tab_rois, '_update_save_button_text'):
            self.app.tab_rois._update_save_button_text()
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
        
        # Run full pipeline to detect color (InnerColorPartsTab sets app.piece_color)
        self.app.initial_update()
        
        # Sync dropdown with detected color from pipeline
        self._sync_color_dropdown()

    def _sync_color_dropdown(self):
        """
        Sync the dropdown with the color detected by the pipeline.
        
        The color detection happens in InnerColorPartsTab and stores the
        result in self.app.piece_color. This method updates the dropdown
        to match, loading the appropriate ROI settings.
        """
        detected = self.app.piece_color
        current = self.color_var.get()
        
        if detected in ("blue", "yellow", "red") and detected != current:
            self.color_var.set(detected)
            self.app.SECOND_COLOR = detected
            # Load ROI settings for the detected color
            self.app.apply_rois_for_color(detected)
            # Update ROI tab button text to show detected color
            if hasattr(self.app, 'tab_rois') and hasattr(self.app.tab_rois, '_update_save_button_text'):
                self.app.tab_rois._update_save_button_text()

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
    """Tab for detecting and filtering large contours by area with per-color ranges.
    
    Known physical sizes (mm):
    - Red: 38 x 76 = 2888 mm²
    - Yellow: 38 x 114 = 4332 mm²
    - Blue: 38 x 150 = 5700 mm²
    
    Each color has its own min/max range. Pieces falling between ranges are flagged as fakes.
    """

    # Physical area constants (mm²)
    PHYSICAL_AREAS = {"red": 2888, "yellow": 4332, "blue": 5700}
    TOLERANCE = 0.05  # 5% tolerance for fake detection

    def __init__(self, parent, app):
        super().__init__(parent, app, title="4) Large contours")

        # Per-color min/max ranges (6 sliders total)
        self.red_min = tk.IntVar(value=5000)
        self.red_max = tk.IntVar(value=80000)
        self.yellow_min = tk.IntVar(value=80000)
        self.yellow_max = tk.IntVar(value=160000)
        self.blue_min = tk.IntVar(value=160000)
        self.blue_max = tk.IntVar(value=250000)

        row = 0
        # Red range
        ttk.Label(self.controls, text="Red (small)", font=("", 9, "bold"),
                  foreground="#008800").grid(row=row, column=0, sticky="w")
        row += 1
        self._add_range_sliders("red", row)
        row += 2

        # Yellow range
        ttk.Label(self.controls, text="Yellow (medium)", font=("", 9, "bold"),
                  foreground="#CC8800").grid(row=row, column=0, sticky="w")
        row += 1
        self._add_range_sliders("yellow", row)
        row += 2

        # Blue range
        ttk.Label(self.controls, text="Blue (large)", font=("", 9, "bold"),
                  foreground="#0066CC").grid(row=row, column=0, sticky="w")
        row += 1
        self._add_range_sliders("blue", row)
        row += 2

        # Area display text box
        ttk.Label(self.controls, text="Detected areas:").grid(row=row, column=0, sticky="w", pady=(8, 0))
        row += 1
        self.area_text = tk.Text(self.controls, height=5, width=28, state="disabled")
        self.area_text.grid(row=row, column=0, sticky="we", pady=(2, 0))
        row += 1

        # Suggested boundaries display
        ttk.Label(self.controls, text="Suggested (±5%):").grid(row=row, column=0, sticky="w", pady=(8, 0))
        row += 1
        self.suggest_text = tk.Text(self.controls, height=5, width=28, state="disabled")
        self.suggest_text.grid(row=row, column=0, sticky="we", pady=(2, 0))
        row += 1

        # Apply suggested button
        self.apply_btn = ttk.Button(self.controls, text="Apply Suggested", command=self._apply_suggested)
        self.apply_btn.grid(row=row, column=0, sticky="we", pady=(5, 0))

        # Store suggested values for apply button
        self._suggested_ranges = None

    def _add_range_sliders(self, color, start_row):
        """Add min/max sliders for a specific color."""
        min_var = getattr(self, f"{color}_min")
        max_var = getattr(self, f"{color}_max")

        # Min slider
        frame_min = ttk.Frame(self.controls)
        frame_min.grid(row=start_row, column=0, sticky="we")
        ttk.Label(frame_min, text="min:", width=4).pack(side="left")
        tk.Scale(frame_min, from_=0, to=500000, orient="horizontal", length=150,
                 variable=min_var, resolution=1000,
                 command=lambda v: self.update_image()).pack(side="left", fill="x", expand=True)

        # Max slider
        frame_max = ttk.Frame(self.controls)
        frame_max.grid(row=start_row + 1, column=0, sticky="we")
        ttk.Label(frame_max, text="max:", width=4).pack(side="left")
        tk.Scale(frame_max, from_=0, to=500000, orient="horizontal", length=150,
                 variable=max_var, resolution=1000,
                 command=lambda v: self.update_image()).pack(side="left", fill="x", expand=True)

    def _calculate_suggested_boundaries(self, main_area, main_color):
        """Calculate suggested per-color ranges based on detected main piece.
        
        Args:
            main_area: Measured pixel area of the main piece
            main_color: Color of the main piece ('red', 'yellow', or 'blue')
        
        Returns:
            dict with suggested min/max for each color
        """
        phys = self.PHYSICAL_AREAS
        tol = self.TOLERANCE

        main_phys_area = phys.get(main_color.lower())
        if not main_phys_area:
            return None

        # Calculate pixels-per-mm² ratio from the known piece
        ratio = main_area / main_phys_area

        # Calculate expected pixel areas and ranges for each color
        ranges = {}
        for clr in ["red", "yellow", "blue"]:
            expected = phys[clr] * ratio
            ranges[clr] = {
                "min": int(expected * (1 - tol)),
                "max": int(expected * (1 + tol)),
                "expected": int(expected),
            }

        return ranges

    def _apply_suggested(self):
        """Apply all suggested boundary values to the sliders."""
        if self._suggested_ranges is None:
            return

        for clr in ["red", "yellow", "blue"]:
            if clr in self._suggested_ranges:
                getattr(self, f"{clr}_min").set(self._suggested_ranges[clr]["min"])
                getattr(self, f"{clr}_max").set(self._suggested_ranges[clr]["max"])
        self.update_image()

    def _classify_by_ranges(self, area):
        """Classify a contour by its area using per-color ranges.
        
        Uses the shared classify_by_color_ranges function and adds
        display-specific information (BGR colors, display labels).
        
        Returns:
            tuple: (label, color_bgr, size_color_name)
            - label: 'small', 'medium', 'large', or 'fake?'
            - color_bgr: BGR color for drawing
            - size_color_name: 'red', 'yellow', 'blue', or None for fake
        """
        red_range = (self.red_min.get(), self.red_max.get())
        yellow_range = (self.yellow_min.get(), self.yellow_max.get())
        blue_range = (self.blue_min.get(), self.blue_max.get())

        size_label, expected_color = classify_by_color_ranges(area, red_range, yellow_range, blue_range)

        # Map to display format with BGR colors
        display_map = {
            "red": ("small", (0, 255, 0)),      # Green for small/red
            "yellow": ("medium", (0, 200, 255)), # Orange for medium/yellow
            "blue": ("large", (255, 150, 0)),    # Blue-ish for large/blue
        }

        if expected_color is not None and expected_color in display_map:
            label, color_bgr = display_map[expected_color]
            return label, color_bgr, expected_color
        else:
            # Doesn't fit any valid range - potential fake
            return "fake?", (128, 0, 128), None

    def update_image(self):
        """Find contours, filter by area ranges, and display with labels."""
        img = self.app.get_base_image()
        gray_closed = self.app.gray_closed
        if img is None or gray_closed is None:
            return

        contours, _ = cv2.findContours(gray_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Expand detection range by 20% to catch potential fakes
        # that are slightly smaller than red_min or larger than blue_max
        detection_min = int(self.red_min.get() * 0.8)  # 20% smaller than red_min
        detection_max = int(self.blue_max.get() * 1.2)  # 20% larger than blue_max

        filtered = []
        labeled_info = []
        filtered_out = []  # Track pieces outside detection range

        for c in contours:
            area = cv2.contourArea(c)
            
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            
            # Filter by expanded bounds (to catch fakes)
            if not (detection_min <= area <= detection_max):
                # Track filtered out pieces for display
                if area > 1000:  # Only track meaningful contours
                    filtered_out.append((area, cx, cy))
                continue
            filtered.append(c)

            label, color, size_color = self._classify_by_ranges(area)
            labeled_info.append((c, label, color, (cx, cy), area, size_color))

        main_cnt = None
        main_label = None
        if filtered:
            main_cnt = max(filtered, key=cv2.contourArea)

        out = img.copy()
        for c, label, color, (cx, cy), area, size_color in labeled_info:
            label_to_draw = label
            draw_color = color

            if main_cnt is not None and np.array_equal(c, main_cnt):
                main_label = label
                # Check color consistency
                expected_piece_color = self.app.piece_color
                if size_color is not None and expected_piece_color != size_color:
                    label_to_draw = "FAKE"
                    draw_color = (0, 0, 255)  # Red for color mismatch

            cv2.drawContours(out, [c], -1, draw_color, 3)
            cv2.putText(out, label_to_draw, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2, cv2.LINE_AA)

        self.app.big_contours = filtered
        self.app.all_contours = contours
        self.app.contour_img = out
        self.app.main_cnt = main_cnt
        self.app.main_size_label = main_label if main_label is not None else "unknown"

        self.set_image(out)

        # Update area display - show both detected and filtered out pieces
        self.area_text.config(state="normal")
        self.area_text.delete("1.0", tk.END)
        main_area = None
        if labeled_info:
            sorted_info = sorted(labeled_info, key=lambda x: x[4], reverse=True)
            for i, (c, label, color, (cx, cy), area, size_color) in enumerate(sorted_info):
                is_main = main_cnt is not None and np.array_equal(c, main_cnt)
                marker = " *" if is_main else ""
                if is_main:
                    main_area = area
                self.area_text.insert(tk.END, f"{i+1}. {label}: {area:,}{marker}\n")
            self.area_text.insert(tk.END, f"Detected: {len(labeled_info)}\n")
        else:
            self.area_text.insert(tk.END, "No pieces detected\n")
        
        # Show filtered out pieces so user can see what's being missed
        if filtered_out:
            self.area_text.insert(tk.END, f"--- Outside range ---\n")
            sorted_out = sorted(filtered_out, key=lambda x: x[0], reverse=True)
            for area, cx, cy in sorted_out[:5]:  # Show top 5
                self.area_text.insert(tk.END, f"  {area:,} @ ({cx},{cy})\n")
            if len(filtered_out) > 5:
                self.area_text.insert(tk.END, f"  (+{len(filtered_out)-5} more)\n")
        self.area_text.config(state="disabled")

        # Update suggested boundaries
        self.suggest_text.config(state="normal")
        self.suggest_text.delete("1.0", tk.END)
        piece_color = self.app.piece_color
        if main_area and piece_color in ["blue", "red", "yellow"]:
            suggested = self._calculate_suggested_boundaries(main_area, piece_color)
            if suggested:
                self._suggested_ranges = suggested
                self.suggest_text.insert(tk.END, f"From {piece_color.upper()} ({main_area:,}px):\n")
                for clr in ["red", "yellow", "blue"]:
                    r = suggested[clr]
                    self.suggest_text.insert(tk.END, f"  {clr}: {r['min']:,} - {r['max']:,}\n")
            else:
                self.suggest_text.insert(tk.END, "Cannot calculate")
                self._suggested_ranges = None
        else:
            self.suggest_text.insert(tk.END, "Place known piece,\nset color in Tab 1")
            self._suggested_ranges = None
        self.suggest_text.config(state="disabled")

        # Save config (now with per-color ranges)
        self.app.save_config_section("big_contours", {
            "red_min": self.red_min.get(),
            "red_max": self.red_max.get(),
            "yellow_min": self.yellow_min.get(),
            "yellow_max": self.yellow_max.get(),
            "blue_min": self.blue_min.get(),
            "blue_max": self.blue_max.get(),
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
                current_part, features_list, main_contour_height=h, top_fraction=0.33
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

        # Save ROIs button with current color indicator
        self.save_btn = ttk.Button(
            self.controls, text="💾 Save ROIs", command=self._save_rois_explicit
        )
        self.save_btn.grid(row=6, column=0, sticky="we", pady=(10, 0))
        
        # Status label to show save status
        self.status_var = tk.StringVar(value="")
        self.status_label = ttk.Label(self.controls, textvariable=self.status_var, foreground="green")
        self.status_label.grid(row=7, column=0, sticky="w")

        self.count_text = tk.Text(self.controls, height=8, width=32)
        self.count_text.grid(row=8, column=0, sticky="we", pady=(10, 0))
        
        # Update button text to show current color
        self._update_save_button_text()

    def _update_save_button_text(self):
        """Update the save button to show current color."""
        color = self.app.piece_color.upper()
        self.save_btn.config(text=f"💾 Save ROIs for {color}")

    def _save_rois_explicit(self):
        """Explicitly save ROI settings for the current color."""
        color = self.app.piece_color
        
        # Save to config
        self.app.save_config_section("rois", {
            "num_rois": self.num_rois.get(),
            "offset_pct": self.offset_pct.get(),
            "height_pct": self.height_pct.get(),
        })
        
        # Show confirmation
        self.status_var.set(f"✓ Saved for {color.upper()}")
        
        # Clear status after 3 seconds
        self.app.root.after(3000, lambda: self.status_var.set(""))

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
        
        # Show current values (NOT auto-saved - use Save button)
        color = self.app.piece_color.upper()
        self.count_text.insert(
            tk.END,
            f"\n[{color}] ROIs: {self.num_rois.get()}, "
            f"Offset: {self.offset_pct.get()}%, "
            f"Height: {self.height_pct.get()}%"
        )

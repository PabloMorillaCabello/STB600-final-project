"""
Result Viewer GUI - Automatic Pipeline with Value Display
==========================================================

A simple GUI that loads an image, runs the full pipeline automatically,
and displays the result with bounding boxes and decoded values for multiple pieces.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from typing import List, Optional

# NEW: Basler / pypylon
from pypylon import pylon

from ..color import classify_by_color_ranges
from ..contours import (
    extract_contour_features,
    label_contour_by_extent_and_area,
    crop_and_align_vertical,
)
from ..detection import detect_pieces_in_frame, get_detection_ranges
from ..display import annotate_multiple_pieces
from ..decoding import decode_roi_to_number, compute_total_value_from_rois
from ..transforms import rotate_if_marker_bottom, rotate_if_blue_piece_upside_down
from ..pipeline import PieceResult
from .config import load_config, get_color_config, DEFAULT_CONFIG_PATH


def compute_rois_from_config(image_shape, features_list, piece_color, roi_config):
    """
    Compute ROIs using saved config settings (matching run_pipeline_gui behavior).
    """
    if not roi_config:
        raise ValueError("ROI config is required. Run run_pipeline_gui to configure ROI settings.")
    
    h, w = image_shape[:2]
    color = (piece_color or "").lower()

    # Get color-specific ROI settings from config
    color_roi = roi_config.get(color)
    if not color_roi:
        raise ValueError(f"No ROI config found for '{color}'. Run run_pipeline_gui and save ROI settings for {color}.")
    
    num_rois = color_roi.get("num_rois")
    offset_pct = color_roi.get("offset_pct")
    height_pct = color_roi.get("height_pct")
    
    if num_rois is None or offset_pct is None or height_pct is None:
        raise ValueError(f"Incomplete ROI config for '{color}'. Expected num_rois, offset_pct, height_pct.")

    # Compute boundaries (same logic as ROIsTab.compute_boundaries)
    offset_frac = max(0.0, min(offset_pct / 100.0, 1.0))
    height_frac = max(0.05, min(height_pct / 100.0, 1.0 - offset_frac))

    roi_y_start = int(h * offset_frac)
    roi_y_end = int(h * (offset_frac + height_frac))
    roi_y_start = max(0, min(roi_y_start, h - 1))
    roi_y_end = max(roi_y_start + 1, min(roi_y_end, h))

    total_roi_height = roi_y_end - roi_y_start
    step = total_roi_height // num_rois if num_rois > 0 else total_roi_height

    boundaries = []
    for i in range(num_rois):
        y_start = roi_y_start + i * step
        if i == num_rois - 1:
            y_end = roi_y_end
        else:
            y_end = roi_y_start + (i + 1) * step
        boundaries.append((y_start, y_end))

    # Build ROIs with counts
    rois = []
    for i, (ys, ye) in enumerate(boundaries):
        rois.append({
            "index": i,
            "y_start": ys,
            "y_end": ye,
            "counts": {"marker": 0, "small": 0, "medium": 0, "large": 0, "unknown": 0}
        })

    # Count features per ROI
    for f in features_list:
        cx, cy = f.get("centroid", (None, None))
        label = f.get("label", "unknown")
        if cy is None:
            continue
        cy = int(cy)

        for roi in rois:
            if roi["y_start"] <= cy < roi["y_end"]:
                if label in roi["counts"]:
                    roi["counts"][label] += 1
                else:
                    roi["counts"]["unknown"] += 1
                break

    return rois, boundaries


def process_single_contour(
    img,
    contour,
    red_range=(5000, 80000),
    yellow_range=(80000, 160000),
    blue_range=(160000, 250000),
    inner_min_area=50,
    inner_max_area=10000,
    extent_threshold=0.5,
    label_small_area_max=2100,
    label_medium_area_max=5600,
    roi_config=None,
    debug=False,
):
    """
    Process a single contour through the detection pipeline.
    """
    area = cv2.contourArea(contour)
    
    # First, classify by area using per-color ranges
    size_label, expected_color_from_size = classify_by_color_ranges(
        area, red_range, yellow_range, blue_range
    )
    
    # If area doesn't fit any range, it's a fake by size
    size_is_fake = (size_label is None)
    if size_is_fake:
        size_label = "UNKNOWN"  # Will show as fake
    
    # Detect actual piece color from image
    # Create contour mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # HSV ranges for color detection
    blue_lower, blue_upper = np.array([100, 80, 80]), np.array([130, 255, 255])
    yellow_lower, yellow_upper = np.array([20, 80, 80]), np.array([35, 255, 255])
    red_lower1, red_upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    red_lower2, red_upper2 = np.array([160, 70, 50]), np.array([179, 255, 255])
    
    blue_mask = cv2.bitwise_and(cv2.inRange(hsv, blue_lower, blue_upper), mask)
    yellow_mask = cv2.bitwise_and(cv2.inRange(hsv, yellow_lower, yellow_upper), mask)
    red_mask = cv2.bitwise_or(
        cv2.bitwise_and(cv2.inRange(hsv, red_lower1, red_upper1), mask),
        cv2.bitwise_and(cv2.inRange(hsv, red_lower2, red_upper2), mask)
    )
    
    color_counts = {
        "blue": cv2.countNonZero(blue_mask),
        "yellow": cv2.countNonZero(yellow_mask),
        "red": cv2.countNonZero(red_mask),
    }
    
    # Determine actual piece color
    piece_color = max(color_counts, key=color_counts.get)
    dominant_pixels = color_counts[piece_color]
    color_pct = (dominant_pixels / area * 100) if area > 0 else 0
    
    # Determine consistency:
    # 1. Size must fit a valid range
    # 2. Detected color must match expected color for that size
    if size_is_fake:
        is_consistent = False
    else:
        is_consistent = (piece_color == expected_color_from_size)
    
    # Debug output
    if debug:
        print(f"Piece detected: color={piece_color}, size={size_label}, area={area:.0f}, "
              f"color_pct={color_pct:.1f}%, expected_color={expected_color_from_size}, consistent={is_consistent}")
    
    second_color = piece_color if is_consistent else "blue"

    # Crop and align
    cropped_aligned = crop_and_align_vertical(img, contour)
    if cropped_aligned is None or cropped_aligned.size == 0:
        x, y, w, h = cv2.boundingRect(contour)
        H, W = img.shape[:2]
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        cropped_aligned = img[y:y + h, x:x + w]

    current_part = cropped_aligned

    # Detect internal color parts (same logic as InnerColorPartsTab)
    hsv = cv2.cvtColor(current_part, cv2.COLOR_BGR2HSV)
    
    if second_color == "blue":
        lower = np.array([90, 50, 50])
        upper = np.array([130, 255, 255])
    elif second_color == "red":
        lower = np.array([0, 50, 50])
        upper = np.array([10, 255, 255])
    else:  # yellow
        lower = np.array([20, 50, 50])
        upper = np.array([35, 255, 255])
    
    inner_mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inner_mask_closed = cv2.morphologyEx(inner_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    inner_contours, _ = cv2.findContours(inner_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area (using config values)
    part_contours = []
    for c in inner_contours:
        c_area = cv2.contourArea(c)
        if inner_min_area <= c_area <= inner_max_area:
            part_contours.append(c)

    # Extract features and label
    features_list = extract_contour_features(part_contours)
    for f in features_list:
        f["label"] = label_contour_by_extent_and_area(
            f,
            extent_threshold=extent_threshold,
            small_area_max=label_small_area_max,
            medium_area_max=label_medium_area_max
        )

    # Check rotation
    rotated = False
    rotated_part, did_rotate = rotate_if_marker_bottom(
        current_part, features_list, bottom_fraction=0.33
    )
    if did_rotate:
        rotated = True
        current_part = rotated_part

    if not rotated and piece_color == "blue":
        rotated_part, did_rotate = rotate_if_blue_piece_upside_down(
            current_part, features_list,
            main_contour_height=current_part.shape[0],
            top_fraction=0.33
        )
        if did_rotate:
            rotated = True
            current_part = rotated_part

    # If rotated, recalculate features using same logic as above
    if rotated:
        hsv = cv2.cvtColor(current_part, cv2.COLOR_BGR2HSV)
        
        if second_color == "blue":
            lower = np.array([90, 50, 50])
            upper = np.array([130, 255, 255])
        elif second_color == "red":
            lower = np.array([0, 50, 50])
            upper = np.array([10, 255, 255])
        else:  # yellow
            lower = np.array([20, 50, 50])
            upper = np.array([35, 255, 255])
        
        inner_mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        inner_mask_closed = cv2.morphologyEx(inner_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        inner_contours, _ = cv2.findContours(inner_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        part_contours = []
        for c in inner_contours:
            c_area = cv2.contourArea(c)
            if inner_min_area <= c_area <= inner_max_area:
                part_contours.append(c)
        
        features_list = extract_contour_features(part_contours)
        for f in features_list:
            f["label"] = label_contour_by_extent_and_area(
                f,
                extent_threshold=extent_threshold,
                small_area_max=label_small_area_max,
                medium_area_max=label_medium_area_max
            )

    # Compute ROIs using config (must be provided)
    rois, boundaries = compute_rois_from_config(
        image_shape=current_part.shape,
        features_list=features_list,
        piece_color=piece_color,
        roi_config=roi_config,
    )

    decoded_digits = []
    for r in rois:
        try:
            digit = decode_roi_to_number(r["counts"])
            decoded_digits.append(digit)
        except ValueError as e:
            if debug:
                print(f"  ROI {r.get('index', '?')} decode failed: {e}, counts={r['counts']}")
            decoded_digits.append(None)

    total_value = compute_total_value_from_rois(decoded_digits, piece_color)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)

    return PieceResult(
        piece_color=piece_color,
        size_label=size_label,
        is_consistent=is_consistent,
        total_value=total_value,
        contour=contour,
        bounding_box=(x, y, w, h),
    )


class ResultViewerApp:
    """
    Result viewer that processes multiple pieces in loaded images.

    Displays annotated image with bounding boxes and decoded values for all pieces.
    Designed for easy extension to video and camera input.
    """

    def __init__(self, root, config_path=DEFAULT_CONFIG_PATH):
        self.root = root
        self.root.title("STB600 Result Viewer - Multi-Piece")
        self.root.geometry("1000x800")

        # Load config
        self.config_path = config_path
        self.config = load_config(config_path)

        # Current image and results
        self.current_image = None
        self.current_results = []  # List of PieceResult
        self.annotated_image = None

        # Source type: "image" or "camera"
        self.source_type = "image"

        # Basler camera state
        self.basler_camera: Optional[pylon.InstantCamera] = None
        self.basler_converter: Optional[pylon.ImageFormatConverter] = None
        self.camera_running = False

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the user interface."""
        # Top control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Load buttons
        btn_load_image = ttk.Button(
            control_frame, text="Load Image", command=self.load_image
        )
        btn_load_image.pack(side=tk.LEFT, padx=(0, 5))

        btn_load_video = ttk.Button(
            control_frame, text="Load Video", command=self.load_video, state="disabled"
        )
        btn_load_video.pack(side=tk.LEFT, padx=5)
        self.btn_load_video = btn_load_video

        # Camera button now enabled: toggles Basler camera
        btn_camera = ttk.Button(
            control_frame, text="Camera", command=self.toggle_camera
        )
        btn_camera.pack(side=tk.LEFT, padx=5)
        self.btn_camera = btn_camera

        # Status label
        self.status_var = tk.StringVar(value="Load an image to begin")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=10)

        # Main content area with paned window
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Left side: Image display
        image_frame = ttk.Frame(paned)
        paned.add(image_frame, weight=3)

        self.canvas = tk.Canvas(image_frame, bg="#2d2d2d")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right side: Results panel
        results_frame = ttk.LabelFrame(paned, text="Detected Pieces")
        paned.add(results_frame, weight=1)

        # Results treeview with scrollbar
        tree_frame = ttk.Frame(results_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        columns = ("color", "size", "value", "status")
        self.results_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15)

        self.results_tree.heading("color", text="Color")
        self.results_tree.heading("size", text="Size")
        self.results_tree.heading("value", text="Value")
        self.results_tree.heading("status", text="Status")

        self.results_tree.column("color", width=70, anchor="center")
        self.results_tree.column("size", width=70, anchor="center")
        self.results_tree.column("value", width=60, anchor="center")
        self.results_tree.column("status", width=60, anchor="center")

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Summary label
        self.summary_var = tk.StringVar(value="No pieces detected")
        summary_label = ttk.Label(results_frame, textvariable=self.summary_var, font=("TkDefaultFont", 10, "bold"))
        summary_label.pack(pady=10)

        # Total value label
        self.total_var = tk.StringVar(value="")
        total_label = ttk.Label(results_frame, textvariable=self.total_var, font=("TkDefaultFont", 14, "bold"), foreground="#2563eb")
        total_label.pack(pady=5)

        # Store photo reference
        self._photo = None

        # Bind resize event
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------ Image / Video / Camera control ------------------------

    def load_image(self):
        """Open file dialog and load an image."""
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            self.status_var.set("Error: Could not read image")
            return

        # Stop camera if running
        if self.camera_running:
            self._stop_camera()

        self.current_image = img
        self.source_type = "image"
        self.status_var.set("Processing...")
        self.root.update()

        # Run pipeline
        self._process_and_display(img)

    def load_video(self):
        """Load a video file (placeholder for future implementation)."""
        pass

    # --- Basler camera helpers ------------------------------------------------

    def _start_basler_camera(self) -> bool:
        """Initialize and start grabbing from the Basler camera."""
        try:
            self.basler_camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self.basler_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)  # continuous, low latency [web:1]

            self.basler_converter = pylon.ImageFormatConverter()
            self.basler_converter.OutputPixelFormat = pylon.PixelType_BGR8packed  # OpenCV BGR [web:1]
            self.basler_converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            return True
        except Exception as e:
            self.status_var.set(f"Error: Could not open Basler camera ({e})")
            self.basler_camera = None
            self.basler_converter = None
            return False

    def _stop_camera(self):
        """Stop camera grabbing and release resources."""
        self.camera_running = False
        if self.basler_camera is not None:
            try:
                if self.basler_camera.IsGrabbing():
                    self.basler_camera.StopGrabbing()
            except Exception:
                pass
            self.basler_camera = None
        self.basler_converter = None
        self.source_type = "image"
        self.status_var.set("Camera stopped")

    def _grab_camera_frame(self) -> Optional[np.ndarray]:
        """Grab a single frame from Basler camera."""
        if not self.basler_camera or not self.basler_converter:
            return None
        if not self.basler_camera.IsGrabbing():
            return None

        try:
            grab_result = self.basler_camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )
        except Exception:
            return None

        frame = None
        if grab_result is not None:
            if grab_result.GrabSucceeded():
                image = self.basler_converter.Convert(grab_result)
                frame = image.GetArray()
            grab_result.Release()
        return frame

    def toggle_camera(self):
        """Toggle camera input."""
        if self.camera_running:
            self._stop_camera()
            return

        # Start camera
        if not self._start_basler_camera():
            return

        self.camera_running = True
        self.source_type = "camera"
        self.status_var.set("Camera running")
        self.current_image = None
        self.annotated_image = None
        self._clear_results()

        # Kick off camera loop
        self._camera_loop()

    def _camera_loop(self):
        """Periodic loop to grab from camera, process, and display."""
        if not self.camera_running:
            return

        frame = self._grab_camera_frame()
        if frame is not None:
            self.current_image = frame
            self._process_and_display(frame)
        # Schedule next iteration
        self.root.after(50, self._camera_loop)

    # -------------------- Processing and display ------------------------------

    def _process_and_display(self, img):
        """Run the pipeline on all pieces and display results."""
        try:
            # Get config parameters
            cfg = get_color_config(self.config, "blue")
            
            bc = cfg.get("big_contours", {})
            ip = cfg.get("inner_parts", {})
            lb = cfg.get("labeled", {})
            
            # Get ROI config (stored at root level, not under "shared")
            roi_config = self.config.get("rois", {})

            # Use shared detection pipeline (handles preprocessing, inversion, contour finding)
            contours, processed_binary, _ = detect_pieces_in_frame(
                img, cfg, auto_invert=True
            )
            
            # Get detection ranges from shared function
            red_range, yellow_range, blue_range, _, _ = get_detection_ranges(cfg)
            
            if not contours:
                self.status_var.set("No pieces found")
                self._clear_results()
                self._display_image(img)
                return

            # Step 5: Process each contour
            results = []
            for contour in contours:
                try:
                    result = process_single_contour(
                        img,
                        contour,
                        red_range=red_range,
                        yellow_range=yellow_range,
                        blue_range=blue_range,
                        inner_min_area=ip.get("min_area", 50),
                        inner_max_area=ip.get("max_area", 10000),
                        extent_threshold=lb.get("extent_threshold", 0.5),
                        label_small_area_max=lb.get("small_area_max", 2100),
                        label_medium_area_max=lb.get("medium_area_max", 5600),
                        roi_config=roi_config,
                        debug=False,
                    )
                    
                    # Skip rejected contours (not enough color)
                    if result is not None:
                        results.append(result)
                        
                except Exception:
                    continue

            self.current_results = results

            # Annotate image with all results
            self.annotated_image = annotate_multiple_pieces(img, results)

            # Update UI
            self._update_results_display(results)
            self._display_image(self.annotated_image)

            self.status_var.set(f"Found {len(results)} piece(s)")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self._clear_results()
            if img is not None:
                self._display_image(img)

    def _update_results_display(self, results: List[PieceResult]):
        """Update the results panel with detection results."""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Add new items
        total_value = 0
        valid_count = 0

        for i, result in enumerate(results):
            color = result.piece_color.upper()
            size = result.size_label
            value = str(result.total_value) if result.total_value is not None else "?"
            status = "OK" if result.is_consistent else "!"

            self.results_tree.insert("", tk.END, values=(color, size, value, status))

            if result.total_value is not None:
                total_value += result.total_value
                valid_count += 1

        # Update summary
        self.summary_var.set(f"{len(results)} piece(s) detected")

        if valid_count > 0:
            self.total_var.set(f"Total: {total_value}")
        else:
            self.total_var.set("")

    def _clear_results(self):
        """Clear the results display."""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.summary_var.set("No pieces detected")
        self.total_var.set("")
        self.current_results = []

    def _display_image(self, img_bgr):
        """Display a BGR image on the canvas, scaled to fit."""
        if img_bgr is None:
            return

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Get canvas size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 10 or canvas_h < 10:
            self.root.after(50, lambda: self._display_image(img_bgr))
            return

        # Scale image to fit canvas
        img_h, img_w = img_rgb.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        if new_w > 0 and new_h > 0:
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img_rgb

        # Convert to PhotoImage
        pil_img = Image.fromarray(img_resized)
        self._photo = ImageTk.PhotoImage(pil_img)

        # Draw centered
        self.canvas.delete("all")
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self._photo)

    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        if self.annotated_image is not None:
            self._display_image(self.annotated_image)
        elif self.current_image is not None:
            self._display_image(self.current_image)

    def on_close(self):
        """Handle window close."""
        if self.camera_running:
            self._stop_camera()
        cv2.destroyAllWindows()
        self.root.destroy()


def run_result_viewer(config_path=DEFAULT_CONFIG_PATH):
    """
    Launch the result viewer GUI for multi-piece detection.
    """
    root = tk.Tk()
    app = ResultViewerApp(root, config_path=config_path)
    root.mainloop()

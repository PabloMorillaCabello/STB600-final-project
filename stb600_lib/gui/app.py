"""
GUI App Module - Main Pipeline Application
==========================================

Provides the main PipelineApp class that orchestrates all GUI tabs.
"""

import cv2
import os
import tkinter as tk
from tkinter import ttk

# NEW: pypylon imports for Basler
from pypylon import pylon  # type: ignore

from .base import cv_bgr_to_tk
from .config import load_config, save_config, get_color_config, update_color_config, DEFAULT_CONFIG_PATH
from .tabs import (
    InputTab,
    RemoveGreenTab,
    BinarizeTab,
    OpeningTab,
    BigContoursTab,
    CroppedPartTab,
    InnerColorPartsTab,
    LabeledPartsTab,
    ROIsTab,
)


class PipelineApp:
    """
    Main application class for the interactive pipeline GUI.

    Provides a multi-tab interface for tuning parameters at each processing step,
    with optional camera support for real-time processing.

    Parameters
    ----------
    root : tk.Tk
        Root Tkinter window.
    image_path : str, optional
        Path to initial image to load.
    use_camera : bool
        Whether to enable camera on startup.
    config_path : str
        Path to JSON config file for saving/loading per-color settings.
    """

    def __init__(self, root, image_path=None, use_camera=False, config_path=DEFAULT_CONFIG_PATH):
        self.root = root
        self.root.title("STB600 Pipeline Viewer")
        self.root.geometry("1200x900")

        # Config management
        self.config_path = config_path
        self.config = load_config(config_path)

        # Pipeline state variables
        self.base_image = None
        self.no_green = None
        self.green_mask = None
        self.gray = None
        self.binary = None
        self.gray_closed = None
        self.big_contours = []
        self.all_contours = []
        self.contour_img = None
        self.main_cnt = None
        self.main_size_label = "unknown"
        self.cropped_aligned = None
        self.current_part = None
        self.SECOND_COLOR = "blue"
        self.piece_color = "blue"
        self.features_list = None
        self.part_contours = None
        self.rois = None
        self.roi_vis = None
        self.roi_counts = None
        self.boundaries = None

        # Camera state
        # CHANGED: use Basler camera instead of cv2.VideoCapture
        self.camera_running = False
        self.basler_camera = None
        self.basler_converter = None

        # Load initial image if provided
        if use_camera:
            self.set_camera_enabled(True)
        elif image_path and os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                self.base_image = img

        # Create notebook with tabs
        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Create all tabs
        self.tab_input = InputTab(notebook, self)
        self.tab_remove_green = RemoveGreenTab(notebook, self)
        self.tab_binarize = BinarizeTab(notebook, self)
        self.tab_opening = OpeningTab(notebook, self)
        self.tab_big_contours = BigContoursTab(notebook, self)
        self.tab_cropped = CroppedPartTab(notebook, self)
        self.tab_inner = InnerColorPartsTab(notebook, self)
        self.tab_labeled = LabeledPartsTab(notebook, self)
        self.tab_rois = ROIsTab(notebook, self)

        # Add tabs to notebook
        notebook.add(self.tab_input, text="0) Input")
        notebook.add(self.tab_remove_green, text="1) Remove green")
        notebook.add(self.tab_binarize, text="2) Binarize")
        notebook.add(self.tab_opening, text="3) Opening")
        notebook.add(self.tab_big_contours, text="4) Big contours")
        notebook.add(self.tab_cropped, text="4b) Cropped main")
        notebook.add(self.tab_inner, text="5) Inner parts")
        notebook.add(self.tab_labeled, text="6) Labeled")
        notebook.add(self.tab_rois, text="7) ROIs")

        self.notebook = notebook
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # Apply config for initial color
        self.apply_config_for_color(self.piece_color)

        # Run initial pipeline update
        self.root.after(50, self.initial_update)

        # Start camera loop if enabled
        if self.camera_running:
            self.root.after(30, self.update_camera)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def apply_config_for_color(self, color):
        """
        Apply saved configuration values for a specific color to all sliders.

        Shared settings are applied for all tabs, ROI settings are per-color.

        Parameters
        ----------
        color : str
            Color name ('blue', 'red', 'yellow').
        """
        cfg = get_color_config(self.config, color)

        # Apply shared settings
        self._apply_shared_settings(cfg)

        # Apply color-specific ROI settings
        self._apply_roi_settings(cfg)

    def _apply_shared_settings(self, cfg):
        """Apply shared settings (same for all colors) to sliders."""
        # Remove green
        rg = cfg.get("remove_green", {})
        if "h_low" in rg:
            self.tab_remove_green.h_low.set(int(rg["h_low"]))
        if "h_high" in rg:
            self.tab_remove_green.h_high.set(int(rg["h_high"]))

        # Binarize
        bz = cfg.get("binarize", {})
        if "thresh_value" in bz:
            self.tab_binarize.thresh_value.set(int(bz["thresh_value"]))

        # Opening
        op = cfg.get("opening", {})
        if "kernel_size" in op:
            self.tab_opening.kernel_size.set(int(op["kernel_size"]))
        if "iterations" in op:
            self.tab_opening.iterations.set(int(op["iterations"]))

        # Big contours (per-color ranges)
        bc = cfg.get("big_contours", {})
        if "red_min" in bc:
            self.tab_big_contours.red_min.set(int(bc["red_min"]))
        if "red_max" in bc:
            self.tab_big_contours.red_max.set(int(bc["red_max"]))
        if "yellow_min" in bc:
            self.tab_big_contours.yellow_min.set(int(bc["yellow_min"]))
        if "yellow_max" in bc:
            self.tab_big_contours.yellow_max.set(int(bc["yellow_max"]))
        if "blue_min" in bc:
            self.tab_big_contours.blue_min.set(int(bc["blue_min"]))
        if "blue_max" in bc:
            self.tab_big_contours.blue_max.set(int(bc["blue_max"]))

        # Inner parts
        ip = cfg.get("inner_parts", {})
        if "min_area" in ip:
            self.tab_inner.min_area.set(int(ip["min_area"]))
        if "max_area" in ip:
            self.tab_inner.max_area.set(int(ip["max_area"]))

        # Labeled
        lb = cfg.get("labeled", {})
        if "extent_threshold" in lb:
            self.tab_labeled.extent_threshold.set(float(lb["extent_threshold"]))
        if "small_area_max" in lb:
            self.tab_labeled.small_area_max.set(int(lb["small_area_max"]))
        if "medium_area_max" in lb:
            self.tab_labeled.medium_area_max.set(int(lb["medium_area_max"]))

    def _apply_roi_settings(self, cfg):
        """Apply ROI settings (color-specific) to sliders."""
        r = cfg.get("rois", {})
        if "num_rois" in r:
            self.tab_rois.num_rois.set(int(r["num_rois"]))
        if "offset_pct" in r:
            self.tab_rois.offset_pct.set(int(r["offset_pct"]))
        if "height_pct" in r:
            self.tab_rois.height_pct.set(int(r["height_pct"]))

    def apply_rois_for_color(self, color):
        """
        Apply only ROI settings for a specific color.

        Called when color changes to load color-specific ROI values.

        Parameters
        ----------
        color : str
            Color name ('blue', 'red', 'yellow').
        """
        cfg = get_color_config(self.config, color)
        self._apply_roi_settings(cfg)

    def save_config_section(self, section, values):
        """
        Save configuration values for current color and section to JSON.

        Parameters
        ----------
        section : str
            Section name (e.g., 'remove_green', 'binarize', etc.).
        values : dict
            Values to save.
        """
        update_color_config(self.config, self.piece_color, section, values)
        save_config(self.config, self.config_path)

    def set_base_image(self, img):
        """Set a new base image and reset pipeline state."""
        self.base_image = img
        self.no_green = None
        self.green_mask = None
        self.gray = None
        self.binary = None
        self.gray_closed = None
        self.big_contours = []
        self.all_contours = []
        self.contour_img = None
        self.main_cnt = None
        self.main_size_label = "unknown"
        self.cropped_aligned = None
        self.current_part = None
        self.features_list = None
        self.part_contours = None
        self.rois = None
        self.roi_vis = None
        self.roi_counts = None
        self.boundaries = None

    def _start_basler_camera(self):
        """
        Initialize and start grabbing from the Basler camera using pypylon.
        """
        # Connect to first available Basler camera
        self.basler_camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())  
        self.basler_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)  # continuous with minimal delay [web:7]

        # ImageFormatConverter to get BGR for OpenCV
        self.basler_converter = pylon.ImageFormatConverter() 
        self.basler_converter.OutputPixelFormat = pylon.PixelType_BGR8packed  # BGR8 for OpenCV [web:5]
        self.basler_converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned  

    def _stop_basler_camera(self):
        """Stop grabbing and release Basler camera resources."""
        if self.basler_camera is not None:
            try:
                if self.basler_camera.IsGrabbing():
                    self.basler_camera.StopGrabbing()
            except Exception:
                pass
            self.basler_camera = None
        self.basler_converter = None

    def set_camera_enabled(self, enabled):
        """Enable or disable Basler camera input."""
        if enabled and not self.camera_running:
            self._start_basler_camera()
            self.camera_running = True
            self.root.after(30, self.update_camera)
        elif not enabled and self.camera_running:
            self._stop_basler_camera()
            self.camera_running = False

    def get_base_image(self):
        """Get the current base image."""
        return self.base_image

    def initial_update(self):
        """Run initial pipeline update on all tabs."""
        if self.base_image is None:
            self.tab_input.update_image()
            return
        self.tab_input.update_image()
        self.tab_remove_green.update_image()
        self.tab_binarize.update_image()
        self.tab_opening.update_image()
        self.tab_big_contours.update_image()
        self.tab_cropped.update_image()
        self.tab_inner.update_image()
        self.tab_labeled.update_image()
        self.tab_rois.update_image()

    def on_tab_changed(self, event):
        """Handle tab change events."""
        selected_id = event.widget.select()
        tab_widget = event.widget.nametowidget(selected_id)
        if hasattr(tab_widget, "update_image"):
            tab_widget.update_image()

    def update_camera(self):
        """
        Camera update loop using Basler frames.

        Grabs the latest image from the Basler camera and updates the input tab.
        """
        if self.camera_running and self.basler_camera is not None and self.basler_camera.IsGrabbing():
            try:
                grab_result = self.basler_camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)  
            except Exception:
                grab_result = None

            if grab_result is not None:
                if grab_result.GrabSucceeded():  
                    # Convert to OpenCV BGR image
                    image = self.basler_converter.Convert(grab_result) 
                    frame = image.GetArray()  
                    self.base_image = frame
                    self.tab_input.update_image()
                grab_result.Release()
        # Schedule next frame grab
        if self.camera_running:
            self.root.after(30, self.update_camera)

    def on_close(self):
        """Handle window close."""
        # Stop Basler camera if running
        self._stop_basler_camera()
        cv2.destroyAllWindows()
        self.root.destroy()


def run_pipeline_gui(image_path=None, use_camera=False, config_path=DEFAULT_CONFIG_PATH):
    """
    Launch the interactive pipeline GUI.

    Parameters
    ----------
    image_path : str, optional
        Path to initial image to load.
    use_camera : bool
        Whether to enable camera on startup.
    config_path : str
        Path to JSON config file (default: pipeline_config.json).

    Example
    -------
    >>> from stb600_lib.gui import run_pipeline_gui
    >>> run_pipeline_gui("images/blue.png")
    """
    root = tk.Tk()
    app = PipelineApp(root, image_path=image_path, use_camera=use_camera, config_path=config_path)
    root.mainloop()

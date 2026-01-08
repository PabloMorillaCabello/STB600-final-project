"""
Result Viewer GUI - Automatic Pipeline with Value Display
==========================================================

A simple GUI that loads an image, runs the full pipeline automatically,
and displays the result with bounding box and decoded value.
"""

import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

from ..pipeline import process_piece, PipelineResult
from ..color import detect_piece_color_and_check_size
from .config import load_config, get_color_config, DEFAULT_CONFIG_PATH


class ResultViewerApp:
    """
    Simple result viewer that runs the full pipeline on loaded images.

    Displays the annotated image with bounding box and decoded value.
    Designed for easy extension to video and camera input.

    Parameters
    ----------
    root : tk.Tk
        Root Tkinter window.
    config_path : str
        Path to configuration JSON file.
    """

    def __init__(self, root, config_path=DEFAULT_CONFIG_PATH):
        self.root = root
        self.root.title("STB600 Result Viewer")
        self.root.geometry("900x700")

        # Load config
        self.config_path = config_path
        self.config = load_config(config_path)

        # Current image and result
        self.current_image = None
        self.current_result = None

        # Source type for future expansion (image, video, camera)
        self.source_type = "image"
        self.cap = None  # For video/camera

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

        # Placeholder buttons for future video/camera support
        btn_load_video = ttk.Button(
            control_frame, text="Load Video", command=self.load_video, state="disabled"
        )
        btn_load_video.pack(side=tk.LEFT, padx=5)
        self.btn_load_video = btn_load_video

        btn_camera = ttk.Button(
            control_frame, text="Camera", command=self.toggle_camera, state="disabled"
        )
        btn_camera.pack(side=tk.LEFT, padx=5)
        self.btn_camera = btn_camera

        # Status label
        self.status_var = tk.StringVar(value="Load an image to begin")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=10)

        # Result info panel
        info_frame = ttk.LabelFrame(self.root, text="Detection Result")
        info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Result labels
        result_grid = ttk.Frame(info_frame)
        result_grid.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(result_grid, text="Color:").grid(row=0, column=0, sticky="e", padx=5)
        self.color_var = tk.StringVar(value="—")
        self.color_label = ttk.Label(
            result_grid, textvariable=self.color_var, font=("TkDefaultFont", 12, "bold")
        )
        self.color_label.grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(result_grid, text="Size:").grid(row=0, column=2, sticky="e", padx=5)
        self.size_var = tk.StringVar(value="—")
        ttk.Label(result_grid, textvariable=self.size_var, font=("TkDefaultFont", 12)).grid(
            row=0, column=3, sticky="w", padx=5
        )

        ttk.Label(result_grid, text="Value:").grid(row=0, column=4, sticky="e", padx=5)
        self.value_var = tk.StringVar(value="—")
        self.value_label = ttk.Label(
            result_grid, textvariable=self.value_var,
            font=("TkDefaultFont", 18, "bold"), foreground="#2563eb"
        )
        self.value_label.grid(row=0, column=5, sticky="w", padx=5)

        ttk.Label(result_grid, text="Status:").grid(row=0, column=6, sticky="e", padx=5)
        self.status_result_var = tk.StringVar(value="—")
        self.status_result_label = ttk.Label(
            result_grid, textvariable=self.status_result_var, font=("TkDefaultFont", 12, "bold")
        )
        self.status_result_label.grid(row=0, column=7, sticky="w", padx=5)

        # Image display canvas
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.canvas = tk.Canvas(canvas_frame, bg="#2d2d2d")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Store photo reference to prevent garbage collection
        self._photo = None

        # Bind resize event
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

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
            self.status_var.set(f"Error: Could not read image")
            return

        self.current_image = img
        self.source_type = "image"
        self.status_var.set("Processing...")
        self.root.update()

        # Run pipeline
        self._process_and_display(img)

    def load_video(self):
        """Load a video file (placeholder for future implementation)."""
        # TODO: Implement video loading
        pass

    def toggle_camera(self):
        """Toggle camera input (placeholder for future implementation)."""
        # TODO: Implement camera toggle
        pass

    def _process_and_display(self, img):
        """Run the pipeline and display the result."""
        try:
            # Get shared config parameters
            cfg = get_color_config(self.config, "blue")  # Use shared settings

            # Extract parameters from config
            rg = cfg.get("remove_green", {})
            bz = cfg.get("binarize", {})
            op = cfg.get("opening", {})
            bc = cfg.get("big_contours", {})
            ip = cfg.get("inner_parts", {})
            lb = cfg.get("labeled", {})

            # Run the pipeline
            result = process_piece(
                img,
                debug=False,
                threshold_value=bz.get("thresh_value", 0),
                morph_kernel_size=(op.get("kernel_size", 5), op.get("kernel_size", 5)),
                morph_iterations=op.get("iterations", 1),
                main_min_area=bc.get("min_area", 10000),
                main_max_area=bc.get("max_area", 500000),
                small_area_max=bc.get("small_max", 30000),
                medium_area_max=bc.get("medium_max", 90000),
                inner_min_area=ip.get("min_area", 50),
                inner_max_area=ip.get("max_area", 10000),
                extent_threshold=lb.get("extent_threshold", 0.5),
                label_small_area_max=lb.get("small_area_max", 2100),
                label_medium_area_max=lb.get("medium_area_max", 5600),
            )

            self.current_result = result

            # Update result display
            self._update_result_display(result)

            # Display annotated image
            self._display_image(result.annotated_image)

            self.status_var.set("Processing complete")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self._clear_result_display()
            # Still show original image
            if img is not None:
                self._display_image(img)

    def _update_result_display(self, result: PipelineResult):
        """Update the result info panel with detection results."""
        # Color
        self.color_var.set(result.piece_color.upper())
        color_map = {
            "blue": "#2563eb",
            "red": "#dc2626",
            "yellow": "#ca8a04",
        }
        fg = color_map.get(result.piece_color.lower(), "gray")
        self.color_label.config(foreground=fg)

        # Size
        self.size_var.set(result.size_label)

        # Value
        if result.total_value is not None:
            self.value_var.set(str(result.total_value))
        else:
            self.value_var.set("—")

        # Status
        if result.is_consistent:
            self.status_result_var.set("OK")
            self.status_result_label.config(foreground="green")
        else:
            self.status_result_var.set("MISMATCH")
            self.status_result_label.config(foreground="orange")

    def _clear_result_display(self):
        """Clear the result info panel."""
        self.color_var.set("—")
        self.color_label.config(foreground="gray")
        self.size_var.set("—")
        self.value_var.set("—")
        self.status_result_var.set("—")
        self.status_result_label.config(foreground="gray")

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
            # Canvas not ready yet, schedule retry
            self.root.after(50, lambda: self._display_image(img_bgr))
            return

        # Scale image to fit canvas while maintaining aspect ratio
        img_h, img_w = img_rgb.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        if new_w > 0 and new_h > 0:
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img_rgb

        # Convert to PIL and then to PhotoImage
        pil_img = Image.fromarray(img_resized)
        self._photo = ImageTk.PhotoImage(pil_img)

        # Clear canvas and draw image centered
        self.canvas.delete("all")
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self._photo)

    def _on_canvas_resize(self, event):
        """Handle canvas resize to update image display."""
        if self.current_result is not None:
            self._display_image(self.current_result.annotated_image)
        elif self.current_image is not None:
            self._display_image(self.current_image)

    def on_close(self):
        """Handle window close."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


def run_result_viewer(config_path=DEFAULT_CONFIG_PATH):
    """
    Launch the result viewer GUI.

    Parameters
    ----------
    config_path : str
        Path to configuration JSON file.

    Example
    -------
    >>> from stb600_lib.gui import run_result_viewer
    >>> run_result_viewer()
    """
    root = tk.Tk()
    app = ResultViewerApp(root, config_path=config_path)
    root.mainloop()

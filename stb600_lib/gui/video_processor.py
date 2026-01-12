"""
Video processor GUI for detecting and counting pieces.
Supports video files and live camera feeds with object tracking.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Dict, Tuple
import time


from pypylon import pylon

from ..tracking import (
    NORFAIR_AVAILABLE,
    create_tracker,
    detection_to_norfair,
    CountingLine,
    PieceCounter,
    TrackedPiece,
)
from ..detection import detect_pieces_in_frame, get_detection_ranges
from ..display import draw_piece_box
from .config import load_config, get_color_config
from .result_viewer import process_single_contour


class VideoProcessorApp:
    """
    GUI application for processing video/camera feed with piece tracking.
    Detects pieces, tracks them across frames, and counts unique pieces.
    """

    def __init__(self, root: tk.Tk):
        """Initialize the video processor application."""
        self.root = root
        self.root.title("Piece Counter - Video Processor")
        self.root.geometry("1200x800")

        # Check Norfair availability
        if not NORFAIR_AVAILABLE:
            messagebox.showerror(
                "Missing Dependency",
                "Norfair is not installed.\nInstall with: pip install norfair"
            )
            root.destroy()
            return

        # Load config
        self.config = load_config()

        # Video state
        # For video files, use cv2.VideoCapture
        self.video_capture: Optional[cv2.VideoCapture] = None
        # For live Basler camera
        self.basler_camera: Optional[pylon.InstantCamera] = None
        self.basler_converter: Optional[pylon.ImageFormatConverter] = None

        self.is_playing = False
        self.is_camera = False
        self.current_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.fps = 30

        # Tracking state
        self.tracker = None
        self.counting_line: Optional[CountingLine] = None
        self.piece_counter = PieceCounter()
        # Per-ID cached analysis result so we do not re-analyze
        # track_id -> dict with piece_color, size_label, value, is_consistent, bounding_box, contour
        self.analysis_results: Dict[int, dict] = {}

        # Processing settings
        self.process_every_n = 3  # analysis is event-triggered now, so keep 1
        self._frame_index = 0

        self.line_position = tk.DoubleVar(value=0.5)  # Centered
        self.line_direction = tk.StringVar(value="horizontal")

        # Frame dimensions (for counting line updates before video loaded)
        self.frame_width = 640
        self.frame_height = 480

        # Runtime FPS measurement
        self._last_frame_time = None
        self.runtime_fps = 0.0
        self.fps_var = tk.StringVar(value="FPS: 0.0")

        # Display resolution (only GUI)
        self.display_width = 640

        self._setup_ui()
        self._init_tracker()

        # Initialize counting line with default dimensions
        self._init_counting_line(self.frame_width, self.frame_height)

    def _init_tracker(self):
        """Initialize the Norfair tracker."""
        # Pure ID-based tracking: Norfair uses only positions + simple distance.
        self.tracker = create_tracker(
            distance_threshold=150.0,
            hit_counter_max=20,
            initialization_delay=2,
        )

    def _setup_ui(self):
        """Set up the user interface."""
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            controls_frame, text="📁 Load Video",
            command=self._load_video
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            controls_frame, text="📷 Start Camera",
            command=self._start_camera
        ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(controls_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )

        self.play_btn = ttk.Button(
            controls_frame, text="▶ Play",
            command=self._toggle_play
        )
        self.play_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(
            controls_frame, text="⏹ Stop",
            command=self._stop
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            controls_frame, text="🔄 Reset Count",
            command=self._reset_count
        ).pack(side=tk.LEFT, padx=2)

        self.status_var = tk.StringVar(value="Ready - Load a video or start camera")
        ttk.Label(
            controls_frame, textvariable=self.status_var
        ).pack(side=tk.RIGHT, padx=10)

        ttk.Label(
            controls_frame, textvariable=self.fps_var
        ).pack(side=tk.RIGHT, padx=10)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        video_frame = ttk.LabelFrame(content_frame, text="Video Feed", padding=5)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        results_frame = ttk.Frame(content_frame, width=300)
        results_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        results_frame.pack_propagate(False)

        total_frame = ttk.LabelFrame(results_frame, text="Total Value", padding=10)
        total_frame.pack(fill=tk.X, pady=(0, 10))

        self.total_var = tk.StringVar(value="0")
        ttk.Label(
            total_frame, textvariable=self.total_var,
            font=("Helvetica", 48, "bold"),
            foreground="#2ecc71"
        ).pack()

        self.pieces_count_var = tk.StringVar(value="Pieces: 0")
        ttk.Label(
            total_frame, textvariable=self.pieces_count_var,
            font=("Helvetica", 14)
        ).pack()

        list_frame = ttk.LabelFrame(results_frame, text="Counted Pieces", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("id", "color", "size", "value")
        self.pieces_tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", height=15
        )
        self.pieces_tree.heading("id", text="#")
        self.pieces_tree.heading("color", text="Color")
        self.pieces_tree.heading("size", text="Size")
        self.pieces_tree.heading("value", text="Value")

        self.pieces_tree.column("id", width=30)
        self.pieces_tree.column("color", width=60)
        self.pieces_tree.column("size", width=70)
        self.pieces_tree.column("value", width=50)

        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL,
            command=self.pieces_tree.yview
        )
        self.pieces_tree.configure(yscrollcommand=scrollbar.set)

        self.pieces_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        summary_frame = ttk.LabelFrame(results_frame, text="Summary", padding=5)
        summary_frame.pack(fill=tk.X, pady=(10, 0))

        self.summary_text = tk.Text(
            summary_frame, height=4, width=30,
            state="disabled", font=("Consolas", 9)
        )
        self.summary_text.pack(fill=tk.X)

    def _load_video(self):
        """Load a video file."""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return

        self._stop()
        self.is_camera = False

        self.video_capture = cv2.VideoCapture(file_path)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Could not open video file")
            return

        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
        self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, frame = self.video_capture.read()
        if ret:
            h, w = frame.shape[:2]
            self._init_counting_line(w, h)
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._reset_tracking()
        self.status_var.set(f"Loaded: {file_path.split('/')[-1]} ({self.frame_count} frames)")
        self._last_frame_time = None
        self.runtime_fps = 0.0
        self.fps_var.set("FPS: 0.0")

    # --- Basler camera helpers -------------------------------------------------

    def _start_basler_camera(self) -> bool:
        """
        Initialize and start grabbing from the Basler camera.
        Returns True if successful, False otherwise.
        """
        try:
            self.basler_camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self.basler_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)  # continuous grabbing [web:45]

            self.basler_converter = pylon.ImageFormatConverter()
            self.basler_converter.OutputPixelFormat = pylon.PixelType_BGR8packed  # OpenCV BGR format [web:6]
            self.basler_converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self.fps = 30  # You can tune this based on camera configuration
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not open Basler camera:\n{e}")
            self.basler_camera = None
            self.basler_converter = None
            return False

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

    # ---------------------------------------------------------------------------

    def _start_camera(self):
        """Start camera feed (Basler)."""
        self._stop()
        self.is_camera = True

        # Use Basler instead of cv2.VideoCapture(0)
        if not self._start_basler_camera():
            # If Basler fails, do not start playing
            self.is_camera = False
            return

        # Try to grab one frame to initialize line
        init_frame = self._grab_camera_frame()
        if init_frame is not None:
            h, w = init_frame.shape[:2]
            self._init_counting_line(w, h)

        self._reset_tracking()
        self.status_var.set("Camera ready")

        self._last_frame_time = None
        self.runtime_fps = 0.0
        self.fps_var.set("FPS: 0.0")

        self._toggle_play()

    def _grab_camera_frame(self) -> Optional[np.ndarray]:
        """
        Grab a single frame from the active source:
        - If video file: from cv2.VideoCapture
        - If camera: from Basler via pypylon
        """
        if self.is_camera:
            if self.basler_camera is None or self.basler_converter is None:
                return None
            if not self.basler_camera.IsGrabbing():
                return None

            try:
                grab_result = self.basler_camera.RetrieveResult(
                    5000, pylon.TimeoutHandling_ThrowException
                )  # blocking wait [web:6]
            except Exception:
                return None

            frame = None
            if grab_result is not None:
                if grab_result.GrabSucceeded():
                    image = self.basler_converter.Convert(grab_result)
                    frame = image.GetArray()
                grab_result.Release()
            return frame
        else:
            # Video file path
            if self.video_capture is None:
                return None
            ret, frame = self.video_capture.read()
            if not ret:
                return None
            return frame

    def _init_counting_line(self, width: int, height: int):
        """Initialize the counting line."""
        self.frame_width = width
        self.frame_height = height
        self.counting_line = CountingLine(
            position=self.line_position.get(),
            direction=self.line_direction.get(),
            frame_width=width,
            frame_height=height,
        )

    def _reset_tracking(self):
        """Reset tracker and counter."""
        self._init_tracker()
        self.piece_counter.reset()
        self.analysis_results.clear()
        self._update_results_display()
        self._frame_index = 0

    def _reset_count(self):
        """Reset just the count (keep video)."""
        self._reset_tracking()
        self.status_var.set("Count reset")

    def _toggle_play(self):
        """Toggle play/pause."""
        # For camera mode, we rely on Basler; for video, on cv2.VideoCapture
        if not self.is_camera and self.video_capture is None:
            return

        self.is_playing = not self.is_playing
        self.play_btn.config(text="⏸ Pause" if self.is_playing else "▶ Play")

        if self.is_playing:
            self._last_frame_time = time.perf_counter()
            self._process_loop()

    def _stop(self):
        """Stop playback and release video/camera."""
        self.is_playing = False
        self.play_btn.config(text="▶ Play")

        # Release video file
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

        # Stop Basler camera if active
        self._stop_basler_camera()
        self.is_camera = False

        self._last_frame_time = None
        self.runtime_fps = 0.0
        self.fps_var.set("FPS: 0.0")

    def _process_loop(self):
        """Main processing loop for video frames."""
        if not self.is_playing:
            return

        now = time.perf_counter()
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 0:
                self.runtime_fps = 1.0 / dt
                self.fps_var.set(f"FPS: {self.runtime_fps:.1f}")
        self._last_frame_time = now

        frame = self._grab_camera_frame()
        if frame is None:
            if not self.is_camera and self.video_capture is not None:
                # End of video file
                self.is_playing = False
                self.play_btn.config(text="▶ Play")
                self.status_var.set("Video finished")
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        self.current_frame = frame
        self._frame_index += 1

        processed_frame = self._process_frame(frame)
        self._display_frame(processed_frame)

        delay = int(1000 / self.fps) if self.fps > 0 else 33
        self.root.after(delay, self._process_loop)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect centroids, track IDs, and
        trigger one-time analysis on line touch/cross.
        """
        # --- FAST STAGE: detection + ID-based tracking only ---

        cfg = get_color_config(self.config, "blue")
        roi_config = self.config.get("rois", {})

        contours, _, _ = detect_pieces_in_frame(
            frame, cfg, auto_invert=True
        )

        detections = []

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            x, y, w, h = cv2.boundingRect(contour)

            det = detection_to_norfair(
                centroid=(cx, cy),
                data={
                    "bounding_box": (x, y, w, h),
                    "contour": contour,
                },
            )
            detections.append(det)

        tracked_objects = self.tracker.update(detections=detections)

        out_frame = frame.copy()

        # --- SLOW STAGE: only when line is touched/crossed, and only once per ID ---

        ranges_cached = False
        red_range = yellow_range = blue_range = None
        ip = cfg.get("inner_parts", {})
        lb = cfg.get("labeled", {})

        for obj in tracked_objects:
            track_id = obj.id

            if not obj.last_detection or not obj.last_detection.data:
                continue

            data = obj.last_detection.data
            cx, cy = int(obj.estimate[0][0]), int(obj.estimate[0][1])
            bbox = data["bounding_box"]
            contour = data["contour"]

            self.piece_counter.update_track_history(track_id, (cx, cy))

            if self.piece_counter.is_counted(track_id):
                analysis = self.analysis_results.get(track_id)
                if analysis is not None:
                    self._draw_piece(out_frame, bbox, analysis)
                continue

            crossed_or_touched = False
            if self.counting_line:
                prev_centroid = self.piece_counter.get_previous_centroid(track_id)
                crossed_geom = self.counting_line.has_crossed(prev_centroid, (cx, cy))

                touched_now = False
                if self.counting_line.direction == "horizontal":
                    y_line = int(self.counting_line.position * self.frame_height)
                    touched_now = abs(cy - y_line) <= 2
                else:
                    x_line = int(self.counting_line.position * self.frame_width)
                    touched_now = abs(cx - x_line) <= 2

                crossed_or_touched = crossed_geom or touched_now

            if crossed_or_touched:
                if not ranges_cached:
                    red_range, yellow_range, blue_range, _, _ = get_detection_ranges(cfg)
                    ranges_cached = True

                try:
                    result = process_single_contour(
                        frame,
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
                except Exception:
                    result = None

                if result is not None:
                    analysis = {
                        "piece_color": result.piece_color or "unknown",
                        "size_label": result.size_label,
                        "value": result.total_value,
                        "is_consistent": result.is_consistent,
                        "bounding_box": bbox,
                    }
                else:
                    analysis = {
                        "piece_color": "unknown",
                        "size_label": "",
                        "value": None,
                        "is_consistent": False,
                        "bounding_box": bbox,
                    }

                # Cache the analysis result for this ID
                self.analysis_results[track_id] = analysis

                # Ensure fake parts do not contribute to total value
                value_for_total = analysis["value"]
                if not analysis["is_consistent"]:
                    value_for_total = None  # or 0, depending on PieceCounter.sum implementation

                piece = TrackedPiece(
                    track_id=track_id,
                    piece_color=analysis["piece_color"],
                    size_label=analysis["size_label"],
                    value=value_for_total,
                    is_consistent=analysis["is_consistent"],
                    centroid=(cx, cy),
                    bounding_box=bbox,
                    crossed_line=True,
                    counted=True,
                )
                self.piece_counter.add_piece(piece)
                self._update_results_display()

                self._draw_piece(out_frame, bbox, analysis)
            else:
                cached = self.analysis_results.get(track_id)
                if cached is not None:
                    self._draw_piece(out_frame, bbox, cached)
                else:
                    temp_analysis = {
                        "piece_color": "unknown",
                        "size_label": "",
                        "value": None,
                        "is_consistent": True,
                    }
                    self._draw_piece(out_frame, bbox, temp_analysis)

        if self.counting_line:
            start, end = self.counting_line.get_line_coords()
            cv2.line(out_frame, start, end, (0, 255, 255), 2)

        return out_frame

    def _draw_piece(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], analysis: dict):
        """Draw a piece bounding box + label using shared draw function."""
        x, y, w, h = bbox

        if analysis["is_consistent"]:
            color_name = analysis["piece_color"].upper()
        else:
            color_name = "FAKE"

        if analysis["value"] is not None:
            label = f"{color_name} - {analysis['value']}"
        else:
            label = color_name

        draw_piece_box(
            frame,
            bounding_box=(x, y, w, h),
            piece_color=analysis["piece_color"],
            label_text=label,
            is_consistent=analysis["is_consistent"],
            is_counted=self.piece_counter.is_counted_for_color(analysis["piece_color"])
            if hasattr(self.piece_counter, "is_counted_for_color") else False,
            draw_center=True,
            font_scale=1.1,   # increased label size
            thickness=2,      # thicker text stroke
            box_thickness=2,
            center_radius=4,
        )

    def _display_frame(self, frame: np.ndarray):
        """Display frame in the GUI."""
        h, w = frame.shape[:2]
        scale = self.display_width / w
        new_h = int(h * scale)

        frame_resized = cv2.resize(frame, (self.display_width, new_h))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(img)

        self.video_label.config(image=photo)
        self.video_label.image = photo

    def _update_results_display(self):
        """Update the results display panel."""
        self.total_var.set(str(self.piece_counter.total_value))
        self.pieces_count_var.set(f"Pieces: {len(self.piece_counter.counted_pieces)}")

        self.pieces_tree.delete(*self.pieces_tree.get_children())
        for i, piece in enumerate(self.piece_counter.counted_pieces):
            value_str = str(piece.value) if piece.value else "?"
            self.pieces_tree.insert(
                "", tk.END,
                values=(i + 1, piece.piece_color, piece.size_label, value_str)
            )

        if self.pieces_tree.get_children():
            self.pieces_tree.see(self.pieces_tree.get_children()[-1])

        summary = self.piece_counter.get_summary()
        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", tk.END)

        by_color = summary.get("by_color", {})
        if by_color:
            self.summary_text.insert(tk.END, "By Color:\n")
            for color, count in by_color.items():
                self.summary_text.insert(tk.END, f"  {color}: {count}\n")

        self.summary_text.config(state="disabled")

    def run(self):
        """Start the application main loop."""
        self.root.mainloop()


def run_video_processor():
    """Launch the video processor GUI."""
    root = tk.Tk()
    app = VideoProcessorApp(root)
    app.run()


if __name__ == "__main__":
    run_video_processor()

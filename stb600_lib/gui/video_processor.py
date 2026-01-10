"""
Video processor GUI for detecting and counting LEGO pieces.
Supports video files and live camera feeds with object tracking.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, List, Dict, Tuple
import threading
import time

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
        self.root.title("LEGO Piece Counter - Video Processor")
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
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.is_playing = False
        self.is_camera = False
        self.current_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.fps = 30
        
        # Tracking state
        self.tracker = None
        self.counting_line: Optional[CountingLine] = None
        self.piece_counter = PieceCounter()
        self.active_tracks: Dict[int, dict] = {}  # track_id -> piece info
        
        # Processing settings
        self.process_every_n = 1  # Process every N frames
        self.line_position = tk.DoubleVar(value=0.5)  # Centered
        self.line_direction = tk.StringVar(value="horizontal")
        
        # Frame dimensions (for counting line updates before video loaded)
        self.frame_width = 640
        self.frame_height = 480
        
        self._setup_ui()
        self._init_tracker()
        
        # Initialize counting line with default dimensions
        self._init_counting_line(self.frame_width, self.frame_height)
    
    def _init_tracker(self):
        """Initialize the Norfair tracker."""
        self.tracker = create_tracker(
            distance_threshold=150.0,
            hit_counter_max=20,
            initialization_delay=2,
        )
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Source buttons
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
        
        # Playback controls
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
        
        
        # Status label
        self.status_var = tk.StringVar(value="Ready - Load a video or start camera")
        ttk.Label(
            controls_frame, textvariable=self.status_var
        ).pack(side=tk.RIGHT, padx=10)
        
        # Content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video display (left)
        video_frame = ttk.LabelFrame(content_frame, text="Video Feed", padding=5)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Results panel (right)
        results_frame = ttk.Frame(content_frame, width=300)
        results_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        results_frame.pack_propagate(False)
        
        # Total value display
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
        
        # Counted pieces list
        list_frame = ttk.LabelFrame(results_frame, text="Counted Pieces", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for pieces
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
        
        # Summary frame
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
        
        # Read first frame to get dimensions
        ret, frame = self.video_capture.read()
        if ret:
            h, w = frame.shape[:2]
            self._init_counting_line(w, h)
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self._reset_tracking()
        self.status_var.set(f"Loaded: {file_path.split('/')[-1]} ({self.frame_count} frames)")
    
    def _start_camera(self):
        """Start camera feed."""
        self._stop()
        self.is_camera = True
        
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
        
        self.fps = 30
        
        # Read first frame to get dimensions
        ret, frame = self.video_capture.read()
        if ret:
            h, w = frame.shape[:2]
            self._init_counting_line(w, h)
        
        self._reset_tracking()
        self.status_var.set("Camera ready")
        
        # Auto-start for camera
        self._toggle_play()
    
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
        self.active_tracks.clear()
        self._update_results_display()
    
    def _reset_count(self):
        """Reset just the count (keep video)."""
        self._reset_tracking()
        self.status_var.set("Count reset")
    
    def _toggle_play(self):
        """Toggle play/pause."""
        if self.video_capture is None:
            return
        
        self.is_playing = not self.is_playing
        self.play_btn.config(text="⏸ Pause" if self.is_playing else "▶ Play")
        
        if self.is_playing:
            self._process_loop()
    
    def _stop(self):
        """Stop playback and release video."""
        self.is_playing = False
        self.play_btn.config(text="▶ Play")
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
    
    def _process_loop(self):
        """Main processing loop for video frames."""
        if not self.is_playing or self.video_capture is None:
            return
        
        ret, frame = self.video_capture.read()
        if not ret:
            if not self.is_camera:
                # End of video
                self.is_playing = False
                self.play_btn.config(text="▶ Play")
                self.status_var.set("Video finished")
                # Reset to beginning
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        
        self.current_frame = frame
        
        # Process frame
        processed_frame = self._process_frame(frame)
        
        # Display
        self._display_frame(processed_frame)
        
        # Schedule next frame
        delay = int(1000 / self.fps)
        self.root.after(delay, self._process_loop)
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect pieces, track, count.
        Uses the same pipeline as result_viewer for consistency.
        
        Args:
            frame: BGR image frame
        
        Returns:
            Annotated frame
        """
        # Get config and use shared detection pipeline
        cfg = get_color_config(self.config, "blue")
        
        ip = cfg.get("inner_parts", {})
        lb = cfg.get("labeled", {})
        
        # Get ROI config (stored at root level, not under "shared")
        roi_config = self.config.get("rois", {})
        
        # Use shared detection function (same as result_viewer)
        contours, _, _ = detect_pieces_in_frame(
            frame, cfg, auto_invert=True
        )
        
        # Extract detection ranges
        red_range, yellow_range, blue_range, _, _ = get_detection_ranges(cfg)
        
        # Process each contour using the same function as result_viewer
        detections = []
        
        for contour in contours:
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Process using result_viewer's process_single_contour
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
                
                if result is None:
                    continue
                
                # Create Norfair detection from PieceResult
                det = detection_to_norfair(
                    centroid=(cx, cy),
                    data={
                        "piece_color": result.piece_color or "unknown",
                        "size_label": result.size_label,
                        "value": result.total_value,
                        "is_consistent": result.is_consistent,
                        "bounding_box": (x, y, w, h),
                        "contour": contour,
                    }
                )
                detections.append(det)
                
            except Exception:
                continue
        
        # Update tracker
        tracked_objects = self.tracker.update(detections=detections)
        
        # Process tracked objects
        out_frame = frame.copy()
        
        for obj in tracked_objects:
            track_id = obj.id
            
            # Get detection data
            if obj.last_detection and obj.last_detection.data:
                data = obj.last_detection.data
            else:
                continue
            
            cx, cy = int(obj.estimate[0][0]), int(obj.estimate[0][1])
            
            # Update track history
            self.piece_counter.update_track_history(track_id, (cx, cy))
            
            # Check if crossed counting line
            if self.counting_line and not self.piece_counter.is_counted(track_id):
                prev_centroid = self.piece_counter.get_previous_centroid(track_id)
                if self.counting_line.has_crossed(prev_centroid, (cx, cy)):
                    # Count this piece
                    piece = TrackedPiece(
                        track_id=track_id,
                        piece_color=data["piece_color"],
                        size_label=data["size_label"],
                        value=data["value"],
                        is_consistent=data["is_consistent"],
                        centroid=(cx, cy),
                        bounding_box=data["bounding_box"],
                        crossed_line=True,
                        counted=True,
                    )
                    self.piece_counter.add_piece(piece)
                    self._update_results_display()
            
            # Draw on frame using shared drawing function
            x, y, w, h = data["bounding_box"]
            
            # Generate label: "COLOR - value" or "FAKE - value"
            if data["is_consistent"]:
                color_name = data["piece_color"].upper()
            else:
                color_name = "FAKE"
            
            if data["value"] is not None:
                label = f"{color_name} - {data['value']}"
            else:
                label = color_name
            
            # Use shared drawing function (is_counted=False to keep original color)
            draw_piece_box(
                out_frame,
                bounding_box=(x, y, w, h),
                piece_color=data["piece_color"],
                label_text=label,
                is_consistent=data["is_consistent"],
                is_counted=False,
                draw_center=True,
                font_scale=0.8,
                thickness=2,
                box_thickness=3,
                center_radius=6,
            )
        
        # Draw counting line (no text label)
        if self.counting_line:
            start, end = self.counting_line.get_line_coords()
            cv2.line(out_frame, start, end, (0, 255, 255), 2)
        
        return out_frame
    
    def _display_frame(self, frame: np.ndarray):
        """Display frame in the GUI."""
        # Resize to fit display
        display_width = 800
        h, w = frame.shape[:2]
        scale = display_width / w
        new_h = int(h * scale)
        
        frame_resized = cv2.resize(frame, (display_width, new_h))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and then to PhotoImage
        img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(img)
        
        self.video_label.config(image=photo)
        self.video_label.image = photo  # Keep reference
    
    def _update_results_display(self):
        """Update the results display panel."""
        # Update total
        self.total_var.set(str(self.piece_counter.total_value))
        self.pieces_count_var.set(f"Pieces: {len(self.piece_counter.counted_pieces)}")
        
        # Update pieces list
        self.pieces_tree.delete(*self.pieces_tree.get_children())
        for i, piece in enumerate(self.piece_counter.counted_pieces):
            value_str = str(piece.value) if piece.value else "?"
            self.pieces_tree.insert(
                "", tk.END,
                values=(i + 1, piece.piece_color, piece.size_label, value_str)
            )
        
        # Auto-scroll to bottom
        if self.pieces_tree.get_children():
            self.pieces_tree.see(self.pieces_tree.get_children()[-1])
        
        # Update summary
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

"""
Object tracking utilities for video processing.
Uses Norfair library for multi-object tracking.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

try:
    import norfair
    from norfair import Detection, Tracker
    NORFAIR_AVAILABLE = True
except ImportError:
    NORFAIR_AVAILABLE = False


@dataclass
class TrackedPiece:
    """Represents a tracked piece with its properties."""
    track_id: int
    piece_color: str
    size_label: str
    value: Optional[int]
    is_consistent: bool
    centroid: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    crossed_line: bool = False
    counted: bool = False


def create_tracker(
    distance_threshold: float = 100.0,
    hit_counter_max: int = 15,
    initialization_delay: int = 3,
    pointwise_hit_counter_max: int = 4,
) -> "Tracker":
    """
    Create a Norfair tracker configured for conveyor belt tracking.
    
    Args:
        distance_threshold: Max distance to associate detections with tracks
        hit_counter_max: Frames to keep track alive without detections
        initialization_delay: Frames before track is confirmed
        pointwise_hit_counter_max: Hit counter for individual points
    
    Returns:
        Configured Norfair Tracker instance
    """
    if not NORFAIR_AVAILABLE:
        raise ImportError(
            "Norfair is not installed. Install with: pip install norfair"
        )
    
    return Tracker(
        distance_function="euclidean",
        distance_threshold=distance_threshold,
        hit_counter_max=hit_counter_max,
        initialization_delay=initialization_delay,
        pointwise_hit_counter_max=pointwise_hit_counter_max,
    )


def detection_to_norfair(
    centroid: Tuple[int, int],
    scores: Optional[List[float]] = None,
    data: Optional[dict] = None,
) -> "Detection":
    """
    Convert a detection to Norfair Detection format.
    
    Args:
        centroid: (cx, cy) center point of detection
        scores: Optional confidence scores
        data: Optional metadata to attach to detection
    
    Returns:
        Norfair Detection object
    """
    if not NORFAIR_AVAILABLE:
        raise ImportError("Norfair is not installed")
    
    points = np.array([[centroid[0], centroid[1]]])
    scores_arr = np.array(scores) if scores else np.array([1.0])
    
    return Detection(points=points, scores=scores_arr, data=data)


class CountingLine:
    """
    Manages a counting line for tracking when objects cross a threshold.
    Objects are counted when their centroid crosses this line.
    """
    
    def __init__(
        self,
        position: float = 0.7,
        direction: str = "horizontal",
        frame_width: int = 640,
        frame_height: int = 480,
    ):
        """
        Initialize counting line.
        
        Args:
            position: Relative position (0-1) along the axis
            direction: "horizontal" or "vertical"
            frame_width: Width of video frame
            frame_height: Height of video frame
        """
        self.position = position
        self.direction = direction
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._update_line()
    
    def _update_line(self):
        """Update line coordinates based on position and direction."""
        if self.direction == "horizontal":
            self.line_coord = int(self.frame_height * self.position)
        else:
            self.line_coord = int(self.frame_width * self.position)
    
    def update_frame_size(self, width: int, height: int):
        """Update frame dimensions and recalculate line position."""
        self.frame_width = width
        self.frame_height = height
        self._update_line()
    
    def has_crossed(
        self,
        prev_centroid: Optional[Tuple[int, int]],
        curr_centroid: Tuple[int, int],
    ) -> bool:
        """
        Check if object crossed the counting line.
        
        Args:
            prev_centroid: Previous frame centroid (None if new track)
            curr_centroid: Current frame centroid
        
        Returns:
            True if object crossed the line this frame
        """
        if prev_centroid is None:
            return False
        
        if self.direction == "horizontal":
            # Check if crossed from above to below (or vice versa)
            prev_y, curr_y = prev_centroid[1], curr_centroid[1]
            return (prev_y < self.line_coord <= curr_y) or \
                   (prev_y > self.line_coord >= curr_y)
        else:
            # Check if crossed from left to right (or vice versa)
            prev_x, curr_x = prev_centroid[0], curr_centroid[0]
            return (prev_x < self.line_coord <= curr_x) or \
                   (prev_x > self.line_coord >= curr_x)
    
    def get_line_coords(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get start and end points of the counting line for drawing."""
        if self.direction == "horizontal":
            return (0, self.line_coord), (self.frame_width, self.line_coord)
        else:
            return (self.line_coord, 0), (self.line_coord, self.frame_height)


class PieceCounter:
    """
    Manages counting of unique pieces as they cross the counting line.
    Stores history of counted pieces and running total.
    """
    
    def __init__(self):
        """Initialize the piece counter."""
        self.counted_pieces: List[TrackedPiece] = []
        self.total_value: int = 0
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}  # track_id -> centroid history
    
    def reset(self):
        """Reset all counters and history."""
        self.counted_pieces.clear()
        self.total_value = 0
        self.track_history.clear()
    
    def update_track_history(self, track_id: int, centroid: Tuple[int, int]):
        """Update centroid history for a track."""
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        self.track_history[track_id].append(centroid)
        # Keep only last 10 positions
        if len(self.track_history[track_id]) > 10:
            self.track_history[track_id].pop(0)
    
    def get_previous_centroid(self, track_id: int) -> Optional[Tuple[int, int]]:
        """Get previous centroid for a track."""
        history = self.track_history.get(track_id, [])
        if len(history) >= 2:
            return history[-2]
        return None
    
    def add_piece(self, piece: TrackedPiece):
        """Add a counted piece and update total."""
        piece.counted = True
        self.counted_pieces.append(piece)
        if piece.value is not None:
            self.total_value += piece.value
    
    def is_counted(self, track_id: int) -> bool:
        """Check if a track has already been counted."""
        return any(p.track_id == track_id for p in self.counted_pieces)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "total_pieces": len(self.counted_pieces),
            "total_value": self.total_value,
            "by_color": self._count_by_color(),
            "by_size": self._count_by_size(),
        }
    
    def _count_by_color(self) -> Dict[str, int]:
        """Count pieces by color."""
        counts = {}
        for piece in self.counted_pieces:
            counts[piece.piece_color] = counts.get(piece.piece_color, 0) + 1
        return counts
    
    def _count_by_size(self) -> Dict[str, int]:
        """Count pieces by size."""
        counts = {}
        for piece in self.counted_pieces:
            counts[piece.size_label] = counts.get(piece.size_label, 0) + 1
        return counts

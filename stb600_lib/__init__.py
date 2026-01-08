"""
STB600 Image Processing Library
===============================

A library for detecting and analyzing colored pieces with encoded values
using computer vision techniques.

Modules:
    - io: Image/video/camera loading utilities
    - display: Visualization and annotation functions
    - color: HSV color processing and detection
    - morphology: Morphological operations (opening, closing, binarization)
    - contours: Contour detection, features, and labeling
    - roi: Region of Interest computation and counting
    - decoding: Value decoding from ROI counts
    - transforms: Image rotation utilities
    - pipeline: High-level orchestration functions
"""

# IO functions
from .io import load_source_cv2

# Display functions
from .display import (
    show_image,
    annotate_part_result,
    draw_labeled_contours_colored,
    draw_horizontal_rois_from_boundaries,
    resize_to_screen,
    pad_to_same_size,
    make_row,
    make_grid,
)

# Color processing functions
from .color import (
    remove_color_hsv,
    detect_piece_color_and_check_size,
    detect_color_parts,
)

# Morphology functions
from .morphology import (
    apply_morphological_closing,
    apply_morphological_opening,
    binarize_and_invert,
)

# Contour functions
from .contours import (
    find_and_draw_contours_with_area_limits,
    extract_contour_features,
    label_contour_by_extent_and_area,
    crop_and_align_vertical,
)

# ROI functions
from .roi import (
    split_fixed_horizontal_rois_and_count,
)

# Decoding functions
from .decoding import (
    decode_roi_to_number,
    digits_to_int,
    compute_total_value_from_rois,
)

# Transform functions
from .transforms import (
    rotate_if_marker_bottom,
    rotate_if_blue_piece_upside_down,
)

# Pipeline functions
from .pipeline import process_piece

__version__ = "1.0.0"
__all__ = [
    # IO
    "load_source_cv2",
    # Display
    "show_image",
    "annotate_part_result",
    "draw_labeled_contours_colored",
    "draw_horizontal_rois_from_boundaries",
    "resize_to_screen",
    "pad_to_same_size",
    "make_row",
    "make_grid",
    # Color
    "remove_color_hsv",
    "detect_piece_color_and_check_size",
    "detect_color_parts",
    # Morphology
    "apply_morphological_closing",
    "apply_morphological_opening",
    "binarize_and_invert",
    # Contours
    "find_and_draw_contours_with_area_limits",
    "extract_contour_features",
    "label_contour_by_extent_and_area",
    "crop_and_align_vertical",
    # ROI
    "split_fixed_horizontal_rois_and_count",
    # Decoding
    "decode_roi_to_number",
    "digits_to_int",
    "compute_total_value_from_rois",
    # Transforms
    "rotate_if_marker_bottom",
    "rotate_if_blue_piece_upside_down",
    # Pipeline
    "process_piece",
]

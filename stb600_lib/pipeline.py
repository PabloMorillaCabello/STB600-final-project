"""
Pipeline Module - High-Level Orchestration
==========================================

High-level functions that combine all processing steps into a complete pipeline.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .io import load_source_cv2
from .color import remove_color_hsv, detect_piece_color_and_check_size, detect_color_parts
from .morphology import binarize_and_invert, apply_morphological_opening
from .contours import (
    find_and_draw_contours_with_area_limits,
    extract_contour_features,
    label_contour_by_extent_and_area,
    crop_and_align_vertical,
)
from .display import (
    show_image,
    draw_labeled_contours_colored,
    draw_horizontal_rois_from_boundaries,
    annotate_part_result,
)
from .roi import split_fixed_horizontal_rois_and_count
from .decoding import decode_roi_to_number, compute_total_value_from_rois
from .transforms import rotate_if_marker_bottom, rotate_if_blue_piece_upside_down


@dataclass
class PipelineResult:
    """Container for pipeline processing results."""
    # Input
    original_image: np.ndarray

    # Detection results
    piece_color: str
    size_label: str
    is_consistent: bool
    total_value: Optional[int]

    # Intermediate images
    cropped_part: np.ndarray
    labeled_part: np.ndarray
    roi_visualization: np.ndarray
    annotated_image: np.ndarray

    # Detailed data
    main_contour: np.ndarray
    features_list: List[Dict[str, Any]]
    rois: List[Dict[str, Any]]
    decoded_digits: List[Optional[int]]
    boundaries: List[int]


def process_piece(
    img,
    debug=False,
    # Green removal parameters
    green_color="green",
    # Binarization parameters
    threshold_value=127,
    # Morphology parameters
    morph_kernel_size=(5, 5),
    morph_iterations=1,
    # Main contour detection parameters
    main_min_area=10_000,
    main_max_area=200_000,
    # Color detection parameters
    small_area_max=90_000,
    medium_area_max=130_000,
    # Inner part detection parameters
    inner_min_area=100,
    inner_max_area=200_000,
    # Labeling parameters
    extent_threshold=0.5,
    label_small_area_max=1000,
    label_medium_area_max=2100,
    # ROI parameters
    roi_search_range=75,
    roi_margin=5,
):
    """
    Process a piece image through the complete detection pipeline.

    This function performs:
    1. Background (green) removal
    2. Binarization and morphological processing
    3. Main piece contour detection
    4. Color and size classification
    5. Cropping and alignment
    6. Internal sub-part detection
    7. Labeling (marker/small/medium/large)
    8. ROI computation and counting
    9. Value decoding
    10. Result annotation

    Parameters
    ----------
    img : np.ndarray
        Input BGR image.
    debug : bool
        If True, display intermediate images.

    Returns
    -------
    PipelineResult
        Container with all detection results and intermediate images.

    Raises
    ------
    RuntimeError
        If no valid contours are found.
    """
    # Store original
    original = img.copy()

    # 1) Remove green background
    no_green, green_mask = remove_color_hsv(img, green_color)
    if debug:
        show_image(img, "Original")
        show_image(no_green, "Without green (HSV)")

    # 2) Binarize
    gray, binary_inv = binarize_and_invert(no_green, threshold_value=threshold_value)
    if debug:
        show_image(cv2.cvtColor(binary_inv, cv2.COLOR_GRAY2BGR), "Binary inverse")

    # 3) Morphological opening
    gray_closed = apply_morphological_opening(
        gray, kernel_size=morph_kernel_size, iterations=morph_iterations
    )
    if debug:
        show_image(cv2.cvtColor(gray_closed, cv2.COLOR_GRAY2BGR), "After opening")

    # 4) Find main contours
    contours, contour_img = find_and_draw_contours_with_area_limits(
        binary_img=gray_closed,
        original_bgr=img,
        min_area=main_min_area,
        max_area=main_max_area
    )

    if debug:
        show_image(contour_img, "Contours (area filtered)")

    if not contours:
        raise RuntimeError("No contours found for parts.")

    # 4b) Get main contour and crop/align
    main_cnt = max(contours, key=cv2.contourArea)
    cropped_aligned = crop_and_align_vertical(img, main_cnt)
    if debug:
        show_image(cropped_aligned, "Cropped & aligned")

    # 4c) Detect piece color and check consistency
    try:
        piece_color, size_label, is_consistent, color_counts = detect_piece_color_and_check_size(
            img_bgr=img,
            contour=main_cnt,
            small_area_max=small_area_max,
            medium_area_max=medium_area_max,
            color_area_fraction_thresh=0,
            debug=debug
        )
        if debug:
            print(f"Piece color: {piece_color}, size: {size_label}, consistent: {is_consistent}")

        second_color = piece_color if is_consistent else "blue"

    except ValueError as e:
        if debug:
            print(f"[WARN] Color detection failed: {e}")
        piece_color = "unknown"
        size_label = "unknown"
        is_consistent = False
        second_color = "blue"

    # Work on current_part (may be rotated later)
    current_part = cropped_aligned

    # 5) Detect internal color parts
    part_contours, part_contour_img, mask_closed = detect_color_parts(
        current_part, second_color, debug=debug
    )

    # 6) Extract features and label
    features_list = extract_contour_features(part_contours)
    for f in features_list:
        f["label"] = label_contour_by_extent_and_area(
            f,
            extent_threshold=extent_threshold,
            small_area_max=label_small_area_max,
            medium_area_max=label_medium_area_max
        )

    colored_part = draw_labeled_contours_colored(
        original_bgr=current_part,
        contours=part_contours,
        features_list=features_list,
        extent_threshold=extent_threshold,
        small_area_max=label_small_area_max,
        medium_area_max=label_medium_area_max
    )
    if debug:
        show_image(colored_part, "Labeled contours")

    # 7a) Rotate if marker is in bottom
    rotated_part, rotated = rotate_if_marker_bottom(
        img_bgr=current_part,
        features_list=features_list,
        bottom_fraction=0.33
    )

    # 7b) Rotate if blue piece is upside down
    if not rotated and piece_color == "blue":
        rotated_part, rotated = rotate_if_blue_piece_upside_down(
            img_bgr=current_part,
            features_list=features_list,
            main_contour_height=current_part.shape[0],
            bottom_fraction=0.33
        )

    if debug:
        show_image(rotated_part, f"After rotation check (rotated={rotated})")

    # If rotated, recalculate features
    if rotated:
        current_part = rotated_part

        part_contours, part_contour_img, mask_closed = detect_color_parts(
            current_part, second_color, debug=False
        )

        features_list = extract_contour_features(part_contours)
        for f in features_list:
            f["label"] = label_contour_by_extent_and_area(
                f,
                extent_threshold=extent_threshold,
                small_area_max=label_small_area_max,
                medium_area_max=label_medium_area_max
            )

        colored_part = draw_labeled_contours_colored(
            original_bgr=current_part,
            contours=part_contours,
            features_list=features_list,
            extent_threshold=extent_threshold,
            small_area_max=label_small_area_max,
            medium_area_max=label_medium_area_max
        )
        if debug:
            show_image(colored_part, "Labeled contours (after rotation)")

    # 8) Compute ROIs
    rois, boundaries = split_fixed_horizontal_rois_and_count(
        image_shape=current_part.shape,
        features_list=features_list,
        piece_color=piece_color,
        search_range=roi_search_range,
        margin=roi_margin,
    )

    if debug:
        for r in rois:
            print(f"ROI {r['index']} (y: {r['y_start']}-{r['y_end']}) -> {r['counts']}")

    roi_vis = draw_horizontal_rois_from_boundaries(current_part, boundaries, alpha=0.3)
    if debug:
        show_image(roi_vis, "ROI visualization")

    # 9) Decode digits
    decoded_digits = []
    for r in rois:
        try:
            digit = decode_roi_to_number(r["counts"])
            decoded_digits.append(digit)
            if debug:
                print(f"ROI {r['index']} -> digit: {digit}")
        except ValueError as e:
            if debug:
                print(f"ROI {r['index']} -> ERROR: {e}")
            decoded_digits.append(None)

    # Compute total value
    total_value = compute_total_value_from_rois(decoded_digits, piece_color)
    if debug:
        print(f"Total value: {total_value}")

    # 10) Annotate result
    annotated = annotate_part_result(
        img, piece_color, size_label, is_consistent, total_value,
        main_contour=main_cnt, box_scale=2.0
    )
    if debug:
        show_image(annotated, "Final result")

    return PipelineResult(
        original_image=original,
        piece_color=piece_color,
        size_label=size_label,
        is_consistent=is_consistent,
        total_value=total_value,
        cropped_part=current_part,
        labeled_part=colored_part,
        roi_visualization=roi_vis,
        annotated_image=annotated,
        main_contour=main_cnt,
        features_list=features_list,
        rois=rois,
        decoded_digits=decoded_digits,
        boundaries=boundaries,
    )

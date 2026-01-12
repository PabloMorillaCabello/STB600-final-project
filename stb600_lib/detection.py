"""
Detection Module - Shared Pipeline Functions
=============================================

Core detection functions shared by result_viewer, video_processor, and setup GUI.
Ensures consistent processing across all components.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from .color import remove_color_hsv
from .morphology import binarize_and_invert, apply_morphological_opening


def preprocess_frame(
    img: np.ndarray,
    threshold_value: int = 0,
    kernel_size: int = 5,
    iterations: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply standard preprocessing pipeline to an image.
    
    Steps:
    1. Remove green background
    2. Convert to grayscale and binarize (with inversion)
    3. Apply morphological opening
    
    Parameters
    ----------
    img : np.ndarray
        Input BGR image.
    threshold_value : int
        Threshold for binarization (0 = Otsu auto).
    kernel_size : int
        Size of morphological kernel.
    iterations : int
        Number of morphological iterations.
    
    Returns
    -------
    no_green : np.ndarray
        Image after green removal.
    binary : np.ndarray
        Binary image after binarization.
    processed : np.ndarray
        Final processed binary image after morphological operations.
    """
    # Step 1: Remove green background
    no_green, _ = remove_color_hsv(img, "green")
    
    # Step 2: Binarize
    _, binary = binarize_and_invert(no_green, threshold_value=threshold_value)
    
    # Step 3: Morphological opening
    processed = apply_morphological_opening(
        binary, kernel_size=(kernel_size, kernel_size), iterations=iterations
    )
    
    return no_green, binary, processed


def find_piece_contours(
    binary_img: np.ndarray,
    min_area: int,
    max_area: int,
    auto_invert: bool = True,
) -> Tuple[List[np.ndarray], bool, Dict[str, Any]]:
    """
    Find contours of pieces in a binary image with smart inversion detection.
    
    Automatically detects if the binary image needs inversion (pieces are black
    instead of white) and handles it accordingly.
    
    Parameters
    ----------
    binary_img : np.ndarray
        Binary image (single channel).
    min_area : int
        Minimum contour area to keep.
    max_area : int
        Maximum contour area to keep.
    auto_invert : bool
        If True, automatically detect and handle inverted binary images.
        If False, use the binary image as-is.
    
    Returns
    -------
    filtered_contours : list
        List of contours that pass the area filter.
    used_inverted : bool
        True if the inverted binary was used.
    debug_info : dict
        Debug information including pixel counts and all contour areas.
    """
    total_pixels = binary_img.shape[0] * binary_img.shape[1]
    white_pixels = cv2.countNonZero(binary_img)
    black_pixels = total_pixels - white_pixels
    
    # Find contours on original binary
    all_contours_normal, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    areas_normal = sorted([cv2.contourArea(c) for c in all_contours_normal], reverse=True)
    
    used_inverted = False
    all_contours = all_contours_normal
    
    if auto_invert:
        # Also find contours on inverted binary
        binary_inverted = cv2.bitwise_not(binary_img)
        all_contours_inverted, _ = cv2.findContours(
            binary_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        areas_inverted = sorted([cv2.contourArea(c) for c in all_contours_inverted], reverse=True)
        
        # If normal finds mostly one huge contour (>50% of frame), use inverted
        # This happens when pieces are BLACK and background is WHITE
        if len(areas_normal) > 0 and areas_normal[0] > total_pixels * 0.5:
            used_inverted = True
            all_contours = all_contours_inverted
    else:
        areas_inverted = []
    
    # Filter contours by area
    filtered_contours = []
    all_areas = []
    
    for c in all_contours:
        area = cv2.contourArea(c)
        if area > 100:  # Skip tiny noise
            all_areas.append(area)
        if min_area <= area <= max_area:
            filtered_contours.append(c)
    
    all_areas.sort(reverse=True)
    
    # Collect debug info
    debug_info = {
        "total_pixels": total_pixels,
        "white_pixels": white_pixels,
        "black_pixels": black_pixels,
        "white_pct": 100 * white_pixels / total_pixels if total_pixels > 0 else 0,
        "total_contours": len(all_contours),
        "filtered_contours": len(filtered_contours),
        "all_areas": all_areas,
        "areas_normal": areas_normal[:5] if areas_normal else [],
        "areas_inverted": areas_inverted[:5] if areas_inverted else [],
        "used_inverted": used_inverted,
    }
    
    return filtered_contours, used_inverted, debug_info


def get_detection_ranges(config: dict) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], int, int]:
    """
    Extract detection ranges from config dictionary.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with 'big_contours' section.
    
    Returns
    -------
    red_range : tuple
        (min, max) area for red pieces.
    yellow_range : tuple
        (min, max) area for yellow pieces.
    blue_range : tuple
        (min, max) area for blue pieces.
    overall_min : int
        Expanded minimum (80% of red_min) for catching fakes.
    overall_max : int
        Expanded maximum (150% of blue_max) for catching fakes.
    """
    bc = config.get("big_contours", {})
    
    red_range = (bc.get("red_min", 5000), bc.get("red_max", 80000))
    yellow_range = (bc.get("yellow_min", 80000), bc.get("yellow_max", 160000))
    blue_range = (bc.get("blue_min", 160000), bc.get("blue_max", 250000))
    
    # Expand detection range to catch potential fakes
    # 20% smaller than red_min, 50% larger than blue_max
    overall_min = int(red_range[0] * 0.8)
    overall_max = int(blue_range[1] * 1.5)
    
    return red_range, yellow_range, blue_range, overall_min, overall_max


def detect_pieces_in_frame(
    img: np.ndarray,
    config: dict,
    auto_invert: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    Complete detection pipeline: preprocess and find piece contours.
    
    This is the main entry point for piece detection, combining preprocessing
    and contour finding into a single call.
    
    Parameters
    ----------
    img : np.ndarray
        Input BGR image.
    config : dict
        Configuration dictionary with preprocessing and detection settings.
    auto_invert : bool
        If True, automatically handle inverted binary images.
    
    Returns
    -------
    contours : list
        List of detected piece contours.
    processed_binary : np.ndarray
        The processed binary image (possibly inverted).
    debug_info : dict
        Debug information from all processing steps.
    """ 
    # Get config values
    bz = config.get("binarize", {})
    op = config.get("opening", {})
    
    # Preprocess
    no_green, binary, processed = preprocess_frame(
        img,
        threshold_value=bz.get("thresh_value", 0),
        kernel_size=op.get("kernel_size", 5),
        iterations=op.get("iterations", 1),
    )
    
    # Get detection ranges
    red_range, yellow_range, blue_range, overall_min, overall_max = get_detection_ranges(config)
    
    # Find contours
    contours, used_inverted, contour_debug = find_piece_contours(
        processed, overall_min, overall_max, auto_invert=auto_invert
    )
    
    # Prepare the binary image that was actually used
    if used_inverted:
        processed_binary = cv2.bitwise_not(processed)
    else:
        processed_binary = processed
    
    # Combine debug info
    debug_info = {
        **contour_debug,
        "red_range": red_range,
        "yellow_range": yellow_range,
        "blue_range": blue_range,
        "overall_min": overall_min,
        "overall_max": overall_max,
        "preprocessing": {
            "threshold": bz.get("thresh_value", 0),
            "kernel_size": op.get("kernel_size", 5),
            "iterations": op.get("iterations", 1),
        },
    }
    
    return contours, processed_binary, debug_info

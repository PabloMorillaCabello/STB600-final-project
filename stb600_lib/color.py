"""
Color Module - HSV Color Processing and Detection
=================================================

Functions for color removal, detection, and piece classification based on color.
"""

import cv2
import numpy as np


def remove_color_hsv(img_bgr, color_name="green"):
    """
    Remove (mask out) a specific color region in a BGR image using HSV color space.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input BGR image from OpenCV.
    color_name : str
        Color to isolate and remove. One of: "green", "blue", "red", "yellow".

    Returns
    -------
    result_bgr : np.ndarray
        BGR image with the selected color removed (set to black).
    mask_color : np.ndarray
        Binary mask where the selected color pixels are 255 and others are 0.

    Raises
    ------
    ValueError
        If input image is None or color_name is invalid.
    """
    if img_bgr is None:
        raise ValueError("Input image is None.")

    # Convert from BGR to HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Predefined HSV ranges for basic colors
    color_ranges = {
        "green": {
            "lower": np.array([30, 42, 52]),
            "upper": np.array([90, 255, 255])
        },
        "blue": {
            "lower": np.array([90, 50, 70]),
            "upper": np.array([128, 255, 255])
        },
        "yellow": {
            "lower": np.array([25, 50, 70]),
            "upper": np.array([35, 255, 255])
        },
        # Red wraps around 0, so we use two ranges
        "red": {
            "lower1": np.array([0, 30, 40]),
            "upper1": np.array([12, 255, 255]),
            "lower2": np.array([165, 30, 40]),
            "upper2": np.array([180, 255, 255])
        }
    }

    color_name = color_name.lower()
    if color_name not in color_ranges:
        raise ValueError("color_name must be one of: 'green', 'blue', 'red', 'yellow'.")

    if color_name == "red":
        # Two ranges for red: near 0 and near 180 hue
        ranges = color_ranges["red"]
        mask1 = cv2.inRange(img_hsv, ranges["lower1"], ranges["upper1"])
        mask2 = cv2.inRange(img_hsv, ranges["lower2"], ranges["upper2"])
        mask_color = cv2.bitwise_or(mask1, mask2)
    else:
        ranges = color_ranges[color_name]
        mask_color = cv2.inRange(img_hsv, ranges["lower"], ranges["upper"])

    # Invert mask to keep everything except the selected color
    mask_not_color = cv2.bitwise_not(mask_color)

    # Keep only non-selected-color parts of the original image
    result_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_not_color)

    return result_bgr, mask_color


def detect_piece_color_and_check_size(
    img_bgr,
    contour,
    small_area_max=50_000,
    medium_area_max=120_000,
    color_area_fraction_thresh=0,
    debug=False
):
    """
    Detect the dominant color (red, yellow, or blue) inside a piece contour
    and check if the total area matches the expected size for that color.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input BGR image.
    contour : np.ndarray
        Contour defining the piece boundary.
    small_area_max : int
        Maximum area for small pieces.
    medium_area_max : int
        Maximum area for medium pieces (above is big).
    color_area_fraction_thresh : float
        Minimum fraction of color pixels required.
    debug : bool
        If True, print debug information.

    Returns
    -------
    piece_color : str
        Detected color: "red", "yellow", or "blue".
    size_label : str
        Size classification: "SMALL", "MEDIUM", or "BIG".
    is_consistent : bool
        Whether the color matches expected size.
    color_counts : dict
        Pixel counts for each color.

    Raises
    ------
    ValueError
        If contour area is non-positive or color detection fails.
    """
    area = cv2.contourArea(contour)
    if debug:
        print("AREA:", area)
    if area <= 0:
        raise ValueError("Contour area is non-positive")

    # Size classification by area
    if area <= small_area_max:
        size_label = "SMALL"
    elif area <= medium_area_max:
        size_label = "MEDIUM"
    else:
        size_label = "BIG"

    # Create contour mask (irregular ROI)
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # HSV ranges for each color
    blue_lower = np.array([100, 80, 80])
    blue_upper = np.array([130, 255, 255])

    yellow_lower = np.array([20, 80, 80])
    yellow_upper = np.array([35, 255, 255])

    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 70, 50])
    red_upper2 = np.array([179, 255, 255])

    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Restrict to interior of piece
    blue_mask = cv2.bitwise_and(blue_mask, blue_mask, mask=mask)
    yellow_mask = cv2.bitwise_and(yellow_mask, yellow_mask, mask=mask)
    red_mask = cv2.bitwise_and(red_mask, red_mask, mask=mask)

    blue_count = cv2.countNonZero(blue_mask)
    yellow_count = cv2.countNonZero(yellow_mask)
    red_count = cv2.countNonZero(red_mask)

    color_counts = {
        "blue": blue_count,
        "yellow": yellow_count,
        "red": red_count,
    }

    # Choose dominant color
    piece_color = max(color_counts, key=color_counts.get)
    dominant_pixels = color_counts[piece_color]
    fraction = dominant_pixels / float(area)

    # Check if dominant color has enough pixels
    if fraction < color_area_fraction_thresh:
        raise ValueError(
            f"Dominant color {piece_color} has too few pixels "
            f"({fraction:.2f} < {color_area_fraction_thresh})"
        )

    # Consistency rules: color -> expected size
    expected_size = {
        "red": "SMALL",
        "yellow": "MEDIUM",
        "blue": "BIG",
    }
    is_consistent = (expected_size[piece_color] == size_label)

    return piece_color, size_label, is_consistent, color_counts


def detect_color_parts(part_img, color_name, debug=False):
    """
    Detect colored sub-parts within a piece image.

    Parameters
    ----------
    part_img : np.ndarray
        Cropped piece image (BGR).
    color_name : str
        Color to detect ("blue", "red", "yellow").
    debug : bool
        If True, display intermediate images.

    Returns
    -------
    part_contours : list
        List of detected contours.
    part_contour_img : np.ndarray
        Image with contours drawn.
    mask_closed : np.ndarray
        Processed binary mask.
    """
    from .morphology import apply_morphological_closing
    from .contours import find_and_draw_contours_with_area_limits
    from .display import show_image

    no_color_p, mask = remove_color_hsv(part_img, color_name)
    if debug:
        show_image(no_color_p, f"Part without {color_name}")
        show_image(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), f"Mask of {color_name}")

    # Clean up mask with closing
    mask_closed = apply_morphological_closing(
        mask, kernel_size=(5, 5), iterations=1
    )
    if debug:
        show_image(cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2BGR), "Mask closed")

    part_contours, part_contour_img = find_and_draw_contours_with_area_limits(
        binary_img=mask_closed,
        original_bgr=part_img,
        min_area=100,
        max_area=200_000
    )
    if debug:
        show_image(part_contour_img, "Contours inside part")

    return part_contours, part_contour_img, mask_closed

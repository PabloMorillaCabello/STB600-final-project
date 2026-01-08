"""
Transforms Module - Image Rotation Utilities
============================================

Functions for rotating images based on marker position and piece orientation.
"""

import cv2
import numpy as np


def rotate_if_marker_bottom(img_bgr, features_list, bottom_fraction=0.33):
    """
    Rotate image 180 degrees if a marker is found in the bottom portion.

    Parameters
    ----------
    img_bgr : np.ndarray
        Original image.
    features_list : list of dict
        Feature dictionaries with "label" and "centroid" keys.
    bottom_fraction : float
        Fraction of image height considered "bottom" (e.g., 0.33 = bottom third).

    Returns
    -------
    rotated_img : np.ndarray
        Rotated image if condition met, otherwise original.
    rotated : bool
        True if image was rotated, False otherwise.

    Raises
    ------
    ValueError
        If input image is None.
    """
    if img_bgr is None:
        raise ValueError("img_bgr is None.")

    h, w = img_bgr.shape[:2]
    bottom_start_y = int(h * (1.0 - bottom_fraction))

    marker_in_bottom = False

    for f in features_list:
        label = f.get("label", None)
        cx, cy = f.get("centroid", (None, None))

        if label != "marker":
            continue
        if cx is None or cy is None:
            continue

        if cy >= bottom_start_y:
            marker_in_bottom = True
            break

    if marker_in_bottom:
        rotated_img = cv2.rotate(img_bgr, cv2.ROTATE_180)
        return rotated_img, True
    else:
        return img_bgr, False


def rotate_if_blue_piece_upside_down(
    img_bgr,
    features_list,
    main_contour_height,
    bottom_fraction=0.33
):
    """
    Rotate image 180 degrees if a blue piece appears upside down.

    Detects upside-down blue pieces by checking if the highest sub-section
    contour is more than 1/3 from the top.

    Parameters
    ----------
    img_bgr : np.ndarray
        Original image.
    features_list : list of dict
        Feature dictionaries with "label" and "bounding_rect" keys.
    main_contour_height : int
        Height of the main contour for reference.
    bottom_fraction : float
        Fraction of height considered "bottom" threshold.

    Returns
    -------
    rotated_img : np.ndarray
        Rotated image if condition met, otherwise original.
    rotated : bool
        True if image was rotated, False otherwise.

    Raises
    ------
    ValueError
        If input image is None.
    """
    if img_bgr is None:
        raise ValueError("img_bgr is None.")

    bottom_start_y = int(main_contour_height * (1.0 - bottom_fraction))

    highest_cy = None

    for f in features_list:
        label = f.get("label", None)
        bounding_rect = f.get("bounding_rect", (None, None, None, None))
        cy = bounding_rect[1]  # y from bbox

        if label not in ("small", "medium", "large"):
            continue
        if cy is None:
            continue

        if highest_cy is None or cy < highest_cy:
            highest_cy = cy

    if highest_cy is not None and highest_cy >= bottom_start_y:
        rotated_img = cv2.rotate(img_bgr, cv2.ROTATE_180)
        return rotated_img, True
    else:
        return img_bgr, False

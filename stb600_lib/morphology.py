"""
Morphology Module - Morphological Operations
============================================

Functions for morphological image processing (opening, closing, binarization).
"""

import cv2
import numpy as np


def binarize_and_invert(img_bgr, threshold_value=127, max_value=255):
    """
    Convert a BGR image to grayscale, then apply inverse binary thresholding.

    Uses Otsu's method for automatic threshold selection combined with the
    provided threshold_value.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input BGR image from OpenCV.
    threshold_value : int
        Threshold value (0-255).
    max_value : int
        Maximum value used in binary output (usually 255).

    Returns
    -------
    gray : np.ndarray
        Grayscale version of the input image.
    binary_inv : np.ndarray
        Inverse binary image (single channel).

    Raises
    ------
    ValueError
        If input image is None.
    """
    if img_bgr is None:
        raise ValueError("Input image is None.")

    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Inverse binary threshold with Otsu
    _, binary_inv = cv2.threshold(
        gray,
        threshold_value,
        max_value,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    return gray, binary_inv


def apply_morphological_closing(gray_img, kernel_size=(5, 5), iterations=1):
    """
    Apply morphological closing on a grayscale/binary image.

    Closing = dilation followed by erosion.
    Useful for filling small holes and connecting nearby objects.

    Parameters
    ----------
    gray_img : np.ndarray
        Input single-channel image (grayscale or binary).
    kernel_size : tuple of int
        Size of the structuring element (width, height).
    iterations : int
        Number of times the closing operation is applied.

    Returns
    -------
    gray_fixed : np.ndarray
        Image after morphological closing.

    Raises
    ------
    ValueError
        If input is None or not single-channel.
    """
    if gray_img is None:
        raise ValueError("Input gray_img is None.")

    if len(gray_img.shape) != 2:
        raise ValueError("gray_img must be a single-channel grayscale/binary image.")

    # Create structuring element (kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Apply morphological closing
    gray_fixed = cv2.morphologyEx(
        gray_img,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations
    )

    return gray_fixed


def apply_morphological_opening(gray_img, kernel_size=(5, 5), iterations=1):
    """
    Apply morphological opening on a grayscale/binary image.

    Opening = erosion followed by dilation.
    Useful for removing small bright noise and separating thin connections.

    Parameters
    ----------
    gray_img : np.ndarray
        Input single-channel image (grayscale or binary).
    kernel_size : tuple of int
        Size of the structuring element (width, height).
    iterations : int
        Number of times the opening operation is applied.

    Returns
    -------
    gray_fixed : np.ndarray
        Image after morphological opening.

    Raises
    ------
    ValueError
        If input is None or not single-channel.
    """
    if gray_img is None:
        raise ValueError("Input gray_img is None.")

    if len(gray_img.shape) != 2:
        raise ValueError("gray_img must be a single-channel grayscale/binary image.")

    # Create structuring element (kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Apply morphological opening
    gray_fixed = cv2.morphologyEx(
        gray_img,
        cv2.MORPH_OPEN,
        kernel,
        iterations=iterations,
    )

    return gray_fixed

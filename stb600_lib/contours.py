"""
Contours Module - Contour Detection and Processing
==================================================

Functions for finding, analyzing, labeling, and cropping contours.
"""

import cv2
import numpy as np
import math


def find_and_draw_contours_with_area_limits(
    binary_img,
    original_bgr,
    min_area=100,
    max_area=100000
):
    """
    Find contours on a binary image, filter by area, draw on original image.

    Parameters
    ----------
    binary_img : np.ndarray
        Binary/single-channel image for contour detection.
    original_bgr : np.ndarray
        Original BGR image for drawing contours.
    min_area : int
        Minimum contour area to keep.
    max_area : int
        Maximum contour area to keep.

    Returns
    -------
    filtered_contours : list
        List of contours that pass the area filter.
    output_img : np.ndarray
        Original image with filtered contours drawn and numbered.

    Raises
    ------
    ValueError
        If inputs are None or binary_img is not single-channel.
    """
    if binary_img is None:
        raise ValueError("binary_img is None.")
    if original_bgr is None:
        raise ValueError("original_bgr is None.")

    if len(binary_img.shape) != 2:
        raise ValueError("binary_img must be single-channel (binary).")

    contours, hierarchy = cv2.findContours(
        binary_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    total_contours = len(contours)

    filtered_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            filtered_contours.append(c)

    filtered_count = len(filtered_contours)

    # Draw and number filtered contours
    output_img = original_bgr.copy()
    cv2.drawContours(output_img, filtered_contours, -1, (0, 0, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, c in enumerate(filtered_contours):
        # Centroid via moments (fallback to boundingRect)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2

        label = str(idx)
        cv2.putText(
            output_img,
            label,
            (cx, cy),
            font,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    print(f"Total contours found: {total_contours}")
    print(f"Contours after area filter [{min_area}, {max_area}]: {filtered_count}")

    return filtered_contours, output_img


def extract_contour_features(contours):
    """
    Extract multiple geometric features for each contour.

    Parameters
    ----------
    contours : list of np.ndarray
        Contours from cv2.findContours.

    Returns
    -------
    features_list : list of dict
        One dictionary per contour with keys:
        - index, area, perimeter, centroid, bounding_rect
        - aspect_ratio, rect_area, extent, hull_area, solidity
        - equivalent_diameter, min_enclosing_circle, is_convex, approx_vertices
    """
    features_list = []

    for idx, cnt in enumerate(contours):
        # Area
        area = cv2.contourArea(cnt)

        # Perimeter (arc length)
        perimeter = cv2.arcLength(cnt, True)

        # Moments and centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = None, None

        # Axis-aligned bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = float(w * h) if w > 0 and h > 0 else 0.0

        # Aspect ratio: width / height
        aspect_ratio = (float(w) / h) if h > 0 else None

        # Extent: area / bounding_rect_area
        extent = (area / rect_area) if rect_area > 0 else None

        # Convex hull and solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = (area / hull_area) if hull_area > 0 else None

        # Equivalent diameter: diameter of circle with same area
        equiv_diameter = math.sqrt(4 * area / math.pi) if area > 0 else 0.0

        # Minimum enclosing circle
        (circle_x, circle_y), radius = cv2.minEnclosingCircle(cnt)

        # Convexity
        is_convex = cv2.isContourConvex(cnt)

        # Contour approximation (polygon)
        epsilon = 0.01 * perimeter  # tolerance as 1% of perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx_vertices = len(approx)

        features = {
            "index": idx,
            "area": area,
            "perimeter": perimeter,
            "centroid": (cx, cy),
            "bounding_rect": (x, y, w, h),
            "aspect_ratio": aspect_ratio,
            "rect_area": rect_area,
            "extent": extent,
            "hull_area": hull_area,
            "solidity": solidity,
            "equivalent_diameter": equiv_diameter,
            "min_enclosing_circle": (circle_x, circle_y, radius),
            "is_convex": is_convex,
            "approx_vertices": approx_vertices,
        }

        features_list.append(features)

    return features_list


def label_contour_by_extent_and_area(
    features,
    extent_threshold=0.5,
    small_area_max=1000,
    medium_area_max=2000
):
    """
    Label a contour based on its extent and area.

    Rules:
        - if extent < extent_threshold -> "marker"
        - else if area <= small_area_max -> "small"
        - else if area <= medium_area_max -> "medium"
        - else -> "large"

    Parameters
    ----------
    features : dict
        Feature dictionary with "extent" and "area" keys.
    extent_threshold : float
        Threshold below which contour is labeled "marker".
    small_area_max : int
        Maximum area for "small" label.
    medium_area_max : int
        Maximum area for "medium" label.

    Returns
    -------
    str
        Label: "marker", "small", "medium", "large", or "unknown".
    """
    extent = features["extent"]
    area = features["area"]

    if extent is None:
        return "unknown"

    if extent < extent_threshold:
        return "marker"

    # extent >= threshold -> classify by area
    if area <= small_area_max:
        return "small"
    elif area <= medium_area_max:
        return "medium"
    else:
        return "large"


def crop_region_from_contour(image, contour, use_mask=False, padding=0):
    """
    Crop the region corresponding to a contour from an image.

    Parameters
    ----------
    image : np.ndarray
        Original BGR (or grayscale) image.
    contour : np.ndarray
        Single contour as returned by cv2.findContours.
    use_mask : bool
        If False: use bounding rectangle (fast, rectangular crop).
        If True: use mask to keep only exact contour shape.
    padding : int
        Extra pixels to expand the bounding box on each side.

    Returns
    -------
    cropped : np.ndarray
        Cropped image region (BGR or grayscale).
    bbox : tuple
        Bounding box (x, y, w, h) used for cropping.

    Raises
    ------
    ValueError
        If image or contour is None/empty.
    """
    if image is None:
        raise ValueError("Input image is None.")
    if contour is None or len(contour) == 0:
        raise ValueError("Contour is empty or None.")

    # Get bounding box around contour
    x, y, w, h = cv2.boundingRect(contour)

    # Apply optional padding
    x_p = max(x - padding, 0)
    y_p = max(y - padding, 0)
    x2_p = min(x + w + padding, image.shape[1])
    y2_p = min(y + h + padding, image.shape[0])

    # Basic rectangular crop
    roi = image[y_p:y2_p, x_p:x2_p]

    if not use_mask:
        return roi, (x_p, y_p, x2_p - x_p, y2_p - y_p)

    # Mask-based precise crop (keep only inside the contour)
    if len(image.shape) == 3:
        mask = np.zeros((y2_p - y_p, x2_p - x_p), dtype=np.uint8)
    else:
        mask = np.zeros_like(roi, dtype=np.uint8)

    # Shift contour coordinates to ROI coordinate system
    contour_shifted = contour.copy()
    contour_shifted[:, 0, 0] = contour[:, 0, 0] - x_p
    contour_shifted[:, 0, 1] = contour[:, 0, 1] - y_p

    # Draw filled contour on mask
    cv2.drawContours(mask, [contour_shifted], -1, 255, thickness=-1)

    # If original is color, keep 3 channels
    if len(image.shape) == 3:
        mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cropped = cv2.bitwise_and(roi, mask_3c)
    else:
        cropped = cv2.bitwise_and(roi, mask)

    return cropped, (x_p, y_p, x2_p - x_p, y2_p - y_p)


def crop_and_align_vertical(img, cnt):
    """
    Crop a contour region and rotate it so the longest side is vertical.

    Also flips 180 degrees if the centroid suggests the piece is upside down.

    Parameters
    ----------
    img : np.ndarray
        Original BGR image.
    cnt : np.ndarray
        Contour to crop and align.

    Returns
    -------
    cropped : np.ndarray
        Cropped and vertically aligned image.
    """
    # Minimum area rectangle
    rect = cv2.minAreaRect(cnt)
    (rcx, rcy), (rw, rh), angle = rect

    # OpenCV returns angle in [-90, 0). We want the long side vertical.
    if rw < rh:
        # Long side ~ vertical: angle is already correct
        rot_angle = angle
    else:
        # Long side ~ horizontal: add 90 to make it vertical
        rot_angle = angle + 90.0

    # Rotate image around its center
    h, w = img.shape[:2]
    img_center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(img_center, rot_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    # Transform contour with the same matrix
    cnt_rot = cv2.transform(cnt, M)

    # Crop bounding box of rotated contour
    x, y, bw, bh = cv2.boundingRect(cnt_rot)
    cropped = rotated[y:y+bh, x:x+bw]

    # Extra adjustment with centroid to avoid upside-down piece
    M_cnt = cv2.moments(cnt)
    if M_cnt["m00"] != 0:
        cx = M_cnt["m10"] / M_cnt["m00"]
        cy = M_cnt["m01"] / M_cnt["m00"]
    else:
        cx, cy = rcx, rcy  # fallback to rect center

    # Center of original image
    img_mid_y = h / 2.0

    # If centroid is below image center, piece is likely upside down
    if cy > img_mid_y:
        cropped = cv2.rotate(cropped, cv2.ROTATE_180)

    return cropped

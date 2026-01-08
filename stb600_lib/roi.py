"""
ROI Module - Region of Interest Computation
==========================================

Functions for computing horizontal ROIs and counting features within them.
"""

import cv2
import numpy as np


def _find_empty_horizontal_line(target_y, intervals, h, search_range=40):
    """
    Find a horizontal line near target_y that doesn't intersect object intervals.

    Parameters
    ----------
    target_y : float
        Target y-coordinate.
    intervals : list of tuple
        List of (top, bottom) intervals occupied by objects.
    h : int
        Image height.
    search_range : int
        Search window size.

    Returns
    -------
    int
        Best y-coordinate for boundary.
    """
    if not intervals:
        return int(target_y)

    intervals_arr = np.array(intervals, dtype=np.int32)

    y_min = max(0, int(target_y) - search_range)
    y_max = min(h - 1, int(target_y) + search_range)

    best_y = int(target_y)
    best_dist = -1

    for y in range(y_min, y_max + 1):
        inside = np.logical_and(intervals_arr[:, 0] <= y, y < intervals_arr[:, 1])
        if not inside.any():
            return y

        # Heuristic: keep y farthest from any interval border
        dists_top = np.abs(y - intervals_arr[:, 0])
        dists_bottom = np.abs(y - intervals_arr[:, 1])
        min_to_border = np.minimum(dists_top, dists_bottom).min()
        if min_to_border > best_dist:
            best_dist = min_to_border
            best_y = y

    return best_y


def split_fixed_horizontal_rois_and_count(
    image_shape,
    features_list,
    piece_color=None,
    search_range=40,
    margin=5,
):
    """
    Split image into 2 or 3 horizontal ROIs at fixed target heights.

    Target heights depend on piece_color:
      - 'red'    -> 2 ROIs, boundary around h/2
      - 'yellow' -> 3 ROIs, at 0.35h and 0.7h
      - 'blue'   -> 3 ROIs, at 0.3h and 0.5h
      - other    -> 3 ROIs at h/3 and 2h/3

    Parameters
    ----------
    image_shape : tuple
        Image shape (height, width, ...).
    features_list : list of dict
        Feature dictionaries with "centroid", "label", and optionally "bbox".
    piece_color : str or None
        Piece color for determining ROI configuration.
    search_range : int
        Search window for boundary adjustment.
    margin : int
        Margin around object bounding boxes.

    Returns
    -------
    rois : list of dict
        ROI descriptors.
    boundaries : list of int
        Y-coordinates of boundaries.
    """
    h, w = image_shape[:2]

    # Collect vertical intervals from bounding boxes
    intervals = []
    for f in features_list:
        bbox = f.get("bbox", None)
        if bbox is None:
            continue
        x, y, bw, bh = bbox
        top = max(0, int(y) - margin)
        bottom = min(h, int(y + bh) + margin)
        intervals.append((top, bottom))

    color = (piece_color or "").lower()

    # Define target boundaries per piece color
    if color == "red":
        target_ratios = [0.5]
    elif color == "yellow":
        target_ratios = [0.35, 0.7]
    elif color == "blue":
        target_ratios = [0.3, 0.5]
    else:
        target_ratios = [1.0 / 3.0, 2.0 / 3.0]

    targets = [r * h for r in target_ratios]

    # Find boundary lines that avoid objects
    boundaries = [0]
    for t in targets:
        y_line = _find_empty_horizontal_line(
            target_y=t,
            intervals=intervals,
            h=h,
            search_range=search_range,
        )
        boundaries.append(int(y_line))
    boundaries.append(h)

    # Ensure sorted and within image bounds
    boundaries = sorted(set(max(0, min(h, y)) for y in boundaries))
    if len(boundaries) < 2:
        boundaries = [0, h]

    # Build ROIs
    rois = []
    for i in range(len(boundaries) - 1):
        ys = boundaries[i]
        ye = boundaries[i + 1]
        rois.append({
            "index": i,
            "y_start": ys,
            "y_end": ye,
            "counts": {
                "marker": 0,
                "small": 0,
                "medium": 0,
                "large": 0,
                "unknown": 0
            }
        })

    # Count features per ROI
    for f in features_list:
        cx, cy = f["centroid"]
        label = f.get("label", "unknown")
        if cx is None or cy is None:
            continue
        cy = int(cy)

        for roi in rois:
            if roi["y_start"] <= cy < roi["y_end"]:
                if label not in roi["counts"]:
                    roi["counts"]["unknown"] += 1
                else:
                    roi["counts"][label] += 1
                break

    return rois, boundaries

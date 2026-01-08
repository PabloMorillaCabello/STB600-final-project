"""
ROI Module - Region of Interest Computation
==========================================

Functions for computing horizontal ROIs and counting features within them.
"""

import cv2
import numpy as np


def compute_safe_horizontal_rois(image_shape, features_list, margin=10, search_range=40):
    """
    Compute 3 horizontal ROI boundaries that avoid cutting through centroids.

    Parameters
    ----------
    image_shape : tuple
        Shape of the image (height, width) or (height, width, channels).
    features_list : list of dict
        Feature dictionaries with "centroid" key.
    margin : int
        Minimum distance from boundary to any centroid.
    search_range : int
        Search window size around initial boundary positions.

    Returns
    -------
    list of int
        Four y-coordinates [y0, y1, y2, y3] defining 3 ROI bands.
    """
    h, w = image_shape[:2]

    # Initial equal-split boundaries
    y0 = 0
    y1_init = h // 3
    y2_init = 2 * h // 3
    y3 = h

    # Collect all cy values
    cy_list = []
    for f in features_list:
        cx, cy = f["centroid"]
        if cx is None or cy is None:
            continue
        cy_list.append(int(cy))

    cy_array = np.array(cy_list, dtype=np.int32) if cy_list else np.empty((0,), dtype=np.int32)

    def find_safe_y(y_initial):
        if cy_array.size == 0:
            return y_initial

        best_y = y_initial
        best_min_dist = -1

        y_min = max(0, y_initial - search_range)
        y_max = min(h - 1, y_initial + search_range)

        for y in range(y_min, y_max + 1):
            dists = np.abs(cy_array - y)
            min_dist = dists.min() if dists.size > 0 else margin

            if min_dist >= margin:
                return y
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_y = y

        return best_y

    # Find safe boundaries
    y1 = find_safe_y(y1_init)
    y2 = find_safe_y(y2_init)

    # Ensure correct order and limits
    y1 = max(min(y1, h - 1), 1)
    y2 = max(min(y2, h - 1), y1 + 1)

    # Enforce at least one centroid per ROI
    if cy_array.size > 0:
        def count_in_band(lo, hi):
            return np.logical_and(cy_array >= lo, cy_array < hi).sum()

        def closest_centroid_y(lo, hi):
            mask = np.logical_and(cy_array >= lo, cy_array < hi)
            if not mask.any():
                return None
            band_cy = cy_array[mask]
            mid = (lo + hi) / 2.0
            idx = np.argmin(np.abs(band_cy - mid))
            return int(band_cy[idx])

        c0 = count_in_band(y0, y1)
        c1 = count_in_band(y1, y2)
        c2 = count_in_band(y2, y3)

        if c0 == 0:
            target = closest_centroid_y(y0, y3)
            if target is not None:
                y1 = max(1, min(target, y2 - 1))

        c0 = count_in_band(y0, y1)
        c1 = count_in_band(y1, y2)
        c2 = count_in_band(y2, y3)

        if c1 == 0:
            target = closest_centroid_y(y0, y3)
            if target is not None:
                if abs(target - y1) < abs(target - y2):
                    y1 = max(1, min(target, y2 - 1))
                else:
                    y2 = max(y1 + 1, min(target, y3 - 1))

        c0 = count_in_band(y0, y1)
        c1 = count_in_band(y1, y2)
        c2 = count_in_band(y2, y3)

        if c2 == 0:
            target = closest_centroid_y(y0, y3)
            if target is not None:
                y2 = max(y1 + 1, min(target, h - 1))

        y1 = max(min(y1, h - 1), 1)
        y2 = max(min(y2, h - 1), y1 + 1)

    return [y0, y1, y2, y3]


def split_safe_horizontal_rois_and_count(
    image_shape,
    features_list,
    margin=10,
    search_range=40,
    piece_color=None,
):
    """
    Split image into horizontal ROIs and count labeled features per ROI.

    Parameters
    ----------
    image_shape : tuple
        Shape of the image (height, width, ...).
    features_list : list of dict
        Feature dictionaries with "centroid" and "label" keys.
    margin : int
        Minimum distance from boundary to centroids.
    search_range : int
        Search window for boundary adjustment.
    piece_color : str or None
        Piece color ("red", "yellow", "blue").
        If "red", uses 2 ROIs; otherwise 3 ROIs.

    Returns
    -------
    rois : list of dict
        ROI descriptors with "index", "y_start", "y_end", and "counts".
    boundaries : list of int
        Y-coordinates of ROI boundaries.
    """
    h, w = image_shape[:2]

    y0, y1, y2, y3 = compute_safe_horizontal_rois(
        image_shape,
        features_list,
        margin=margin,
        search_range=search_range
    )

    # Define ROI ranges based on piece color
    if piece_color == "red":
        # 2 ROIs for red
        roi_bounds = [(y0, y1), (y1, y3)]
        boundaries = [y0, y1, y3]
    else:
        # 3 ROIs for others
        roi_bounds = [(y0, y1), (y1, y2), (y2, y3)]
        boundaries = [y0, y1, y2, y3]

    # Initialize ROIs with zero counts
    rois = []
    for i, (ys, ye) in enumerate(roi_bounds):
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

    # Assign features to ROIs by centroid
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


def _find_safe_line_inside_gap(gap_start, gap_end, border_margin=5):
    """
    Find a y-coordinate inside a gap, away from gap borders.

    Parameters
    ----------
    gap_start : int
        Start of the gap.
    gap_end : int
        End of the gap.
    border_margin : int
        Minimum distance from gap borders.

    Returns
    -------
    int
        Y-coordinate inside the gap.
    """
    inner_start = gap_start + border_margin
    inner_end = gap_end - border_margin

    if inner_end <= inner_start:
        # Gap too small, use midpoint
        return int((gap_start + gap_end) / 2)

    return int((inner_start + inner_end) / 2)


def split_rois_by_max_vertical_gaps(
    image_shape,
    features_list,
    piece_color=None,
    margin=5,
    gap_border_margin=5,
):
    """
    Split image into ROIs using the largest vertical gaps between objects.

    ROI boundaries are placed inside gaps, away from object edges.

    Parameters
    ----------
    image_shape : tuple
        Image shape (height, width, ...).
    features_list : list of dict
        Feature dictionaries with "centroid", "label", and optionally "bbox".
    piece_color : str or None
        'red' -> 2 ROIs; others -> 3 ROIs.
    margin : int
        Margin around object bounding boxes.
    gap_border_margin : int
        Distance from gap borders for boundary placement.

    Returns
    -------
    rois : list of dict
        ROI descriptors.
    boundaries : list of int
        Y-coordinates of boundaries.
    """
    h, w = image_shape[:2]
    color = (piece_color or "").lower()

    # Collect vertical intervals with margin
    intervals = []
    for f in features_list:
        bbox = f.get("bbox", None)
        if bbox is None:
            continue
        x, y, bw, bh = bbox
        top = max(0, int(y) - margin)
        bottom = min(h, int(y + bh) + margin)
        intervals.append((top, bottom))

    # If no objects, use trivial splits
    if not intervals:
        if color == "red":
            boundaries = [0, h // 2, h]
        else:
            boundaries = [0, h // 3, 2 * h // 3, h]
    else:
        # Sort and merge intervals
        intervals.sort(key=lambda t: t[0])
        merged = []
        cur_start, cur_end = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_end:
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))

        # Compute gaps between merged regions
        gaps = []
        if merged[0][0] > 0:
            gaps.append((0, merged[0][0]))
        for (s1, e1), (s2, e2) in zip(merged, merged[1:]):
            if s2 > e1:
                gaps.append((e1, s2))
        if merged[-1][1] < h:
            gaps.append((merged[-1][1], h))

        # If no gaps, use fallback
        if not gaps:
            if color == "red":
                boundaries = [0, h // 2, h]
            else:
                boundaries = [0, h // 3, 2 * h // 3, h]
        else:
            # Sort gaps by size
            gaps_with_size = [((s, e), e - s) for (s, e) in gaps]
            gaps_with_size.sort(key=lambda g: g[1], reverse=True)

            if color == "red":
                # One boundary in largest gap
                (g0_s, g0_e), _ = gaps_with_size[0]
                b1 = _find_safe_line_inside_gap(g0_s, g0_e, gap_border_margin)
                boundaries = [0, b1, h]
            else:
                # Two boundaries in two largest gaps
                if len(gaps_with_size) == 1:
                    (g0_s, g0_e), _ = gaps_with_size[0]
                    mid1 = _find_safe_line_inside_gap(
                        g0_s,
                        g0_s + (g0_e - g0_s) // 2,
                        gap_border_margin,
                    )
                    mid2 = _find_safe_line_inside_gap(
                        g0_s + (g0_e - g0_s) // 2,
                        g0_e,
                        gap_border_margin,
                    )
                    boundaries = [0, mid1, mid2, h]
                else:
                    (g0_s, g0_e), _ = gaps_with_size[0]
                    (g1_s, g1_e), _ = gaps_with_size[1]
                    b1 = _find_safe_line_inside_gap(g0_s, g0_e, gap_border_margin)
                    b2 = _find_safe_line_inside_gap(g1_s, g1_e, gap_border_margin)
                    boundaries = [0, b1, b2, h]

    # Normalize boundaries
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

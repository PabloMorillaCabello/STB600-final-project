"""
Display Module - Visualization and Annotation
=============================================

Functions for displaying images, annotating results, and creating visualizations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Default screen size limits
MAX_HEIGHT = 1080
MAX_WIDTH = 1920


def show_image(img, title="Image", cmap_type=None):
    """
    Display an image correctly in Jupyter notebooks.
    Automatically handles BGR to RGB conversion for color images.

    Parameters
    ----------
    img : np.ndarray
        Image to display (BGR or grayscale).
    title : str
        Title for the plot.
    cmap_type : str or None
        Colormap type for matplotlib (only used for grayscale).
    """
    plt.figure(figsize=(6, 6))

    # If the image has 3 channels (color), convert from BGR to RGB
    if len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb, cmap=cmap_type)
    else:
        # For grayscale images
        plt.imshow(img, cmap='gray')

    plt.title(title)
    plt.axis('off')
    plt.show()


def resize_to_screen(img, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """
    Resize an image to fit within screen dimensions while maintaining aspect ratio.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    max_width : int
        Maximum allowed width.
    max_height : int
        Maximum allowed height.

    Returns
    -------
    np.ndarray
        Resized image (or original if already fits).
    """
    h, w = img.shape[:2]
    if w <= max_width and h <= max_height:
        return img
    scale_w = max_width / float(w)
    scale_h = max_height / float(h)
    scale = min(scale_w, scale_h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def pad_to_same_size(img, target_h, target_w):
    """
    Pad an image with black borders to reach target dimensions.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    target_h : int
        Target height.
    target_w : int
        Target width.

    Returns
    -------
    np.ndarray
        Padded image with centered content.
    """
    h, w = img.shape[:2]
    top = (target_h - h) // 2
    bottom = target_h - h - top
    left = (target_w - w) // 2
    right = target_w - w - left
    return cv2.copyMakeBorder(
        img, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )


def make_row(images):
    """
    Concatenate multiple images horizontally into a single row.

    Parameters
    ----------
    images : list of np.ndarray
        List of images to concatenate (can contain None values).

    Returns
    -------
    np.ndarray or None
        Horizontally concatenated image, or None if no valid images.
    """
    # Filter out None values
    imgs = [im for im in images if im is not None]
    if len(imgs) == 0:
        return None

    # Normalize to 3 channels
    imgs3 = []
    for im in imgs:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        imgs3.append(im)

    # Equalize sizes
    max_h = max(im.shape[0] for im in imgs3)
    max_w = max(im.shape[1] for im in imgs3)
    padded = [pad_to_same_size(im, max_h, max_w) for im in imgs3]
    row = cv2.hconcat(padded)
    return row


def make_grid(rows):
    """
    Stack multiple row images vertically into a grid.

    Parameters
    ----------
    rows : list of np.ndarray
        List of row images (each already horizontally concatenated).

    Returns
    -------
    np.ndarray or None
        Vertically stacked grid image, or None if no valid rows.
    """
    rows = [r for r in rows if r is not None]
    if len(rows) == 0:
        return None
    max_w = max(r.shape[1] for r in rows)
    padded_rows = [pad_to_same_size(r, r.shape[0], max_w) for r in rows]
    grid = cv2.vconcat(padded_rows)
    return grid


def draw_horizontal_rois_from_boundaries(image, boundaries, alpha=0.3):
    """
    Draw horizontal ROIs on an image given boundary y-coordinates.
    Works with 2 or more ROIs (any len(boundaries) >= 2).

    Parameters
    ----------
    image : np.ndarray
        Original BGR image.
    boundaries : list of int
        Y coordinates of boundaries. ROI i covers [boundaries[i], boundaries[i+1]).
    alpha : float
        Transparency factor for the colored overlay.

    Returns
    -------
    np.ndarray
        Image with semi-transparent colored ROIs and ROI index labels.
    """
    vis = image.copy()
    overlay = vis.copy()

    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
    ]

    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    n_rois = len(boundaries) - 1
    if n_rois <= 0:
        return vis

    for i in range(n_rois):
        ys = boundaries[i]
        ye = boundaries[i + 1]
        color = colors[i % len(colors)]

        cv2.rectangle(overlay, (0, ys), (w, ye), color, thickness=-1)

        text_x = 10
        text_y = ys + 30
        cv2.putText(
            overlay,
            f"ROI {i}",
            (text_x, text_y),
            font,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    return vis


def draw_labeled_contours_colored(
    original_bgr,
    contours,
    features_list,
    extent_threshold=0.5,
    small_area_max=1200,
    medium_area_max=3000
):
    """
    Draw contours with different colors depending on their label.

    Colors:
      - marker:  magenta
      - small:   green
      - medium:  yellow
      - large:   red
      - unknown: white

    Parameters
    ----------
    original_bgr : np.ndarray
        Original BGR image.
    contours : list of np.ndarray
        List of contours to draw.
    features_list : list of dict
        Feature dictionaries for each contour.
    extent_threshold : float
        Threshold for marker detection.
    small_area_max : int
        Maximum area for small label.
    medium_area_max : int
        Maximum area for medium label.

    Returns
    -------
    np.ndarray
        Image with labeled contours drawn.
    """
    from .contours import label_contour_by_extent_and_area

    img_out = original_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Color map per label (BGR)
    color_map = {
        "marker": (255, 0, 255),   # magenta
        "small":  (0, 255, 0),     # green
        "medium": (0, 255, 255),   # yellow
        "large":  (0, 0, 255),     # red
        "unknown": (255, 255, 255) # white
    }

    for f, cnt in zip(features_list, contours):
        idx = f["index"]

        # Compute label from features
        label = label_contour_by_extent_and_area(
            f,
            extent_threshold=extent_threshold,
            small_area_max=small_area_max,
            medium_area_max=medium_area_max
        )
        f["label"] = label

        # Choose color based on label
        color = color_map.get(label, (255, 255, 255))

        # Draw contour with label color
        cv2.drawContours(img_out, [cnt], -1, color, 2)

        # Centroid (or bbox center) to place text
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

        text = f"{idx}:{label}"
        cv2.putText(
            img_out,
            text,
            (cx, cy),
            font,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return img_out


def annotate_part_result(
    img_bgr,
    piece_color,
    size_label,
    is_consistent,
    total_value,
    main_contour=None,
    box_scale=1.0,
):
    """
    Annotate an image with detection results in a styled info box.

    Parameters
    ----------
    img_bgr : np.ndarray
        Original BGR image.
    piece_color : str
        Detected piece color ("red", "yellow", "blue").
    size_label : str
        Detected size label ("SMALL", "MEDIUM", "BIG").
    is_consistent : bool
        Whether color and size are consistent.
    total_value : int or None
        Decoded total value from ROIs.
    main_contour : np.ndarray or None
        Main contour for positioning the box.
    box_scale : float
        Scale factor for the info box (1.0 = normal size).

    Returns
    -------
    np.ndarray
        Annotated image with info box.
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # Status
    if is_consistent:
        status_text = "OK"
        status_color = (0, 255, 0)
    else:
        status_text = "ERROR"
        status_color = (0, 0, 255)

    color_map = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "blue": (255, 0, 0),
    }
    border_color = color_map.get((piece_color or "").lower(), (255, 255, 255))

    part_type = (size_label or "UNKNOWN").upper()
    color_str = (piece_color or "NO_COLOR").upper()
    value_str = "N/A" if total_value is None else str(total_value)

    lines = [
        f"{status_text} - {color_str} {part_type}",
        f"Value: {value_str}",
    ]

    # Base parameters (scale 1.0)
    base_font_scale = 1.0
    base_thickness = 2
    base_line_height = 28
    base_padding_x = 30
    base_padding_y_top = 20
    base_padding_y_bottom = 20

    # Scaled parameters
    font_scale = base_font_scale * box_scale
    thickness = max(1, int(round(base_thickness * box_scale)))
    line_height = int(round(base_line_height * box_scale))
    padding_x = int(round(base_padding_x * box_scale))
    padding_y_top = int(round(base_padding_y_top * box_scale))
    padding_y_bottom = int(round(base_padding_y_bottom * box_scale))

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Measure text to calculate total height
    text_sizes = []
    max_text_width = 0
    total_text_height = 0
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_sizes.append((tw, th))
        max_text_width = max(max_text_width, tw)
        total_text_height += line_height

    box_width = max_text_width + 2 * padding_x
    box_height = padding_y_top + padding_y_bottom + total_text_height

    # Box position (near contour, or top-left)
    if main_contour is not None:
        x, y, w_box, h_box = cv2.boundingRect(main_contour)
        base_x = max(10, x)
        base_y = max(10 + box_height, y)
    else:
        base_x = 10
        base_y = box_height + 10

    box_x1 = base_x
    box_y1 = base_y - box_height
    box_x2 = min(box_x1 + box_width, w - 1)
    box_y2 = min(box_y1 + box_height, h - 1)

    # Adjust if overflows top
    if box_y1 < 0:
        box_y1 = 0
        box_y2 = box_height

    # Draw box
    cv2.rectangle(out, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.rectangle(out, (box_x1, box_y1), (box_x2, box_y2), border_color, max(2, thickness))

    # Center text block vertically
    content_height = total_text_height
    content_top = box_y1 + (box_height - content_height) // 2

    # Draw each line centered horizontally
    current_y = content_top
    for i, line in enumerate(lines):
        tw, th = text_sizes[i]

        # x centered
        text_x = box_x1 + (box_width - tw) // 2
        # y base-line for this line
        text_y = current_y + line_height - int((line_height - th) / 2)

        color = status_color if i == 0 else (255, 255, 255)
        cv2.putText(
            out,
            line,
            (text_x, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        current_y += line_height

    return out


# Shared color map for piece annotations (used by both result_viewer and video_processor)
PIECE_COLOR_MAP = {
    "blue": (235, 99, 37),      # BGR for blue display
    "red": (38, 38, 220),       # BGR for red display  
    "yellow": (4, 138, 202),    # BGR for yellow display
    "unknown": (128, 128, 128),
}
FAKE_COLOR = (255, 0, 255)       # Magenta for fake/inconsistent pieces
COUNTED_COLOR = (0, 200, 0)      # Green for counted pieces (video processor)
TRACKING_COLOR = (255, 200, 0)   # Cyan for tracking (video processor)


def draw_piece_box(
    img,
    bounding_box,
    piece_color="unknown",
    label_text=None,
    is_consistent=True,
    is_counted=False,
    draw_center=True,
    font_scale=1.2,
    thickness=2,
    box_thickness=3,
    center_radius=8,
):
    """
    Draw a bounding box with label and center marker for a single piece.
    
    Shared function used by both result_viewer and video_processor for
    consistent styling.
    
    Parameters
    ----------
    img : np.ndarray
        Image to draw on (modified in-place).
    bounding_box : tuple
        (x, y, w, h) bounding box coordinates.
    piece_color : str
        Color of the piece ("blue", "red", "yellow", "unknown").
    label_text : str or None
        Text to display. If None, auto-generated from piece_color.
    is_consistent : bool
        Whether piece is consistent (True) or fake (False).
    is_counted : bool
        Whether piece has been counted (for video processor).
    draw_center : bool
        Whether to draw center marker.
    font_scale : float
        Font scale for label text.
    thickness : int
        Text thickness.
    box_thickness : int
        Bounding box line thickness.
    center_radius : int
        Radius of center marker circle.
    
    Returns
    -------
    tuple
        (cx, cy) center coordinates of the bounding box.
    """
    x, y, w, h = bounding_box
    cx, cy = x + w // 2, y + h // 2
    
    # Determine color based on state
    if not is_consistent:
        color = FAKE_COLOR
    elif is_counted:
        color = COUNTED_COLOR
    else:
        color = PIECE_COLOR_MAP.get(piece_color.lower() if piece_color else "unknown", (128, 128, 128))
    
    # Draw bounding box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, box_thickness)
    
    # Draw center marker (crosshair style)
    if draw_center:
        # Filled circle
        cv2.circle(img, (cx, cy), center_radius, color, -1)
        # White outline for visibility
        cv2.circle(img, (cx, cy), center_radius, (255, 255, 255), 2)
        # Crosshair lines
        line_len = center_radius + 5
        cv2.line(img, (cx - line_len, cy), (cx + line_len, cy), (255, 255, 255), 1)
        cv2.line(img, (cx, cy - line_len), (cx, cy + line_len), (255, 255, 255), 1)
    
    # Draw label if provided
    if label_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Position above the bounding box
        text_x = x
        text_y = y - 15
        if text_y < text_h + 10:
            text_y = y + h + text_h + 15
        
        # Draw background rectangle for readability
        padding = 6
        cv2.rectangle(
            img,
            (text_x - padding, text_y - text_h - padding),
            (text_x + text_w + padding, text_y + baseline + padding),
            color,
            -1
        )
        
        # Draw text in white
        cv2.putText(img, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return cx, cy


def annotate_multiple_pieces(img, results, font_scale=1.5, thickness=3, box_thickness=4):
    """
    Annotate image with bounding boxes and values for all detected pieces.

    Draws labeled bounding boxes around each piece with color-coded borders,
    text labels showing the piece color and decoded value, and center markers.

    Parameters
    ----------
    img : np.ndarray
        Original BGR image.
    results : list
        List of PieceResult objects (or objects with piece_color, size_label,
        is_consistent, total_value, and bounding_box attributes).
    font_scale : float
        Font scale for labels (default 1.5 for video readability).
    thickness : int
        Text thickness (default 3).
    box_thickness : int
        Bounding box line thickness (default 4).

    Returns
    -------
    np.ndarray
        Annotated image with bounding boxes, labels, and center markers.

    Examples
    --------
    >>> from stb600_lib.pipeline import PieceResult
    >>> results = [PieceResult("blue", "BIG", True, 123, contour, (10, 20, 100, 150))]
    >>> annotated = annotate_multiple_pieces(img, results)
    """
    out = img.copy()

    for result in results:
        # Generate label text
        if not result.is_consistent:
            label = "FAKE"
        elif result.total_value is not None:
            label = f"{result.piece_color[0].upper()}: {result.total_value}"
        else:
            label = f"{result.piece_color[0].upper()}: ?"
        
        # Use shared drawing function
        draw_piece_box(
            out,
            bounding_box=result.bounding_box,
            piece_color=result.piece_color,
            label_text=label,
            is_consistent=result.is_consistent,
            is_counted=False,
            draw_center=True,
            font_scale=font_scale,
            thickness=thickness,
            box_thickness=box_thickness,
        )

    return out

import cv2
import numpy as np
import os


# ==================== CONFIGURATION ====================
IMAGE_PATH = "frame_at_13s.png"  # Change this to your image file

# ==================== AREA THRESHOLDS ====================
AREA_MIN = 65000        # below → ignore (noise / reject)
AREA_MAX = 150000       # above → ignore (too large)

SMALL_MAX = 90000
MEDIUM_MAX = 120000


# ==================== COLOR EXPECTED PER TYPE ====================
EXPECTED_COLOR_BY_TYPE = {
    "SMALL": "RED",
    "MEDIUM": "YELLOW",
    "LARGE": "BLUE",
}


HSV_RANGES = {
    "RED": [
        ((0, 120, 70), (10, 255, 255)),
        ((170, 120, 70), (180, 255, 255)),
    ],
    "YELLOW": [((20, 100, 100), (35, 255, 255))],
    "BLUE": [((90, 100, 100), (130, 255, 255))],
}


# ==================== DECODE CONFIG ====================
AREA_THRESHOLDS = {
    "SMALL":  (200, 900),
    "MEDIUM": (901, 2000),
    "BIG":    (2001, 5000),
}


MIN_OBJECT_AREA = 150


MAX_ROIS = {
    "SMALL": 2,
    "MEDIUM": 3,
    "LARGE": 3
}


MIN_GAP = 25  # minimum vertical gap to place an ROI boundary


# ==================== DISPLAY SCALING ====================
def resize_to_screen(img, max_width=1200, max_height=800):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


# ==================== ROTATION & CROP ====================
def crop_and_align_vertical(img, cnt):
    """
    Rotate part to be vertical and crop tightly
    
    Args:
        img: Original image
        cnt: Contour of the part
    
    Returns:
        cropped: Oriented and tightly cropped part image
    """
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect
    
    # If width > height, rotate 90 degrees to make it vertical
    if w > h:
        angle += 90
    
    # Get rotation matrix centered on contour centroid
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    
    # Rotate image
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    # Transform contour to rotated space
    cnt_rot = cv2.transform(cnt, M)
    
    # Get tight bounding box of rotated contour
    x, y, w, h = cv2.boundingRect(cnt_rot)
    
    # Crop to the bounding box
    return rotated[y:y+h, x:x+w]


# ==================== ROI DECODING ====================
def decode_roi_to_number(counts):
    """
    Decode ROI counts to a single digit (0-9)
    
    Number 0 = 1 small
    Number 1 = 2 small
    Number 2 = 3 small
    Number 3 = 4 small
    Number 4 = 1 small + 1 medium
    Number 5 = 2 small + 1 medium
    Number 6 = 3 small + 1 medium
    Number 7 = 1 small + 1 big
    Number 8 = 2 small + 1 big
    Number 9 = 3 small + 1 big
    """
    small = counts['SMALL']
    medium = counts['MEDIUM']
    big = counts['BIG']
    
    if medium == 1 and big == 0:
        return small + 3  # 4, 5, 6
    elif big == 1 and medium == 0:
        return small + 6  # 7, 8, 9
    else:
        return small  # 0, 1, 2, 3 (just small squares)


def decode_part_value(roi_counts, part_type):
    """
    Decode the complete part value based on type
    
    LARGE: 3-digit number from 3 ROIs (e.g., 015, 234)
    MEDIUM: 3-digit number from 3 ROIs, multiplied by 10
    SMALL: Product of 2 ROI numbers
    """
    if part_type == "LARGE":
        # 3-digit number: ROI1 ROI2 ROI3
        digits = [decode_roi_to_number(counts) for counts in roi_counts]
        value = int(f"{digits[0]}{digits[1]}{digits[2]}")
        return value, digits
    
    elif part_type == "MEDIUM":
        # 3-digit number multiplied by 10
        digits = [decode_roi_to_number(counts) for counts in roi_counts]
        value = int(f"{digits[0]}{digits[1]}{digits[2]}") * 10
        return value, digits
    
    elif part_type == "SMALL":
        # Product of 2 ROI numbers
        digits = [decode_roi_to_number(counts) for counts in roi_counts]
        value = digits[0] * digits[1]
        return value, digits
    
    return 0, []


# ==================== DECODE HELPERS ====================
def classify_area(area):
    """Classify area based on thresholds"""
    for k, (lo, hi) in AREA_THRESHOLDS.items():
        if lo <= area <= hi:
            return k
    return None


def remove_green(img):
    """Remove green artifacts from image"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([40, 70, 70]), np.array([85, 255, 255]))
    img_no_green = img.copy()
    img_no_green[mask > 0] = (0, 0, 0)
    return img_no_green


def filter_color(img, color_name):
    """Filter image by expected color"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2], np.uint8)
    for lo, hi in HSV_RANGES[color_name]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    result = cv2.bitwise_and(img, img, mask=mask)
    return result, mask


def find_squares(bin_img):
    """Find contours and classify them"""
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_OBJECT_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cls = classify_area(area)
        if cls:
            squares.append({"rect": (x, y, w, h), "class": cls})
    squares.sort(key=lambda s: s["rect"][1])  # top-to-bottom
    return squares


def calculate_roi_boundaries_gap(squares, img_h, max_rois):
    """Place ROIs in the middle of gaps between square groups"""
    if not squares:
        step = img_h // max_rois
        return [(i*step, (i+1)*step) for i in range(max_rois)]

    tops = [s["rect"][1] for s in squares]
    bottoms = [s["rect"][1] + s["rect"][3] for s in squares]

    boundaries = [0]
    for i in range(len(squares)-1):
        gap = tops[i+1] - bottoms[i]
        if gap >= MIN_GAP:
            boundaries.append((bottoms[i] + tops[i+1]) // 2)
    boundaries.append(img_h)

    # Reduce to max_rois by merging smallest gaps if needed
    while len(boundaries)-1 > max_rois:
        gaps = [boundaries[i+1]-boundaries[i] for i in range(len(boundaries)-1)]
        idx = np.argmin(gaps)
        del boundaries[idx+1]  # merge with next

    roi_positions = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
    return roi_positions


def decode_image(img, part_type):
    """
    Decode cropped ROI image and analyze internal structures
    
    Args:
        img: Cropped image from ROI
        part_type: Type of part ("SMALL", "MEDIUM", "LARGE")
    
    Returns:
        roi_counts: List of dicts with counts per ROI
        overlay: Visualization image
    """
    # Remove green artifacts
    img_clean = remove_green(img)

    # Filter by expected color
    expected_color = EXPECTED_COLOR_BY_TYPE[part_type]
    img_color, mask = filter_color(img_clean, expected_color)

    # Binarize
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(bin_img) > bin_img.size // 2:
        bin_img = cv2.bitwise_not(bin_img)

    # Find squares
    squares = find_squares(bin_img)

    # Calculate ROIs using gap strategy
    max_rois = MAX_ROIS[part_type]
    roi_positions = calculate_roi_boundaries_gap(squares, img.shape[0], max_rois)

    # ==================== Medium top/bottom ROI logic ====================
    roi_counts = []
    counting_direction = "top-down"
    first_roi_adjusted = False
    first_roi_idx = 0

    if part_type == "MEDIUM":
        # Top ROI
        top_roi_start, top_roi_end = roi_positions[0]
        top_squares = [s for s in squares if top_roi_start <= s["rect"][1] <= top_roi_end]
        medium_count_top = sum(1 for s in top_squares if s["class"]=="MEDIUM")
        large_count_top = sum(1 for s in top_squares if s["class"]=="BIG")
        big_bbox_top = any((s["class"]=="MEDIUM" and (s["rect"][2]*s["rect"][3])>AREA_THRESHOLDS["MEDIUM"][1]) for s in top_squares)

        # Bottom ROI
        bottom_roi_start, bottom_roi_end = roi_positions[-1]
        bottom_squares = [s for s in squares if bottom_roi_start <= s["rect"][1] <= bottom_roi_end]
        medium_count_bottom = sum(1 for s in bottom_squares if s["class"]=="MEDIUM")
        large_count_bottom = sum(1 for s in bottom_squares if s["class"]=="BIG")
        big_bbox_bottom = any((s["class"]=="MEDIUM" and (s["rect"][2]*s["rect"][3])>AREA_THRESHOLDS["MEDIUM"][1]) for s in bottom_squares)

        # Decide counting order
        if (medium_count_top==2 or (medium_count_top==1 and large_count_top==1) or big_bbox_top):
            counting_direction = "top-down"
            first_roi_idx = 0
            first_roi_adjusted = True
        elif (medium_count_bottom==2 or (medium_count_bottom==1 and large_count_bottom==1) or big_bbox_bottom):
            counting_direction = "bottom-up"
            first_roi_idx = len(roi_positions)-1
            first_roi_adjusted = True

    # Count squares per ROI
    overlay = img_color.copy()
    if counting_direction == "top-down":
        iter_range = range(len(roi_positions))
    else:
        iter_range = reversed(range(len(roi_positions)))

    for display_idx, idx in enumerate(iter_range):
        y0, y1 = roi_positions[idx]
        roi_squares = [s for s in squares if y0 <= s["rect"][1] <= y1]
        counts = {"SMALL":0, "MEDIUM":0, "BIG":0}
        for s in roi_squares:
            counts[s["class"]] += 1
            x, y, w, h = s["rect"]
            color = {"SMALL": (255,255,0), "MEDIUM": (0,255,255), "BIG": (0,0,255)}[s["class"]]
            cv2.rectangle(overlay, (x,y), (x+w, y+h), color, 2)
            cv2.putText(overlay, s["class"], (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Adjust Medium count for first ROI if needed
        if part_type=="MEDIUM" and first_roi_adjusted and idx==first_roi_idx and counts["MEDIUM"]>0:
            counts["MEDIUM"] -= 1

        roi_counts.append(counts)
        cv2.rectangle(overlay, (0, y0), (img.shape[1], y1), (255,255,0), 1)

    return roi_counts, overlay


# ==================== MAIN DETECTION ====================
def detect_and_decode_credit(image_path):
    """
    Detect a single credit in image and decode its value
    
    Args:
        image_path: Path to image file
    
    Returns:
        result: Dictionary with detection results
    """
    # Load image
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read image: {image_path}")
        return None
    
    display = img.copy()
    
    # Detect credit
    gray = img[:, :, 1]
    _, th = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_MIN or area > AREA_MAX:
            continue
        
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(cnt)
        
        detections.append({
            "cnt": cnt,
            "area": area,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy
        })
    
    if not detections:
        print("No credit detected in image")
        return None
    
    # Find largest credit (most likely the one to analyze)
    credit = max(detections, key=lambda d: d["area"])
    
    # Classify by size
    if credit["area"] < SMALL_MAX:
        part_type = "SMALL"
    elif credit["area"] < MEDIUM_MAX:
        part_type = "MEDIUM"
    else:
        part_type = "LARGE"
    
    # Check size error
    is_size_error = credit["area"] > AREA_MAX
    
    # Detect color
    cnt = credit["cnt"]
    mask_obj = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_obj, [cnt], -1, 255, -1)
    
    detected_color = None
    max_pixels = 0
    
    for color_name, ranges in HSV_RANGES.items():
        mask_color = np.zeros_like(mask_obj)
        for lower, upper in ranges:
            mask_color |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask_color &= mask_obj
        
        pixels = cv2.countNonZero(mask_color)
        if pixels > max_pixels:
            max_pixels = pixels
            detected_color = color_name
    
    expected_color = EXPECTED_COLOR_BY_TYPE[part_type]
    is_color_ok = detected_color == expected_color
    is_ok = is_color_ok and not is_size_error
    
    # Draw on display
    box_color = (0, 255, 0) if is_ok else (0, 0, 255)
    if is_size_error:
        box_color = (0, 0, 255)
        status = "ERROR-SIZE"
    else:
        status = "OK" if is_ok else "NOK"
    
    x, y, w, h = credit["x"], credit["y"], credit["w"], credit["h"]
    cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 3)
    cv2.putText(display, f"{part_type} {detected_color} {status}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
    
    # Decode ROIs
    result = {
        "part_type": part_type,
        "area": credit["area"],
        "detected_color": detected_color,
        "expected_color": expected_color,
        "color_ok": is_color_ok,
        "size_error": is_size_error,
        "status": status,
        "decoded_value": None,
        "roi_digits": None,
        "roi_counts": None,
        "decode_overlay": None,
        "display": display,
        "error": None
    }
    
    if is_size_error:
        result["error"] = "Size too large (exceeds AREA_MAX)"
        return result
    
    # Decode
    try:
        roi_crop = crop_and_align_vertical(img, cnt)
        roi_counts, decode_overlay = decode_image(roi_crop, part_type)
        
        expected_rois = MAX_ROIS[part_type]
        actual_rois = len(roi_counts)
        
        if actual_rois < expected_rois:
            result["error"] = f"Not enough ROIs detected (expected {expected_rois}, found {actual_rois})"
            return result
        
        decoded_value, roi_digits = decode_part_value(roi_counts, part_type)
        
        result["decoded_value"] = decoded_value
        result["roi_digits"] = roi_digits
        result["roi_counts"] = roi_counts
        result["decode_overlay"] = decode_overlay
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


# ==================== DISPLAY RESULTS ====================
def display_results(result):
    """Display detection and decoding results"""
    
    print("\n" + "="*70)
    print("CREDIT DETECTION AND DECODING RESULTS")
    print("="*70)
    
    if result is None:
        print("FAILED: No credit detected")
        print("="*70 + "\n")
        return
    
    print(f"Part Type:        {result['part_type']}")
    print(f"Area:             {result['area']:.0f} pixels")
    print(f"Detected Color:   {result['detected_color']}")
    print(f"Expected Color:   {result['expected_color']}")
    print(f"Color Match:      {'✓ YES' if result['color_ok'] else '✗ NO'}")
    print(f"Status:           {result['status']}")
    
    if result["error"]:
        print(f"\nERROR:            {result['error']}")
        print("="*70 + "\n")
        return
    
    # Display ROI analysis
    print("\n" + "-"*70)
    print("ROI ANALYSIS")
    print("-"*70)
    
    if result['roi_counts']:
        for roi_idx, counts in enumerate(result['roi_counts'], 1):
            print(f"ROI {roi_idx}: SMALL={counts['SMALL']} | MEDIUM={counts['MEDIUM']} | BIG={counts['BIG']}")
    
    # Display decoding
    print("\n" + "-"*70)
    print("DECODING RESULTS")
    print("-"*70)
    
    if result['roi_digits']:
        if result['part_type'] == "SMALL":
            print(f"ROI Digits:       {result['roi_digits'][0]}, {result['roi_digits'][1]}")
            print(f"Calculation:      {result['roi_digits'][0]} × {result['roi_digits'][1]}")
            print(f"DECODED VALUE:    {result['decoded_value']}")
        elif result['part_type'] == "MEDIUM":
            print(f"ROI Digits:       {result['roi_digits'][0]}, {result['roi_digits'][1]}, {result['roi_digits'][2]}")
            base_val = int(f"{result['roi_digits'][0]}{result['roi_digits'][1]}{result['roi_digits'][2]}")
            print(f"Calculation:      {base_val} × 10")
            print(f"DECODED VALUE:    {result['decoded_value']}")
        else:  # LARGE
            print(f"ROI Digits:       {result['roi_digits'][0]}, {result['roi_digits'][1]}, {result['roi_digits'][2]}")
            print(f"Calculation:      {result['roi_digits'][0]}{result['roi_digits'][1]}{result['roi_digits'][2]}")
            print(f"DECODED VALUE:    {result['decoded_value']}")
    
    print("="*70 + "\n")


# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("CREDIT DETECTOR - Single Image Analysis")
    print("="*70)
    print(f"Loading image: {IMAGE_PATH}\n")
    
    # Detect and decode
    result = detect_and_decode_credit(IMAGE_PATH)
    
    # Display results
    display_results(result)
    
    # Display images
    if result:
        # Main detection
        cv2.imshow("Credit Detection", resize_to_screen(result['display']))
        
        # Decoded ROI visualization
        if result['decode_overlay'] is not None:
            cv2.imshow("ROI Decoding", resize_to_screen(result['decode_overlay']))
        
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

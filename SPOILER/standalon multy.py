import cv2
import numpy as np
import os

# ==================== CONFIGURATION ====================
IMAGE_PATH = "frame_at_27.75s.png"

# ==================== AREA THRESHOLDS ====================
AREA_MIN = 65000
AREA_MAX = 150000

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

MIN_GAP = 25


# ==================== HELPERS ====================
def resize_to_screen(img, max_width=1200, max_height=800):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def crop_and_align_vertical(img, cnt):
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect
    if w > h:
        angle += 90

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cnt_rot = cv2.transform(cnt, M)
    x, y, w, h = cv2.boundingRect(cnt_rot)
    return rotated[y:y+h, x:x+w]


def classify_area(area):
    for k, (lo, hi) in AREA_THRESHOLDS.items():
        if lo <= area <= hi:
            return k
    return None


def remove_green(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 70, 70), (85, 255, 255))
    img = img.copy()
    img[mask > 0] = (0, 0, 0)
    return img


def filter_color(img, color_name):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2], np.uint8)
    for lo, hi in HSV_RANGES[color_name]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return cv2.bitwise_and(img, img, mask=mask)


def find_squares(bin_img):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_OBJECT_AREA:
            continue
        cls = classify_area(area)
        if cls:
            x, y, w, h = cv2.boundingRect(c)
            squares.append({"rect": (x, y, w, h), "class": cls})
    squares.sort(key=lambda s: s["rect"][1])
    return squares


def calculate_roi_boundaries_gap(squares, img_h, max_rois):
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

    while len(boundaries)-1 > max_rois:
        gaps = [boundaries[i+1]-boundaries[i] for i in range(len(boundaries)-1)]
        del boundaries[np.argmin(gaps)+1]

    return [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]


def decode_roi_to_number(counts):
    s, m, b = counts["SMALL"], counts["MEDIUM"], counts["BIG"]
    if m == 1:
        return s + 3
    if b == 1:
        return s + 6
    return s


def decode_part_value(roi_counts, part_type):
    digits = [decode_roi_to_number(c) for c in roi_counts]
    if part_type == "SMALL":
        return digits[0] * digits[1], digits
    if part_type == "MEDIUM":
        return int("".join(map(str, digits))) * 10, digits
    return int("".join(map(str, digits))), digits


def decode_image(img, part_type):
    img = remove_green(img)
    img = filter_color(img, EXPECTED_COLOR_BY_TYPE[part_type])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(bin_img) > bin_img.size // 2:
        bin_img = cv2.bitwise_not(bin_img)

    squares = find_squares(bin_img)
    rois = calculate_roi_boundaries_gap(squares, img.shape[0], MAX_ROIS[part_type])

    roi_counts = []
    for y0, y1 in rois:
        counts = {"SMALL":0, "MEDIUM":0, "BIG":0}
        for s in squares:
            if y0 <= s["rect"][1] <= y1:
                counts[s["class"]] += 1
        roi_counts.append(counts)

    return roi_counts


# ==================== MAIN MULTI-CREDIT ====================
def detect_and_decode_all_credits(image_path):
    img = cv2.imread(image_path)
    display = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    gray = img[:, :, 1]
    _, th = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_sum = 0
    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_MIN or area > AREA_MAX:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if area < SMALL_MAX:
            part_type = "SMALL"
        elif area < MEDIUM_MAX:
            part_type = "MEDIUM"
        else:
            part_type = "LARGE"

        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        detected_color = None
        best = 0
        for cname, ranges in HSV_RANGES.items():
            cm = np.zeros_like(mask)
            for lo, hi in ranges:
                cm |= cv2.inRange(hsv, lo, hi)
            cm &= mask
            pix = cv2.countNonZero(cm)
            if pix > best:
                best = pix
                detected_color = cname

        expected = EXPECTED_COLOR_BY_TYPE[part_type]
        status = "OK"
        decoded_value = None

        if detected_color != expected:
            status = "NOK"
        else:
            try:
                roi = crop_and_align_vertical(img, cnt)
                roi_counts = decode_image(roi, part_type)
                if len(roi_counts) == MAX_ROIS[part_type]:
                    decoded_value, _ = decode_part_value(roi_counts, part_type)
                    total_sum += decoded_value
                else:
                    status = "ERROR-ROIS"
            except:
                status = "ERROR-DEC"

        color = (0,255,0) if status == "OK" else (0,0,255)
        cv2.rectangle(display, (x,y), (x+w,y+h), color, 2)
        cv2.putText(display, f"{part_type} {status}", (x,y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if decoded_value is not None:
            cv2.putText(display, f"Val: {decoded_value}", (x, y+h+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        results.append(decoded_value)

    cv2.putText(display, f"TOTAL SUM: {total_sum}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

    return display, total_sum


# ==================== RUN ====================
if __name__ == "__main__":
    display, total = detect_and_decode_all_credits(IMAGE_PATH)
    print("TOTAL SUM =", total)
    cv2.imshow("ALL CREDITS", resize_to_screen(display))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

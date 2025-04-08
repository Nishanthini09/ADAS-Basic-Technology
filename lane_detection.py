import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    # Define a trapezoid (adjust for better focus if needed)
    polygon = np.array([[
        (int(0.1 * width), height),
        (int(0.4 * width), int(0.6 * height)),
        (int(0.6 * width), int(0.6 * height)),
        (int(0.9 * width), height),
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def filter_colors(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    lower_yellow = np.array([15, 30, 115])
    upper_yellow = np.array([35, 204, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=combined_mask)
    return masked_image

def average_lines(lines, img_shape):
    left = []
    right = []

    if lines is None:
        return []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if slope < -0.5:
                left.append((slope, intercept))
            elif slope > 0.5:
                right.append((slope, intercept))

    lanes = []
    for group in [left, right]:
        if len(group) > 0:
            avg_slope, avg_intercept = np.mean(group, axis=0)
            y1 = img_shape[0]
            y2 = int(y1 * 0.6)
            x1 = int((y1 - avg_intercept) / avg_slope)
            x2 = int((y2 - avg_intercept) / avg_slope)
            lanes.append([[x1, y1, x2, y2]])

    return lanes

def draw_lines(image, lines, color=(0, 255, 255), thickness=6):
    line_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return cv2.addWeighted(image, 0.8, line_img, 1, 1)

def detect_lanes(frame):
    filtered = filter_colors(frame)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)

    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=40, minLineLength=40, maxLineGap=100)
    averaged = average_lines(lines, frame.shape)
    output = draw_lines(frame, averaged)
    return output

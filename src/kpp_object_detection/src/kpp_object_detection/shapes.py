import cv2
import numpy as np

MIN_AREA = 800
BG_DIST = 40
EPSILON = 0.03
CIRCLE_THRESH = 0.9

def segment(frame):
    # get the borders of the frame.
    border = np.concatenate([
        frame[0, :, :],
        frame[-1, :, :],
        frame[:, 0, :],
        frame[:, -1, :],
    ])

    background = np.median(border, axis=0).astype(np.float32)
    difference = frame.astype(np.float32) - background
    distance = np.linalg.norm(difference, axis=2)

    mask = np.where(distance > BG_DIST, 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def circle_ratio(contour):
    (_, _), radius = cv2.minEnclosingCircle(contour)
    return cv2.contourArea(contour) / (np.pi * radius * radius)

def classify(contour):
    perimeter = cv2.arcLength(contour, True)

    circularity = circle_ratio(contour)
    if circularity >= CIRCLE_THRESH:
        return "circle"

    approximation = cv2.approxPolyDP(
        contour,
        EPSILON * perimeter,
        True,
    )

    labels = {
        3: "triangle",
        4: "square",
        5: "pentagon",
        6: "hexagon",
    }

    return labels.get(len(approximation), "unknown")

def process(frame):
    out = frame.copy()
    mask = segment(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        label = classify(contour)
        x, y, width, height = cv2.boundingRect(contour)

        cv2.drawContours(out, [contour], -1, (0, 255, 0), 2)
        cv2.putText(
            out,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return out
import cv2
import numpy as np

THRESHOLD = 5
MIN_AREA = 20

def segment(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(
        gray,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )

    return mask

def find_objects(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        boxes.append(cv2.boundingRect(contour))

    return boxes

def process(frame):
    out = frame.copy()
    boxes = find_objects(segment(frame))

    for x, y, width, height in boxes:
        cv2.rectangle(
            out,
            (x, y),
            (x + width, y + height),
            (0, 255, 255),
            2,
        )

    cv2.putText(
        out,
        f"objects: {len(boxes)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    return out
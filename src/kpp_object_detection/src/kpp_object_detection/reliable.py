import cv2
import numpy as np

KERNEL_SIZE = 3
THRESHOLD = 20

def process(frame):
    kernel = np.ones(
        (KERNEL_SIZE, KERNEL_SIZE), 
        dtype=np.uint8
    )
    
    _, mask = cv2.threshold(
        frame,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )
    
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed
import cv2
import numpy as np


BITS = 5
SHIFT = 8 - BITS
MIN_SHARE = 0.005

def pack(frame):
    q = frame >> SHIFT
    return (
        (q[..., 0].astype(np.int32) << (2 * BITS))
        | (q[..., 1].astype(np.int32) << BITS)
        | q[..., 2]
    )

def unpack(key):
    blue = (key >> (2 * BITS)) & ((1 << BITS) - 1)
    green = (key >> BITS) & ((1 << BITS) - 1)
    red = key & ((1 << BITS) - 1)

    return np.array(
        [
            (blue << SHIFT) + (1 << (SHIFT - 1)),
            (green << SHIFT) + (1 << (SHIFT - 1)),
            (red << SHIFT) + (1 << (SHIFT - 1)),
        ],
        dtype=np.uint8
    )

def analyze(key):
    counts = np.bincount(
        key.ravel(),
        minlength=1 << (3 * BITS),
    )

    dominant = int(np.argmax(counts))
    n_colors = int(np.count_nonzero(counts >= MIN_SHARE * key.size))
    share = counts[dominant] / key.size

    return dominant, share, n_colors

def draw_annotation(out, color, share, n_colors):
    swatch_x, swatch_y = 10, 10
    swatch_size = 30
    swatch_bg_x, swatch_bg_y = swatch_x - 5, swatch_y - 5
    swatch_bg_size = (270, 50)
    
    # black background for text legibility
    cv2.rectangle(
        out,
        (swatch_bg_x, swatch_bg_y),
        (swatch_bg_x + swatch_bg_size[0], swatch_bg_y + swatch_bg_size[1]),
        (0, 0, 0),
        thickness=-1,
    )
    
    # fill rectangle with the dominant color
    cv2.rectangle(
        out,
        (swatch_x, swatch_y),
        (swatch_x + swatch_size, swatch_y + swatch_size),
        tuple(int(value) for value in color),
        thickness=-1,
    )

    # draw outline
    cv2.rectangle(
        out,
        (swatch_x, swatch_y),
        (swatch_x + swatch_size, swatch_y + swatch_size),
        (255, 255, 255),
        thickness=1,
    )

    cv2.putText(
        out,
        f"colors: {n_colors}",
        (swatch_x + swatch_size + 10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        out,
        f"dominant share: {share:.1%}",
        (swatch_x + swatch_size + 10, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

def process(frame):
    key = pack(frame)
    dominant, share, n_colors = analyze(key)

    mask = (key == dominant).view(np.uint8)
    out = cv2.copyTo(frame, mask, frame >> 1)

    dominant_color = unpack(dominant)
    draw_annotation(out, dominant_color, share, n_colors)

    return out
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best.onnx"
CONF_THRESHOLD = 0.35
NMS_IOU = 0.45
PAD_VALUE = 114
LABEL = "Hollow Knight"

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
INPUT_NAME = session.get_inputs()[0].name
_, _, INPUT_H, INPUT_W = session.get_inputs()[0].shape

def letterbox(frame):
    h, w = frame.shape[:2]
    scale = min(INPUT_W / w, INPUT_H / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (INPUT_W - new_w) // 2
    pad_y = (INPUT_H - new_h) // 2
    canvas = np.full((INPUT_H, INPUT_W, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y

def to_blob(canvas):
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    return blob.transpose(2, 0, 1)[None]

def detect(frame):
    canvas, scale, pad_x, pad_y = letterbox(frame)
    raw = session.run(None, {INPUT_NAME: to_blob(canvas)})[0][0]

    predictions = raw.T
    predictions = predictions[predictions[:, 4] >= CONF_THRESHOLD]
    if len(predictions) == 0:
        return []

    cx, cy, bw, bh, scores = predictions.T

    x = (cx - bw/2 - pad_x) / scale
    y = (cy - bh/2 - pad_y) / scale
    boxes = np.stack([x, y, bw / scale, bh / scale], axis=1).round().astype(int)

    keep = np.array(cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESHOLD, NMS_IOU)).flatten()
    h, w = frame.shape[:2]
    results = []
    for i in keep:
        bx, by, bwid, bhei = boxes[i]
        results.append((
            int(np.clip(bx, 0, w - 1)), int(np.clip(by, 0, h - 1)),
            int(np.clip(bx + bwid, 0, w - 1)), int(np.clip(by + bhei, 0, h - 1)),
            float(scores[i]),
        ))
    return results

def process(frame):
    out = frame.copy()
    for x1, y1, x2, y2, conf in detect(frame):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{LABEL} {conf:.2f}", (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out
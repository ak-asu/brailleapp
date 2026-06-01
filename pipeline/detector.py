from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


def load_onnx_session(model_path: str):
    """
    Load ONNX InferenceSession once. Returns None if file missing or ort unavailable.
    Decorated with @st.cache_resource in app.py — called as a plain function here
    so this module stays importable without streamlit.
    """
    if not _ORT_AVAILABLE:
        return None
    if not Path(model_path).exists():
        return None
    try:
        return ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
    except Exception:
        return None


def detect_hough(
    binary: np.ndarray,
    gray: np.ndarray | None = None,
) -> list[tuple[float, float, float]]:
    """
    Detect Braille dots using Hough Circle Transform.

    Primary: HOUGH_GRADIENT on binary (fast, works on binarised images).
    If gray is provided, also runs HOUGH_GRADIENT_ALT on the gradient image
    and merges results for higher recall.

    Input:  binary — preprocessed binary image (from preprocessor).
            gray   — optional raw grayscale (from preprocessor) for ALT path.
    Returns: list of (x, y, radius) in pixel coordinates.
    """
    seen: set[tuple[int, int]] = set()
    results: list[tuple[float, float, float]] = []

    def _add(circles_raw):
        if circles_raw is None:
            return
        for x, y, r in circles_raw[0]:
            key = (int(round(x / 4)) * 4, int(round(y / 4)) * 4)
            if key not in seen:
                seen.add(key)
                results.append((float(x), float(y), float(r)))

    # Primary: HOUGH_GRADIENT works reliably on binarised images
    _add(cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=8,
        param1=50,
        param2=12,
        minRadius=3,
        maxRadius=12,
    ))

    # Secondary: HOUGH_GRADIENT_ALT on grayscale for additional recall
    if gray is not None:
        _add(cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT_ALT,
            dp=1.5,
            minDist=8,
            param1=100,
            param2=0.7,
            minRadius=3,
            maxRadius=12,
        ))

    return results


def _nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45
) -> list[int]:
    """
    Non-maximum suppression. Returns indices of kept boxes.
    boxes: (N, 4) as [x1, y1, x2, y2]
    scores: (N,)
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_threshold]
    return keep


def detect_onnx(
    frame_bgr: np.ndarray,
    session,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.45,
) -> list[tuple[float, float, float]]:
    """
    YOLOv8n ONNX dot detection.

    Input:  frame_bgr — BGR 640×640 image.
            session   — onnxruntime InferenceSession.
    Returns: list of (x_center, y_center, confidence).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 640, 640]

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]  # [1, 5, 8400]

    preds = output[0].T  # [8400, 5]: (cx, cy, w, h, conf)
    mask = preds[:, 4] > conf_threshold
    preds = preds[mask]
    if len(preds) == 0:
        return []

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    scores = preds[:, 4]

    kept = _nms(boxes, scores, iou_threshold)
    return [(float(cx[i]), float(cy[i]), float(scores[i])) for i in kept]


def ensemble(
    hough_dots: list[tuple[float, float, float]],
    onnx_dots: list[tuple[float, float, float]],
    proximity_px: float = 8.0,
    boost: float = 0.15,
) -> list[tuple[float, float, float]]:
    """
    Merge Hough and ONNX detections by proximity.

    hough_dots: [(x, y, radius), ...]
    onnx_dots:  [(x, y, confidence), ...]
    Returns:    [(x, y, confidence), ...]

    - ONNX detection matched by a Hough dot within proximity_px → confidence += boost
    - ONNX detection with no Hough match → kept as-is
    - Hough detection with no ONNX match → added with confidence=0.45
    """
    result: list[tuple[float, float, float]] = []
    used_hough: set[int] = set()

    for ox, oy, oconf in onnx_dots:
        best_idx, best_dist = None, proximity_px
        for i, (hx, hy, _) in enumerate(hough_dots):
            dist = ((ox - hx) ** 2 + (oy - hy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None:
            used_hough.add(best_idx)
            result.append((ox, oy, min(1.0, oconf + boost)))
        else:
            result.append((ox, oy, oconf))

    for i, (hx, hy, hr) in enumerate(hough_dots):
        if i not in used_hough and 3 <= hr <= 15:
            result.append((hx, hy, 0.45))

    return result

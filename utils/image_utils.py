import base64
import cv2
import numpy as np


def decode_frame(b64_jpeg: str) -> np.ndarray | None:
    """Decode base64 JPEG string to BGR numpy array. Returns None on failure."""
    try:
        data = base64.b64decode(b64_jpeg)
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame  # None if imdecode fails
    except Exception:
        return None


def encode_frame(frame: np.ndarray, quality: int = 85) -> str:
    """Encode BGR numpy array to base64 JPEG string."""
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode('utf-8')


def letterbox(
    frame: np.ndarray, size: int = 640
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize frame to size×size with black padding, preserving aspect ratio.
    Returns: (padded_frame, scale_factor, (pad_x, pad_y))
    scale_factor: multiply padded coords by 1/scale to get original coords.
    """
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    channels = frame.shape[2] if frame.ndim == 3 else 1
    if channels > 1:
        canvas = np.zeros((size, size, channels), dtype=np.uint8)
    else:
        canvas = np.zeros((size, size), dtype=np.uint8)

    pad_y = (size - new_h) // 2
    pad_x = (size - new_w) // 2
    canvas[pad_y: pad_y + new_h, pad_x: pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def draw_dots(
    frame: np.ndarray,
    dots: list[tuple[float, float, float]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw circle overlays for detected dot positions.
    dots: list of (x, y, radius)
    Returns a copy of frame with circles drawn.
    """
    out = frame.copy()
    for x, y, r in dots:
        cv2.circle(out, (int(x), int(y)), max(int(r), 4), color, thickness)
    return out


def map_dots_from_letterbox(
    dots: list[tuple[float, float, float]],
    scale: float,
    pad: tuple[int, int],
    original_shape: tuple[int, ...],
) -> list[tuple[float, float, float]]:
    """Map dot coordinates from a letterboxed image back to the original frame."""
    if scale <= 0:
        return []

    pad_x, pad_y = pad
    h, w = original_shape[:2]
    mapped: list[tuple[float, float, float]] = []
    for x, y, r in dots:
        ox = (x - pad_x) / scale
        oy = (y - pad_y) / scale
        if 0 <= ox < w and 0 <= oy < h:
            mapped.append((float(ox), float(oy), float(r) / scale))
    return mapped

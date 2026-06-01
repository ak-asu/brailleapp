import cv2
import numpy as np


def blur_score(gray: np.ndarray) -> float:
    """Laplacian variance — higher = sharper. Values below 80 indicate blur."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray: np.ndarray) -> float:
    """Mean pixel intensity of a grayscale image (0–255)."""
    return float(gray.mean())


def dot_density(dots: list, frame_w: int, frame_h: int) -> float:
    """Detected dots per megapixel. Used to gauge camera distance."""
    if frame_w == 0 or frame_h == 0:
        return 0.0
    return len(dots) / (frame_w * frame_h) * 1_000_000


def get_guidance(blur: float, bright: float, density: float, skew: float) -> dict:
    """
    Return a guidance message based on current frame quality metrics.
    Priority order: blur → brightness → distance → skew → ok.
    """
    if blur < 50:
        return {"status": "warn", "message": "Hold camera steady"}
    if blur < 80:
        return {"status": "warn", "message": "Almost — hold steadier"}
    if bright < 40:
        return {"status": "warn", "message": "Need more light"}
    if bright > 210:
        return {"status": "warn", "message": "Too bright — find shade"}
    if density < 5:
        return {"status": "warn", "message": "Move closer to the Braille"}
    if density > 200:
        return {"status": "warn", "message": "Move back slightly"}
    if abs(skew) > 15:
        return {"status": "warn", "message": "Tilt camera — align with page edge"}
    return {"status": "ok", "message": "Good — scanning..."}

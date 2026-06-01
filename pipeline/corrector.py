import cv2
import numpy as np


def estimate_skew(
    dots: list[tuple[float, float, float]], row_labels: np.ndarray
) -> float:
    """
    Estimate page skew angle in degrees from dot row geometry.
    Fits a line to each row cluster and returns the median angle.
    Positive angle = clockwise tilt; negative = counter-clockwise.
    Returns 0.0 if not enough data.
    """
    if len(dots) < 4:
        return 0.0

    xs = np.array([d[0] for d in dots], dtype=float)
    ys = np.array([d[1] for d in dots], dtype=float)

    angles: list[float] = []
    for label in np.unique(row_labels):
        mask = row_labels == label
        rx, ry = xs[mask], ys[mask]
        if len(rx) < 2:
            continue
        coeffs = np.polyfit(rx, ry, 1)   # y = m*x + b
        angle = float(np.degrees(np.arctan(coeffs[0])))
        angles.append(angle)

    return float(np.median(angles)) if angles else 0.0


def correct_skew(frame: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate frame to correct skew. Angles below 1° are ignored (no-op).
    Preserves original frame dimensions with BORDER_REPLICATE fill.
    """
    if abs(angle_deg) < 1.0:
        return frame
    h, w = frame.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    return cv2.warpAffine(
        frame, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

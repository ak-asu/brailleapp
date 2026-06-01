import cv2
import numpy as np


def preprocess(
    frame_bgr: np.ndarray, clip_limit: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare a BGR frame for Braille dot detection.

    Input:  frame_bgr — BGR image, should be letterboxed to 640×640 before calling.
    Returns: (binary_cleaned, gray)
        binary_cleaned — thresholded + morphed image suitable for Hough detection
        gray           — raw grayscale for quality signal computation
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned, gray

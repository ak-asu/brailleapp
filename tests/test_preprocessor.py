import numpy as np
import cv2
import pytest
from pipeline.preprocessor import preprocess


def _make_frame(h=640, w=640, val=128) -> np.ndarray:
    frame = np.full((h, w, 3), val, dtype=np.uint8)
    return frame


def test_preprocess_output_shapes():
    frame = _make_frame()
    binary, gray = preprocess(frame)
    assert binary.shape == (640, 640)
    assert gray.shape == (640, 640)


def test_preprocess_binary_values():
    """Binary output must contain only 0 and 255."""
    frame = _make_frame()
    binary, _ = preprocess(frame)
    unique = set(binary.flatten().tolist())
    assert unique.issubset({0, 255})


def test_preprocess_gray_is_uint8():
    frame = _make_frame()
    _, gray = preprocess(frame)
    assert gray.dtype == np.uint8


def test_preprocess_custom_clip_limit():
    """High clip_limit should not crash and still return correct shapes."""
    frame = _make_frame()
    binary, gray = preprocess(frame, clip_limit=3.5)
    assert binary.shape == (640, 640)


def test_preprocess_dark_frame():
    frame = _make_frame(val=5)
    binary, gray = preprocess(frame)
    assert binary.shape == (640, 640)


def test_preprocess_on_non_square_fails_gracefully():
    """Frame must be pre-letterboxed to 640×640 before calling preprocess."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # This should work (no size enforcement in preprocessor) but document intent
    binary, gray = preprocess(frame)
    assert binary.shape == (480, 640)

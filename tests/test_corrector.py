import numpy as np
import pytest
import cv2
from pipeline.corrector import estimate_skew, correct_skew
from pipeline.grid import _cluster_1d


def _tilted_dots(angle_deg: float, n_cols: int = 5, spacing: int = 20) -> list:
    """Generate dots in a horizontal line tilted by angle_deg."""
    rad = np.radians(angle_deg)
    dots = []
    for i in range(n_cols):
        x = 100 + i * spacing * np.cos(rad)
        y = 100 + i * spacing * np.sin(rad)
        dots.append((float(x), float(y), 0.9))
    return dots


def test_estimate_skew_horizontal_line():
    dots = _tilted_dots(0.0, n_cols=6)
    ys = np.array([d[1] for d in dots])
    row_labels = _cluster_1d(ys, gap=10.0)
    angle = estimate_skew(dots, row_labels)
    assert abs(angle) < 2.0


def test_estimate_skew_tilted_line():
    dots = _tilted_dots(10.0, n_cols=6)
    ys = np.array([d[1] for d in dots])
    row_labels = _cluster_1d(ys, gap=5.0)
    angle = estimate_skew(dots, row_labels)
    assert abs(angle - 10.0) < 5.0


def test_estimate_skew_too_few_dots():
    dots = [(10.0, 10.0, 0.9), (20.0, 20.0, 0.9)]
    labels = np.array([0, 0])
    angle = estimate_skew(dots, labels)
    assert angle == pytest.approx(0.0)


def test_correct_skew_returns_same_shape():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    result = correct_skew(frame, 10.0)
    assert result.shape == frame.shape


def test_correct_skew_trivial_angle_returns_same():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    frame[100, 100] = 255
    result = correct_skew(frame, 0.5)  # < 1° threshold
    assert np.array_equal(result, frame)


def test_correct_skew_rotates_content():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 100), 5, (255, 255, 255), -1)
    result = correct_skew(frame, 15.0)
    assert not np.array_equal(result, frame)

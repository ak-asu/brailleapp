import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock
from pipeline.detector import detect_hough, _nms, detect_onnx, ensemble


def _preprocessed_dot_image(cx: int, cy: int, r: int = 8, size: int = 640) -> np.ndarray:
    """
    Simulate a preprocessor output: a light-grey background with a dark circular dot.
    This matches what preprocess() returns for a real Braille dot on paper.
    """
    # Light grey background (simulating paper)
    img = np.full((size, size, 3), 220, dtype=np.uint8)
    # Dark circle (simulating a raised Braille dot casting a shadow)
    cv2.circle(img, (cx, cy), r, (30, 30, 30), -1)
    # Blur to simulate real camera texture
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Run through actual preprocessor to get realistic binary image
    from pipeline.preprocessor import preprocess
    binary, _ = preprocess(img)
    return binary


def test_detect_hough_finds_dot():
    img = _preprocessed_dot_image(320, 320, r=8)
    dots = detect_hough(img)
    assert len(dots) >= 1
    xs = [d[0] for d in dots]
    ys = [d[1] for d in dots]
    assert any(abs(x - 320) < 25 and abs(y - 320) < 25 for x, y in zip(xs, ys))


def test_detect_hough_empty_image():
    img = np.zeros((640, 640), dtype=np.uint8)
    dots = detect_hough(img)
    assert dots == []


def test_detect_hough_returns_list_of_tuples():
    img = _preprocessed_dot_image(200, 200, r=6)
    dots = detect_hough(img)
    for item in dots:
        assert len(item) == 3  # (x, y, r)


def test_nms_removes_duplicates():
    # Two nearly identical boxes should collapse to one
    boxes = np.array([
        [10, 10, 30, 30],
        [12, 12, 32, 32],
        [200, 200, 220, 220],
    ], dtype=float)
    scores = np.array([0.9, 0.8, 0.95])
    kept = _nms(boxes, scores, iou_threshold=0.45)
    assert len(kept) == 2  # two distinct regions kept


def test_nms_empty():
    kept = _nms(np.zeros((0, 4)), np.array([]), iou_threshold=0.45)
    assert kept == []


def test_detect_onnx_uses_session():
    """ONNX path calls session.run and parses output correctly."""
    raw_out = np.zeros((1, 5, 8400), dtype=np.float32)
    raw_out[0, 0, 0] = 320.0   # x_center
    raw_out[0, 1, 0] = 320.0   # y_center
    raw_out[0, 2, 0] = 10.0    # width
    raw_out[0, 3, 0] = 10.0    # height
    raw_out[0, 4, 0] = 0.95    # confidence

    mock_input = MagicMock()
    mock_input.name = "images"
    session = MagicMock()
    session.get_inputs.return_value = [mock_input]
    session.run.return_value = [raw_out]

    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    dots = detect_onnx(frame, session)
    assert len(dots) >= 1
    assert abs(dots[0][0] - 320.0) < 5
    assert abs(dots[0][1] - 320.0) < 5
    assert dots[0][2] > 0.8  # confidence


def test_ensemble_boosts_confirmed_dots():
    hough = [(100.0, 100.0, 6.0)]
    onnx = [(102.0, 101.0, 0.7)]  # within 8px of hough dot
    result = ensemble(hough, onnx)
    assert len(result) == 1
    assert result[0][2] > 0.7  # confidence boosted


def test_ensemble_includes_hough_only():
    hough = [(300.0, 300.0, 6.0)]
    onnx = []  # no ONNX detections
    result = ensemble(hough, onnx)
    assert len(result) == 1
    assert result[0][2] == pytest.approx(0.45)


def test_ensemble_includes_onnx_only():
    hough = []
    onnx = [(200.0, 200.0, 0.8)]
    result = ensemble(hough, onnx)
    assert len(result) == 1
    assert result[0][2] == pytest.approx(0.8)

import base64
import cv2
import numpy as np
import pytest
from utils.image_utils import (
    decode_frame,
    encode_frame,
    letterbox,
    draw_dots,
    map_dots_from_letterbox,
)


def _make_jpeg_b64(w=100, h=80) -> str:
    """Create a solid blue 100×80 JPEG as base64."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)  # blue in BGR
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode('utf-8')


def test_decode_frame_returns_ndarray():
    b64 = _make_jpeg_b64()
    result = decode_frame(b64)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3


def test_decode_frame_invalid_returns_none():
    result = decode_frame("notbase64!!!")
    assert result is None


def test_encode_decode_roundtrip_shape():
    original = np.zeros((100, 80, 3), dtype=np.uint8)
    original[10:20, 10:20] = 128
    b64 = encode_frame(original)
    recovered = decode_frame(b64)
    assert recovered.shape == original.shape


def test_letterbox_output_is_square():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out, scale, (pad_x, pad_y) = letterbox(frame, size=640)
    assert out.shape == (640, 640, 3)


def test_letterbox_scale_and_padding():
    # 480×640 → 640×640: scale=1.0, pad_y=80, pad_x=0
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out, scale, (pad_x, pad_y) = letterbox(frame, size=640)
    assert scale == pytest.approx(1.0)
    assert pad_y == 80
    assert pad_x == 0


def test_letterbox_portrait():
    # 640×480 portrait → 640×640: scale=1.0, pad_x=80, pad_y=0
    frame = np.zeros((640, 480, 3), dtype=np.uint8)
    out, scale, (pad_x, pad_y) = letterbox(frame, size=640)
    assert scale == pytest.approx(1.0)
    assert pad_x == 80
    assert pad_y == 0


def test_draw_dots_returns_copy():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dots = [(50.0, 50.0, 5.0)]
    result = draw_dots(frame, dots)
    assert result is not frame  # must be a copy
    assert result.shape == frame.shape


def test_draw_dots_draws_circle():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    dots = [(100.0, 100.0, 8.0)]
    result = draw_dots(frame, dots)
    # At least some pixels changed from black
    assert result.sum() > 0


def test_map_dots_from_letterbox_removes_padding_and_scale():
    dots = [(320.0, 320.0, 10.0), (10.0, 10.0, 5.0)]
    mapped = map_dots_from_letterbox(
        dots,
        scale=1.0,
        pad=(0, 80),
        original_shape=(480, 640, 3),
    )
    assert mapped == [(320.0, 240.0, 10.0)]

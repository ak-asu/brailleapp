import numpy as np
import cv2
import pytest
from utils.quality import blur_score, brightness, dot_density, get_guidance


def test_blur_score_sharp_image_is_high():
    # Sharp black/white checkerboard has high Laplacian variance
    img = np.zeros((64, 64), dtype=np.uint8)
    img[::8, :] = 255
    img[:, ::8] = 255
    assert blur_score(img) > 200


def test_blur_score_blurry_image_is_low():
    img = np.full((64, 64), 128, dtype=np.uint8)  # uniform grey = no edges
    assert blur_score(img) < 10


def test_brightness_dark():
    img = np.zeros((100, 100), dtype=np.uint8)
    assert brightness(img) == pytest.approx(0.0)


def test_brightness_full():
    img = np.full((100, 100), 255, dtype=np.uint8)
    assert brightness(img) == pytest.approx(255.0)


def test_dot_density_zero_dots():
    assert dot_density([], 640, 640) == pytest.approx(0.0)


def test_dot_density_nonzero():
    dots = [(100.0, 100.0, 0.9)] * 10
    result = dot_density(dots, 640, 640)
    assert result == pytest.approx(10 / (640 * 640) * 1_000_000)


def test_guidance_blur_too_low():
    result = get_guidance(blur=30, bright=128, density=20, skew=0)
    assert result["status"] == "warn"
    assert "steady" in result["message"].lower()


def test_guidance_low_light():
    result = get_guidance(blur=200, bright=20, density=20, skew=0)
    assert result["status"] == "warn"
    assert "light" in result["message"].lower()


def test_guidance_too_bright():
    result = get_guidance(blur=200, bright=220, density=20, skew=0)
    assert result["status"] == "warn"
    assert "bright" in result["message"].lower()


def test_guidance_too_far():
    result = get_guidance(blur=200, bright=128, density=2, skew=0)
    assert result["status"] == "warn"
    assert "closer" in result["message"].lower()


def test_guidance_too_close():
    result = get_guidance(blur=200, bright=128, density=250, skew=0)
    assert result["status"] == "warn"
    assert "back" in result["message"].lower()


def test_guidance_skewed():
    result = get_guidance(blur=200, bright=128, density=20, skew=20)
    assert result["status"] == "warn"
    assert "tilt" in result["message"].lower()


def test_guidance_all_ok():
    result = get_guidance(blur=200, bright=128, density=20, skew=2)
    assert result["status"] == "ok"
    assert "scanning" in result["message"].lower()

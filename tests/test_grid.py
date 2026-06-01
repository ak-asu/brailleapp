import numpy as np
import pytest
from pipeline.grid import _cluster_1d, reconstruct_grid


# Braille dot-to-bit mapping (standard):
# Dot 1 (top-left)  → bit 0
# Dot 2 (mid-left)  → bit 1
# Dot 3 (bot-left)  → bit 2
# Dot 4 (top-right) → bit 3
# Dot 5 (mid-right) → bit 4
# Dot 6 (bot-right) → bit 5
#
# 'a' = dot 1 only → 0b000001 → U+2801 → ⠁
# 'b' = dots 1,2   → 0b000011 → U+2803 → ⠃
# 'h' = dots 1,2,5 → 0b010011 → U+2813 → ⠓


def _cell_dots(
    origin_x: int, origin_y: int, spacing: int, pattern: int
) -> list[tuple[float, float, float]]:
    """
    Build dot list for a single Braille cell.
    origin_x, origin_y: top-left dot position
    spacing: pixels between adjacent dot rows/columns
    pattern: 6-bit integer (bit 0=dot1 … bit 5=dot6)
    """
    positions = [
        (0, 0),  # dot 1: bit 0
        (0, 1),  # dot 2: bit 1
        (0, 2),  # dot 3: bit 2
        (1, 0),  # dot 4: bit 3
        (1, 1),  # dot 5: bit 4
        (1, 2),  # dot 6: bit 5
    ]
    dots = []
    for bit, (col, row) in enumerate(positions):
        if pattern & (1 << bit):
            x = float(origin_x + col * spacing)
            y = float(origin_y + row * spacing)
            dots.append((x, y, 0.9))
    return dots


def test_cluster_1d_two_groups():
    values = np.array([10.0, 12.0, 100.0, 102.0])
    labels = _cluster_1d(values, gap=20.0)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cluster_1d_single_element():
    values = np.array([50.0])
    labels = _cluster_1d(values, gap=20.0)
    assert labels.tolist() == [0]


def test_cluster_1d_all_same_cluster():
    values = np.array([5.0, 8.0, 11.0])
    labels = _cluster_1d(values, gap=20.0)
    assert len(set(labels.tolist())) == 1


def test_reconstruct_grid_letter_a():
    # 'a' = dot 1 only = U+2801
    dots = _cell_dots(origin_x=100, origin_y=100, spacing=20, pattern=0b000001)
    braille_str, conf = reconstruct_grid(dots)
    assert '⠁' in braille_str  # ⠁


def test_reconstruct_grid_letter_b():
    # 'b' = dots 1,2 = U+2803
    dots = _cell_dots(origin_x=100, origin_y=100, spacing=20, pattern=0b000011)
    braille_str, conf = reconstruct_grid(dots)
    assert '⠃' in braille_str  # ⠃


def test_reconstruct_grid_letter_h():
    # 'h' = dots 1,2,5 = U+2813
    dots = _cell_dots(origin_x=100, origin_y=100, spacing=20, pattern=0b010011)
    braille_str, conf = reconstruct_grid(dots)
    assert '⠓' in braille_str  # ⠓


def test_reconstruct_grid_two_cells():
    # 'a' followed by 'b': two cells side by side with a gap
    dots_a = _cell_dots(origin_x=100, origin_y=100, spacing=20, pattern=0b000001)
    dots_b = _cell_dots(origin_x=200, origin_y=100, spacing=20, pattern=0b000011)
    braille_str, conf = reconstruct_grid(dots_a + dots_b)
    assert len(braille_str.replace('\n', '')) == 2


def test_reconstruct_grid_too_few_dots_returns_empty():
    dots = [(100.0, 100.0, 0.9), (110.0, 100.0, 0.9)]  # only 2 dots
    braille_str, conf = reconstruct_grid(dots, min_dots=3)  # explicit quality gate
    assert braille_str == ""


def test_reconstruct_grid_confidence_in_range():
    dots = _cell_dots(origin_x=100, origin_y=100, spacing=20, pattern=0b000001)
    _, conf = reconstruct_grid(dots)
    assert 0.0 <= conf <= 1.0

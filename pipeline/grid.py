import numpy as np
from scipy.spatial.distance import cdist


def _cluster_1d(values: np.ndarray, gap: float) -> np.ndarray:
    """
    Cluster 1D values by gaps greater than `gap`.
    Returns integer cluster labels (0-indexed), same length as values.
    """
    if len(values) == 0:
        return np.array([], dtype=int)
    idx = np.argsort(values)
    sorted_v = values[idx]
    labels = np.zeros(len(values), dtype=int)
    current = 0
    for i in range(1, len(sorted_v)):
        if sorted_v[i] - sorted_v[i - 1] > gap:
            current += 1
        labels[idx[i]] = current
    labels[idx[0]] = 0
    return labels


def _group_consecutive(
    clusters: list[int], medians: dict[int, float], max_gap: float
) -> list[list[int]]:
    """
    Group consecutive cluster IDs into super-groups.
    Adjacent clusters whose median positions differ by <= max_gap are in the same group.
    Returns a list of groups, each group being a list of cluster IDs.
    """
    if not clusters:
        return []
    groups: list[list[int]] = []
    current: list[int] = [clusters[0]]
    for i in range(1, len(clusters)):
        gap = medians[clusters[i]] - medians[clusters[i - 1]]
        if gap <= max_gap:
            current.append(clusters[i])
        else:
            groups.append(current)
            current = [clusters[i]]
    groups.append(current)
    return groups


def _estimate_dot_spacing(xs: np.ndarray, ys: np.ndarray) -> float:
    """Estimate intra-cell dot spacing while ignoring large inter-cell gaps."""
    if len(xs) < 2:
        return 20.0

    diffs: list[float] = []
    for vals in (xs, ys):
        unique = np.unique(np.round(vals, 1))
        if len(unique) < 2:
            continue
        axis_diffs = np.diff(np.sort(unique))
        diffs.extend(float(d) for d in axis_diffs if 5.0 <= d <= 45.0)

    if diffs:
        return float(np.median(diffs))

    pts = np.stack([xs, ys], axis=1)
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    nearest = float(np.median(dists.min(axis=1)))
    return nearest if 5.0 <= nearest <= 45.0 else 20.0


def _group_indices(indices: list[int], max_step: int = 2) -> list[list[int]]:
    """Group snapped lattice indices into cells/lines, preserving empty slots."""
    if not indices:
        return []
    ordered = sorted(set(indices))
    groups: list[list[int]] = [[ordered[0]]]
    for idx in ordered[1:]:
        if idx - groups[-1][-1] <= max_step:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def reconstruct_grid(
    dots: list[tuple[float, float, float]],
    min_dots: int = 1,
    dot_spacing: float | None = None,
    origin: tuple[float, float] | None = None,
) -> tuple[str, float]:
    """
    Convert confirmed dot positions to a Braille Unicode string.

    dots: [(x, y, confidence), ...]
    min_dots: minimum number of dots required (default 1 for max recall).
    dot_spacing: optional known spacing between adjacent dot rows/columns.
    origin: optional top-left lattice origin for preserving sparse dot positions.
    Returns: (braille_unicode_string, mean_confidence)
             Empty string + 0.0 if not enough dots or clustering fails.

    Braille dot numbering:
        1 4
        2 5
        3 6
    Bit 0 = dot 1, bit 1 = dot 2, ..., bit 5 = dot 6.
    Unicode: U+2800 + bitmask.

    The implementation snaps dots to a spacing-based lattice instead of using
    only observed rows/columns. That preserves missing slots in sparse cells.
    """
    if len(dots) < min_dots:
        return "", 0.0

    xs = np.array([d[0] for d in dots], dtype=float)
    ys = np.array([d[1] for d in dots], dtype=float)
    confs = np.array([d[2] for d in dots], dtype=float)

    spacing = float(dot_spacing) if dot_spacing is not None else _estimate_dot_spacing(xs, ys)
    if spacing < 1.0:
        return "", 0.0

    origin_x = float(origin[0]) if origin is not None else float(xs.min())
    origin_y = float(origin[1]) if origin is not None else float(ys.min())

    col_indices = np.rint((xs - origin_x) / spacing).astype(int)
    row_indices = np.rint((ys - origin_y) / spacing).astype(int)

    row_lines = _group_indices(row_indices.tolist(), max_step=2)
    col_cells = _group_indices(col_indices.tolist(), max_step=2)
    if not row_lines or not col_cells:
        return "", 0.0

    presence: dict[tuple[int, int], bool] = {}
    for i in range(len(dots)):
        presence[(int(row_indices[i]), int(col_indices[i]))] = True

    braille_chars: list[str] = []
    column_pitch = 5
    row_pitch = 4

    for line_idx, row_line in enumerate(row_lines):
        row_base = (
            round(row_line[0] / row_pitch) * row_pitch
            if origin is not None
            else row_line[0]
        )
        for col_cell in col_cells:
            col_base = (
                round(col_cell[0] / column_pitch) * column_pitch
                if origin is not None
                else col_cell[0]
            )
            bitmask = 0
            for rc in row_line:
                dot_row = rc - row_base
                if not 0 <= dot_row <= 2:
                    continue
                for cc in col_cell:
                    dot_col = cc - col_base
                    if not 0 <= dot_col <= 1:
                        continue
                    if presence.get((rc, cc), False):
                        bit = dot_row + dot_col * 3
                        bitmask |= (1 << bit)
            braille_chars.append(chr(0x2800 + bitmask))

        if line_idx < len(row_lines) - 1:
            braille_chars.append("\n")

    return "".join(braille_chars), float(confs.mean())

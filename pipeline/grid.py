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
    Adjacent clusters whose median positions differ by ≤ max_gap are in the same group.
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


def reconstruct_grid(
    dots: list[tuple[float, float, float]], min_dots: int = 1
) -> tuple[str, float]:
    """
    Convert confirmed dot positions to a Braille Unicode string.

    dots: [(x, y, confidence), ...]
    min_dots: minimum number of dots required (default 1 for max recall).
    Returns: (braille_unicode_string, mean_confidence)
             Empty string + 0.0 if not enough dots or clustering fails.

    Braille dot numbering (standard):
        1 4
        2 5
        3 6
    Bit 0 = dot 1, bit 1 = dot 2, …, bit 5 = dot 6.
    Unicode: U+2800 + bitmask.

    Algorithm:
    1. Estimate dot spacing from median nearest-neighbour distance.
    2. Fine-cluster x and y with gap = 0.5 × dot_spacing to separate
       individual dot columns/rows.
    3. Group consecutive x-clusters into Braille cells (within-cell gap
       ≈ dot_spacing; inter-cell gap > 1.4 × dot_spacing).
    4. Group consecutive y-clusters into Braille text lines (same threshold).
    5. For each (line × cell) combination, build the 6-bit bitmask.
    """
    if len(dots) < min_dots:
        return "", 0.0

    xs = np.array([d[0] for d in dots], dtype=float)
    ys = np.array([d[1] for d in dots], dtype=float)
    confs = np.array([d[2] for d in dots], dtype=float)

    # Estimate dot spacing from median nearest-neighbour distance
    pts = np.stack([xs, ys], axis=1)
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    dot_spacing = float(np.median(dists.min(axis=1)))
    if dot_spacing < 1.0:
        dot_spacing = 20.0

    # Fine gap: separates adjacent dot rows/columns (< dot_spacing)
    fine_gap = dot_spacing * 0.5

    row_labels = _cluster_1d(ys, fine_gap)
    col_labels = _cluster_1d(xs, fine_gap)

    unique_rows = sorted(set(row_labels.tolist()))
    unique_cols = sorted(set(col_labels.tolist()))

    if not unique_rows or not unique_cols:
        return "", 0.0

    # Median positions for each cluster
    row_meds = {r: float(np.median(ys[row_labels == r])) for r in unique_rows}
    col_meds = {c: float(np.median(xs[col_labels == c])) for c in unique_cols}

    # Group row clusters into Braille text lines
    # Within a line, adjacent dot rows are ≈ dot_spacing apart
    row_lines = _group_consecutive(unique_rows, row_meds, dot_spacing * 1.4)

    # Group col clusters into Braille cells
    # Within a cell, left/right columns are ≈ dot_spacing apart
    col_cells = _group_consecutive(unique_cols, col_meds, dot_spacing * 1.4)

    # Build presence lookup: (row_cluster, col_cluster) → True
    presence: dict[tuple[int, int], bool] = {}
    for i in range(len(dots)):
        presence[(int(row_labels[i]), int(col_labels[i]))] = True

    braille_chars: list[str] = []

    for line_idx, row_line in enumerate(row_lines):
        for col_cell in col_cells:
            bitmask = 0
            for ri, rc in enumerate(row_line[:3]):   # up to 3 dot rows per cell
                for ci, cc in enumerate(col_cell[:2]):  # up to 2 dot cols per cell
                    if presence.get((rc, cc), False):
                        bit = ri + ci * 3  # bits 0–2 left col, 3–5 right col
                        bitmask |= (1 << bit)
            braille_chars.append(chr(0x2800 + bitmask))

        if line_idx < len(row_lines) - 1:
            braille_chars.append('\n')

    return "".join(braille_chars), float(confs.mean())

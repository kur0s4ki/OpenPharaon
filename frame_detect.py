"""
Method B: Detect the 9 cube quadrilaterals directly in the photo,
without any reference.

Strategy:
  1. Grayscale + CLAHE for lighting normalization.
  2. Adaptive threshold to isolate bright papyrus regions from the
     dark wooden frame and varied wall background.
  3. Find connected components, keep ones that are roughly square,
     of similar size, and reasonably solid.
  4. Cluster into a 3x3 grid by x/y centroid percentile binning.
  5. Return 9 quads in row-major order (r0c0, r0c1, r0c2, r1c0, ...).

Returns None if the detector can't confidently find 9 cubes.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Candidate:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def quad(self) -> np.ndarray:
        return np.array(
            [[self.x, self.y], [self.x + self.w, self.y],
             [self.x + self.w, self.y + self.h], [self.x, self.y + self.h]],
            dtype=np.float32,
        )


def _filter_candidates(contours, img_area: int) -> list[Candidate]:
    out: list[Candidate] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bbox_area = w * h
        if bbox_area < img_area * 0.004 or bbox_area > img_area * 0.10:
            continue
        aspect = w / h if h else 0
        if not (0.55 <= aspect <= 1.8):
            continue
        fill = cv2.contourArea(c) / bbox_area if bbox_area else 0
        if fill < 0.55:
            continue
        out.append(Candidate(x, y, w, h))
    return out


def _arrange_into_3x3(cands: list[Candidate]) -> list[Candidate] | None:
    """Pick exactly 9 candidates forming a 3x3 grid by centroid clustering."""
    if len(cands) < 9:
        return None

    # Trim to the largest ~20 candidates so outliers don't skew percentiles.
    cands_sorted = sorted(cands, key=lambda c: -c.area)[:20]
    cxs = np.array([c.cx for c in cands_sorted])
    cys = np.array([c.cy for c in cands_sorted])

    col_thr = np.percentile(cxs, [33.34, 66.67])
    row_thr = np.percentile(cys, [33.34, 66.67])

    grid: dict[tuple[int, int], Candidate] = {}
    for cand in cands_sorted:
        col = 0 if cand.cx < col_thr[0] else (1 if cand.cx < col_thr[1] else 2)
        row = 0 if cand.cy < row_thr[0] else (1 if cand.cy < row_thr[1] else 2)
        key = (row, col)
        # If multiple candidates fall in the same cell, keep the largest.
        if key not in grid or cand.area > grid[key].area:
            grid[key] = cand

    if len(grid) != 9:
        return None
    return [grid[(r, c)] for r in range(3) for c in range(3)]


def _sanity_check(arranged: list[Candidate]) -> bool:
    """Reject grids where sizes are wildly inconsistent (likely false positives)."""
    areas = np.array([c.area for c in arranged], dtype=np.float64)
    if areas.min() <= 0:
        return False
    spread = areas.max() / areas.min()
    return spread <= 3.0  # cubes should be roughly the same physical size


def _top_k_peaks(profile: np.ndarray, k: int, min_separation: int) -> list[tuple[int, float]]:
    """Find up to k strongest peaks in a 1-D signal, each separated by at least
    min_separation. Returns (position, strength) pairs sorted by position."""
    if profile.size == 0:
        return []
    # Smooth a bit so a thick line counts as one peak
    kernel = max(3, min_separation // 5)
    if kernel % 2 == 0:
        kernel += 1
    smoothed = cv2.GaussianBlur(profile.astype(np.float32).reshape(-1, 1), (1, kernel), 0).flatten()

    work = smoothed.copy()
    peaks: list[tuple[int, float]] = []
    for _ in range(k):
        idx = int(np.argmax(work))
        strength = float(work[idx])
        if strength <= 0:
            break
        peaks.append((idx, strength))
        lo = max(0, idx - min_separation)
        hi = min(work.size, idx + min_separation + 1)
        work[lo:hi] = 0
    return sorted(peaks, key=lambda t: t[0])


def _best_equispaced_4(peaks: list[tuple[int, float]], image_dim: int) -> list[int] | None:
    """From a pool of candidate peaks, pick 4 that form the puzzle grid.

    Score combines:
      - strength_frac: total peak strength relative to all candidates
      - equispacing: exp(-gap_cv * 4)
      - size_match: Gaussian centred on gap = 0.20 * image_dim (cubes are
        typically 18-22% of image height/width). This is the critical term
        that rejects tightly-packed false grids on walls and over-wide
        grids that span the entire image including background structure.
    """
    from itertools import combinations
    if len(peaks) < 4:
        return None

    best_score = -np.inf
    best_combo: list[int] | None = None
    total_strength = sum(s for _, s in peaks) + 1e-6
    ideal_gap_ratio = 0.20  # cube takes ~20% of image dim
    sigma = 0.07            # gaussian width — accepts roughly 13-27% range

    for combo in combinations(peaks, 4):
        positions = sorted(p for p, _ in combo)
        strengths = [s for _, s in combo]
        gaps = np.diff(positions)
        if np.any(gaps <= 0):
            continue

        gap_mean = float(np.mean(gaps))
        gap_cv = float(np.std(gaps) / max(gap_mean, 1e-6))
        strength_frac = sum(strengths) / total_strength

        ratio = gap_mean / max(image_dim, 1)
        size_match = float(np.exp(-((ratio - ideal_gap_ratio) / sigma) ** 2))

        score = strength_frac * float(np.exp(-gap_cv * 4)) * size_match
        if score > best_score:
            best_score = score
            best_combo = positions

    return best_combo


def _joint_best_grid(
    col_pool: list[tuple[int, float]],
    row_pool: list[tuple[int, float]],
    w: int, h: int,
    edge_margin_frac: float = 0.04,
) -> tuple[list[int], list[int]] | None:
    """Pick (cols, rows) jointly using a cube-squareness prior.

    Constraints baked into the score:
      - cells must be in a plausible cube size range (10-30% of image dim)
      - cube width ≈ cube height (Egyptian cubes are square)
      - peaks should not sit on the image boundary (those are border artefacts)
      - gaps within each axis should be uniform
    """
    from itertools import combinations
    if len(col_pool) < 4 or len(row_pool) < 4:
        return None

    # Drop peaks too close to the image edges (image-boundary edge artefacts)
    cm = int(w * edge_margin_frac)
    rm = int(h * edge_margin_frac)
    col_pool_f = [(p, s) for (p, s) in col_pool if cm <= p <= w - cm]
    row_pool_f = [(p, s) for (p, s) in row_pool if rm <= p <= h - rm]
    if len(col_pool_f) < 4 or len(row_pool_f) < 4:
        return None

    def axis_score(combo: tuple[tuple[int, float], ...]) -> tuple[float, list[int], float]:
        """Return (strength_frac × uniformity, sorted positions, mean_gap)."""
        positions = sorted(p for p, _ in combo)
        gaps = np.diff(positions)
        if np.any(gaps <= 0):
            return -1.0, positions, 0.0
        gap_mean = float(np.mean(gaps))
        gap_cv = float(np.std(gaps) / max(gap_mean, 1e-6))
        strength = sum(s for _, s in combo)
        uniformity = float(np.exp(-gap_cv * 4))
        return strength * uniformity, positions, gap_mean

    # Pre-score axis candidates so we only iterate top scoring quartets.
    col_combos = []
    for combo in combinations(col_pool_f, 4):
        sc, pos, gm = axis_score(combo)
        if sc > 0:
            col_combos.append((sc, pos, gm))
    row_combos = []
    for combo in combinations(row_pool_f, 4):
        sc, pos, gm = axis_score(combo)
        if sc > 0:
            row_combos.append((sc, pos, gm))

    col_combos.sort(key=lambda t: -t[0])
    row_combos.sort(key=lambda t: -t[0])
    col_combos = col_combos[:60]
    row_combos = row_combos[:60]
    if not col_combos or not row_combos:
        return None

    col_strength_total = sum(s for _, s in col_pool_f) + 1e-6
    row_strength_total = sum(s for _, s in row_pool_f) + 1e-6

    best_score = -np.inf
    best: tuple[list[int], list[int]] | None = None
    ideal_ratio = 0.20  # each cube ≈ 20% of image dimension
    sigma = 0.07

    for c_sc, c_pos, c_gm in col_combos:
        c_size = float(np.exp(-((c_gm / w - ideal_ratio) / sigma) ** 2))
        c_norm = c_sc / col_strength_total
        for r_sc, r_pos, r_gm in row_combos:
            r_size = float(np.exp(-((r_gm / h - ideal_ratio) / sigma) ** 2))
            r_norm = r_sc / row_strength_total

            # Cube-squareness: how close col-gap is to row-gap (in pixels).
            squareness = float(np.exp(-((c_gm - r_gm) / max(c_gm, r_gm, 1)) ** 2 * 8))

            score = c_norm * r_norm * c_size * r_size * squareness
            if score > best_score:
                best_score = score
                best = (c_pos, r_pos)

    return best


def detect_cube_quads(photo: np.ndarray) -> tuple[list[np.ndarray] | None, dict]:
    """
    Find 9 cube quadrilaterals in the photo via projection-profile peak
    detection + joint grid optimization with cube-squareness constraint.
    """
    info: dict = {"method": "frame_detect"}
    h, w = photo.shape[:2]

    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    sob_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sob_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    vert_profile = sob_x.sum(axis=0)
    horiz_profile = sob_y.sum(axis=1)

    min_sep_v = max(20, w // 12)
    min_sep_h = max(20, h // 12)

    col_pool = _top_k_peaks(vert_profile, 12, min_sep_v)
    row_pool = _top_k_peaks(horiz_profile, 12, min_sep_h)
    info["col_candidates"] = len(col_pool)
    info["row_candidates"] = len(row_pool)

    grid = _joint_best_grid(col_pool, row_pool, w, h)
    if grid is None:
        info["fail_reason"] = "could not find a self-consistent cube grid"
        return None, info

    cols, rows = grid
    info["col_xs"] = cols
    info["row_ys"] = rows

    quads: list[np.ndarray] = []
    for r in range(3):
        for c in range(3):
            x1, x2 = cols[c], cols[c + 1]
            y1, y2 = rows[r], rows[r + 1]
            quads.append(np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float32,
            ))

    areas = np.array([(q[2, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) for q in quads])
    if areas.min() <= 0 or areas.max() / areas.min() > 3.0:
        info["fail_reason"] = "detected grid cells have inconsistent sizes"
        return None, info

    info["ok"] = True
    return quads, info

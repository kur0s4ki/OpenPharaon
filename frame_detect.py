"""
Method B: Detect the 9 cube quadrilaterals directly in the photo using
**local variance** as the primary signal.

Insight from probing the 6 puzzle photos:
  - Cubes contain dark hieroglyphs/figures painted on a light papyrus
    background. This produces high local variance everywhere inside.
  - The wooden frame is uniformly dark — variance ~= 0.
  - Walls (stone texture) have moderate variance.
  - Painted figures on side walls and the tiger emblem also have high
    variance, but they are isolated — they don't form a 3x3 grid.

Pipeline:
  1. Compute local variance map (sliding-window E[X^2] - E[X]^2).
  2. Otsu-threshold to extract high-variance regions.
  3. Light morphology to clean up and connect cube interiors.
  4. Find connected components, filter for "cube-shaped" (size + aspect).
  5. Among all candidates, find the subset of 9 that best forms a 3x3 grid
     (uniform spacing + similar sizes). The grid constraint rejects
     isolated painted figures.
"""

from __future__ import annotations

import cv2
import numpy as np


def _local_variance(gray: np.ndarray, window: int = 15) -> np.ndarray:
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (window, window))
    sqr_mean = cv2.blur(g * g, (window, window))
    var = sqr_mean - mean * mean
    return np.clip(var, 0, var.max())


def _grid_uniformity_score(boxes: list[tuple[int, int, int, int]]) -> tuple[float, list[tuple[int, int, int, int]] | None]:
    """Pick the best 9-box subset that looks like a 3x3 grid.

    Returns (score in [0,1], 9 boxes in row-major order) or (0, None).
    """
    if len(boxes) < 9:
        return 0.0, None

    # Sort by area descending and consider the top 18 candidates (helps when
    # painted figures yield extra blobs).
    boxes_sorted = sorted(boxes, key=lambda b: -b[2] * b[3])[:18]
    if len(boxes_sorted) < 9:
        return 0.0, None

    # Centroids
    cxs = np.array([b[0] + b[2] / 2 for b in boxes_sorted])
    cys = np.array([b[1] + b[3] / 2 for b in boxes_sorted])

    best_score = 0.0
    best_arrangement: list[tuple[int, int, int, int]] | None = None

    # Try clustering using each candidate's centroid as a row/col anchor.
    # In practice the largest 9-15 candidates are mostly cubes, so percentile
    # binning on the largest 9-15 yields a clean grid.
    for try_n in (9, 12, 15, 18):
        if try_n > len(boxes_sorted):
            continue
        subset = boxes_sorted[:try_n]
        sxs = cxs[:try_n]
        sys_ = cys[:try_n]
        col_thr = np.percentile(sxs, [33.34, 66.67])
        row_thr = np.percentile(sys_, [33.34, 66.67])

        grid: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        for b, cx, cy in zip(subset, sxs, sys_):
            c = 0 if cx < col_thr[0] else (1 if cx < col_thr[1] else 2)
            r = 0 if cy < row_thr[0] else (1 if cy < row_thr[1] else 2)
            # Keep the largest box per cell if there's a tie
            if (r, c) not in grid or b[2] * b[3] > grid[(r, c)][2] * grid[(r, c)][3]:
                grid[(r, c)] = b
        if len(grid) != 9:
            continue

        # Score the arrangement
        arranged = [grid[(r, c)] for r in range(3) for c in range(3)]
        sizes = np.array([np.sqrt(b[2] * b[3]) for b in arranged])
        size_cv = float(np.std(sizes) / max(np.mean(sizes), 1e-6))
        size_score = float(np.exp(-size_cv * 4))

        # Column-x and row-y uniformity
        col_xs = np.array([np.mean([arranged[r * 3 + c][0] + arranged[r * 3 + c][2] / 2 for r in range(3)]) for c in range(3)])
        row_ys = np.array([np.mean([arranged[r * 3 + c][1] + arranged[r * 3 + c][3] / 2 for c in range(3)]) for r in range(3)])
        col_gaps = np.diff(col_xs)
        row_gaps = np.diff(row_ys)
        col_gap_cv = float(np.std(col_gaps) / max(np.mean(col_gaps), 1e-6))
        row_gap_cv = float(np.std(row_gaps) / max(np.mean(row_gaps), 1e-6))
        gap_score = float(np.exp(-(col_gap_cv + row_gap_cv) * 3))

        # Aspect ratio (cubes are roughly square)
        aspects = np.array([b[2] / max(b[3], 1) for b in arranged])
        aspect_score = float(np.exp(-np.mean(np.abs(aspects - 1.0)) * 3))

        score = size_score * gap_score * aspect_score
        if score > best_score:
            best_score = score
            best_arrangement = arranged

    return best_score, best_arrangement


def detect_cube_quads(photo: np.ndarray) -> tuple[list[np.ndarray] | None, dict]:
    """Find the 9 cube quadrilaterals using:
      - local mean (cube papyrus background is bright)
      - local variance (cube has high-contrast hieroglyphs)
    intersected. Wood pixels fail the brightness test; uniform walls fail
    the variance test; painted figures are isolated and rejected by the
    3x3 grid constraint.
    """
    info: dict = {"method": "frame_detect_variance"}
    h, w = photo.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)

    # Local mean + variance in a single pass
    win = max(7, min(h, w) // 50)
    if win % 2 == 0:
        win += 1
    local_mean = cv2.blur(g, (win, win))
    local_sqr_mean = cv2.blur(g * g, (win, win))
    local_var = np.clip(local_sqr_mean - local_mean * local_mean, 0, None)
    var_norm = (local_var / max(local_var.max(), 1) * 255).astype(np.uint8)

    # Otsu the variance to split high vs low variance regions
    _, var_mask = cv2.threshold(var_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    info["var_otsu_thr"] = int(_)

    # Brightness mask: cube background is bright; wood is dark.
    # Use a fixed-ish floor (papyrus mean is V≈113-158) but with safety margin.
    # Use Otsu on the gray to auto-pick a separator.
    bright_thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # local_mean > bright_thr means "neighbourhood is light overall"
    bright_mask = (local_mean > bright_thr).astype(np.uint8) * 255
    info["bright_otsu_thr"] = int(bright_thr)

    # Combine: cube pixels are both bright (light papyrus background) AND
    # high-variance (have hieroglyph content)
    mask = cv2.bitwise_and(var_mask, bright_mask)

    # Tiny close to bridge hieroglyph gaps but NOT divider gaps.
    # Wooden dividers are typically 8-15 px; hieroglyph gaps are 1-3 px.
    close_k = max(3, min(h, w) // 250)
    if close_k % 2 == 0:
        close_k += 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))

    # Erode slightly to guarantee adjacent cubes don't connect through any
    # remaining thin bridges
    erode_k = max(3, close_k)
    mask = cv2.erode(mask, np.ones((erode_k, erode_k), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours: keep cube-sized, square-ish, reasonably filled blobs
    boxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        box_area = ww * hh
        if box_area < img_area * 0.005 or box_area > img_area * 0.08:
            continue
        aspect = ww / max(hh, 1)
        if not (0.5 <= aspect <= 2.0):
            continue
        fill = cv2.contourArea(c) / box_area
        if fill < 0.30:
            continue
        boxes.append((x, y, ww, hh))

    info["box_candidates"] = len(boxes)

    score, arranged = _grid_uniformity_score(boxes)
    info["grid_score"] = round(score, 3)

    if arranged is None or score < 0.15:
        info["fail_reason"] = (
            f"could not arrange {len(boxes)} candidates into a 3x3 grid "
            f"(best score {score:.2f})"
        )
        return None, info

    quads = [
        np.array([[x, y], [x + ww, y], [x + ww, y + hh], [x, y + hh]], dtype=np.float32)
        for (x, y, ww, hh) in arranged
    ]

    info["ok"] = True
    return quads, info

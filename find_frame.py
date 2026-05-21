"""
Auto-detect the 3x3 puzzle frame in a photo.

Approach:
  1. Grayscale + adaptive threshold to isolate the 9 light (papyrus) cube faces
     from the dark wooden dividers and surrounding wall.
  2. Find contours, keep approximately-square ones above a size floor.
  3. Cluster into a 3x3 grid using k-means on x and y centroids.
  4. Return the outer bounding box of the 9 detected cube faces.

Fallback: if fewer than 9 squares are found, return a bounding box of the
top-9 best candidates.

CLI:
  python find_frame.py <photo.jpg>           # prints detected frame + saves debug
  python find_frame.py <photo.jpg> --save    # writes JSON snippet to stdout
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def detect_frame(img: np.ndarray, debug_path: Path | None = None) -> dict | None:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # The papyrus cube faces are notably lighter than wood/wall.
    # Adaptive threshold isolates them robustly across lighting changes.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=51, C=-10,
    )

    # Close small gaps inside each cube face.
    k = max(3, min(h, w) // 200)
    binr = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    contours, _ = cv2.findContours(binr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int, float]] = []
    img_area = h * w
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < img_area * 0.005 or area > img_area * 0.12:
            continue
        aspect = cw / ch if ch else 0
        if aspect < 0.6 or aspect > 1.6:
            continue
        # Filled-ness: how much of the bounding box is white pixels.
        fill = cv2.contourArea(c) / area if area else 0
        if fill < 0.5:
            continue
        candidates.append((x, y, cw, ch, area))

    if not candidates:
        return None

    # Sort by area descending and take the top ~9-15 candidates.
    candidates.sort(key=lambda r: -r[4])
    top = candidates[:15]

    # Try to find 9 candidates forming a 3x3 grid by clustering centroids.
    cxs = np.array([x + w / 2 for x, y, w, h, _ in top], dtype=np.float32)
    cys = np.array([y + h / 2 for x, y, w, h, _ in top], dtype=np.float32)

    if len(top) >= 9:
        # Use simple percentile-based 3-row / 3-col clustering.
        from numpy import percentile
        col_thresholds = percentile(cxs, [33.33, 66.66])
        row_thresholds = percentile(cys, [33.33, 66.66])
        grid: dict[tuple[int, int], list[int]] = {}
        for i, (cx, cy) in enumerate(zip(cxs, cys)):
            col = 0 if cx < col_thresholds[0] else (1 if cx < col_thresholds[1] else 2)
            row = 0 if cy < row_thresholds[0] else (1 if cy < row_thresholds[1] else 2)
            grid.setdefault((row, col), []).append(i)
        chosen: list[int] = []
        for cell, idxs in grid.items():
            # Pick the largest candidate in each cell.
            idxs.sort(key=lambda i: -top[i][4])
            chosen.append(idxs[0])
        if len({(top[i][0], top[i][1]) for i in chosen}) >= 6:
            selected = [top[i] for i in chosen]
        else:
            selected = top[:9]
    else:
        selected = top

    xs = [r[0] for r in selected] + [r[0] + r[2] for r in selected]
    ys = [r[1] for r in selected] + [r[1] + r[3] for r in selected]
    fx1, fx2 = min(xs), max(xs)
    fy1, fy2 = min(ys), max(ys)

    result = {
        "x1": int(fx1), "y1": int(fy1), "x2": int(fx2), "y2": int(fy2),
        "n_candidates": len(selected),
    }

    if debug_path is not None:
        dbg = img.copy()
        for (x, y, w_, h_, _) in selected:
            cv2.rectangle(dbg, (x, y), (x + w_, y + h_), (0, 200, 0), 2)
        cv2.rectangle(dbg, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
        cv2.imwrite(str(debug_path), dbg)

    return result


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python find_frame.py <photo> [--save]", file=sys.stderr)
        return 2
    p = Path(sys.argv[1])
    img = cv2.imread(str(p))
    if img is None:
        print(f"ERROR: could not read {p}", file=sys.stderr)
        return 2
    dbg = p.parent / f"debug_frame_{p.stem}.png"
    r = detect_frame(img, dbg)
    if r is None:
        print("ERROR: no frame found", file=sys.stderr)
        return 1
    if "--save" in sys.argv:
        print(json.dumps({"x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"]}))
    else:
        print(f"{p.name}  frame=({r['x1']},{r['y1']},{r['x2']},{r['y2']})  n={r['n_candidates']}  debug={dbg.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Diagnostic: 9x9 confusion matrix of photo slots vs all reference cells.

For each photo slot i, scores against all 9 reference cells. The CORRECT
cell is the diagonal (j == i). If the correct cell wins (highest score in
its row), we have a working detector — even if absolute scores are low.

Outputs:
  - Console: confusion matrix per metric + diagonal-wins summary
  - confusion.csv: full matrix
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from pharaon_cv import (
    compute_slot_rects,
    crop,
    hist_correlation,
    load_config,
    orb_good_matches,
    orb_ransac_inliers,
    phash_distance,
)


def split_reference(reference: np.ndarray, rc: dict) -> list[np.ndarray]:
    rx2 = rc["x2"] if rc["x2"] is not None else reference.shape[1]
    ry2 = rc["y2"] if rc["y2"] is not None else reference.shape[0]
    ref_square = reference[rc["y1"]:ry2, rc["x1"]:rx2].copy()
    h, w = ref_square.shape[:2]
    ch, cw = h / 3, w / 3
    cells = []
    for r in range(3):
        for c in range(3):
            y1, y2 = int(round(r * ch)), int(round((r + 1) * ch))
            x1, x2 = int(round(c * cw)), int(round((c + 1) * cw))
            cells.append(ref_square[y1:y2, x1:x2].copy())
    return cells


def main() -> int:
    root = Path(__file__).parent
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "config.json"
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path)
    puzzle = cv2.imread(str(root / cfg["puzzle_image"]))
    reference = cv2.imread(str(root / cfg["reference_image"]))
    if puzzle is None or reference is None:
        print("ERROR: could not read images.", file=sys.stderr)
        return 2

    ref_cells = split_reference(reference, cfg["reference_crop"])
    ps = cfg["puzzle_slots"]
    slot_rects = compute_slot_rects(ps["frame"], ps["rows"], ps["cols"], ps["inner_margin_px"])
    photo_slots = [crop(puzzle, r) for r in slot_rects]

    metrics = {
        "ORB_raw": (orb_good_matches, "high"),
        "ORB_RANSAC": (orb_ransac_inliers, "high"),
        "pHash": (phash_distance, "low"),
        "hist": (hist_correlation, "high"),
    }

    rows = []
    for name, (fn, direction) in metrics.items():
        print(f"\n=== {name} ({'higher=better' if direction == 'high' else 'lower=better'}) ===")
        header = "      " + "  ".join(f"ref{j}" for j in range(9))
        print(header)
        wins_diagonal = 0
        for i, slot in enumerate(photo_slots):
            row_scores = [fn(slot, ref_cells[j]) for j in range(9)]
            if direction == "high":
                best_j = int(np.argmax(row_scores))
            else:
                best_j = int(np.argmin(row_scores))
            wins_diagonal += int(best_j == i)
            cells_str = "  ".join(
                f"[{v:>4.2f}]" if isinstance(v, float) else f"[{v:>4}]"
                for v in row_scores
            )
            marker = "*" if best_j == i else " "
            best_str = f"  -> best=ref{best_j}{marker}"
            print(f"slot{i} {cells_str}{best_str}")
            for j, v in enumerate(row_scores):
                rows.append({"metric": name, "slot": i, "ref": j, "score": v, "is_diagonal": int(i == j)})
        print(f"diagonal wins: {wins_diagonal}/9")

    with (root / "confusion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "slot", "ref", "score", "is_diagonal"])
        w.writeheader()
        w.writerows(rows)
    print("\nWrote: confusion.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

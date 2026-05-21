"""
6-puzzle batch validation.

For each puzzle:
  - Load its reference image, split into 3x3 cells
  - For each of its solved prod photos:
      - Run the per-slot ORB-RANSAC matcher
      - All 9 slots must MATCH (verdict SOLVED)
  - Save a debug overlay per photo

Output: per-photo summary, overall pass rate, debug overlays.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from pharaon_cv import (
    compute_slot_rects,
    crop,
    draw_overlay,
    orb_ransac_inliers,
    verdict_for,
)


def split_reference(reference: np.ndarray, rc: dict) -> list[np.ndarray]:
    rx2 = rc["x2"] if rc["x2"] is not None else reference.shape[1]
    ry2 = rc["y2"] if rc["y2"] is not None else reference.shape[0]
    sq = reference[rc["y1"]:ry2, rc["x1"]:rx2].copy()
    h, w = sq.shape[:2]
    ch, cw = h / 3, w / 3
    cells = []
    for r in range(3):
        for c in range(3):
            y1, y2 = int(round(r * ch)), int(round((r + 1) * ch))
            x1, x2 = int(round(c * cw)), int(round((c + 1) * cw))
            cells.append(sq[y1:y2, x1:x2].copy())
    return cells


def run_photo(photo_cfg: dict, ref_cells: list[np.ndarray], thr: dict, root: Path) -> dict:
    photo_path = root / photo_cfg["photo"]
    img = cv2.imread(str(photo_path))
    if img is None:
        return {"name": photo_cfg["name"], "error": f"could not read {photo_path}"}

    rects = compute_slot_rects(photo_cfg["frame"], 3, 3, photo_cfg.get("inner_margin_px", 16))
    slots = [crop(img, r) for r in rects]

    per_slot = []
    labels = []
    for i, p in enumerate(slots):
        per_ref = [orb_ransac_inliers(p, r) for r in ref_cells]
        best_idx = int(np.argmax(per_ref))
        v = verdict_for(per_ref[i], per_ref[best_idx], best_idx, i, thr)
        per_slot.append({
            "slot": i,
            "expected_inliers": per_ref[i],
            "best_inliers": per_ref[best_idx],
            "best_idx": best_idx,
            "verdict": v,
        })
        if v == "WRONG_FACE":
            bm_r, bm_c = divmod(best_idx, 3)
            labels.append(f"->r{bm_r}c{bm_c} WRONG_FACE")
        else:
            labels.append(v)

    n_match = sum(1 for s in per_slot if s["verdict"] == "MATCH")
    overlay = draw_overlay(img, rects, labels)
    overlay_path = root / f"debug_puzzles_{photo_cfg['name']}.png"
    cv2.imwrite(str(overlay_path), overlay)

    return {
        "name": photo_cfg["name"],
        "photo": str(photo_cfg["photo"]),
        "match": n_match,
        "wrong_face": sum(1 for s in per_slot if s["verdict"] == "WRONG_FACE"),
        "empty": sum(1 for s in per_slot if s["verdict"] == "EMPTY"),
        "per_slot": per_slot,
        "pass": n_match == 9,
    }


def main() -> int:
    root = Path(__file__).parent
    cfg = json.loads((root / "puzzles_batch.json").read_text(encoding="utf-8"))
    thr = cfg["thresholds"]
    rc = cfg["reference_crop"]

    all_results = []
    for puzzle in cfg["puzzles"]:
        print(f"\n=== {puzzle['name']} (ref: {puzzle['ref']}) ===")
        ref = cv2.imread(str(root / puzzle["ref"]))
        if ref is None:
            print(f"  ERROR: cannot read reference {puzzle['ref']}")
            continue
        ref_cells = split_reference(ref, rc)
        for photo_cfg in puzzle["photos"]:
            r = run_photo(photo_cfg, ref_cells, thr, root)
            r["puzzle"] = puzzle["name"]
            if "error" in r:
                print(f"  [{r['name']}] ERROR: {r['error']}")
                all_results.append(r)
                continue
            tag = "PASS" if r["pass"] else "FAIL"
            print(f"  [{r['name']}] {tag}  MATCH={r['match']}/9  WRONG_FACE={r['wrong_face']}  EMPTY={r['empty']}")
            for s in r["per_slot"]:
                row, col = divmod(s["slot"], 3)
                bm_r, bm_c = divmod(s["best_idx"], 3)
                print(f"      r{row}c{col}  exp={s['expected_inliers']:>4}  best={s['best_inliers']:>4} (r{bm_r}c{bm_c})  {s['verdict']}")
            all_results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for r in all_results if r.get("pass"))
    for r in all_results:
        tag = "PASS" if r.get("pass") else "FAIL"
        extra = "" if "error" in r else f"  MATCH={r.get('match',0)}/9"
        print(f"  [{tag}] {r.get('puzzle', '?')}/{r['name']}{extra}")
    print(f"\nOVERALL: {n_pass}/{len(all_results)} photos passed (need 100%).")
    return 0 if n_pass == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())

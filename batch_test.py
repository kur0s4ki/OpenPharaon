"""
Batch test runner: validate the matcher across multiple photos with varying
camera positions, lighting, and puzzle states.

Outputs:
  - Per-photo summary (MATCH / WRONG_FACE / EMPTY counts, verdict)
  - Per-photo debug overlay (PNG)
  - PASS/FAIL vs expected verdict
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


def run_test(test: dict, ref_cells: list[np.ndarray], thr: dict, root: Path) -> dict:
    photo_path = root / test["photo"]
    img = cv2.imread(str(photo_path))
    if img is None:
        return {"name": test["name"], "error": f"could not read {photo_path}"}

    rects = compute_slot_rects(test["frame"], 3, 3, test.get("inner_margin_px", 12))
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
    n_wrong = sum(1 for s in per_slot if s["verdict"] == "WRONG_FACE")
    n_empty = sum(1 for s in per_slot if s["verdict"] == "EMPTY")
    actual = "SOLVED" if n_match == 9 else "NOT_SOLVED"
    expected = test["expected"]

    overlay = draw_overlay(img, rects, labels)
    overlay_path = root / f"debug_batch_{test['name']}.png"
    cv2.imwrite(str(overlay_path), overlay)

    return {
        "name": test["name"],
        "expected": expected,
        "actual": actual,
        "pass": expected == actual,
        "match": n_match,
        "wrong_face": n_wrong,
        "empty": n_empty,
        "per_slot": per_slot,
    }


def main() -> int:
    root = Path(__file__).parent
    cfg = json.loads((root / "test_batch.json").read_text(encoding="utf-8"))

    reference = cv2.imread(str(root / cfg["reference_image"]))
    if reference is None:
        print("ERROR: could not load reference image.", file=sys.stderr)
        return 2
    ref_cells = split_reference(reference, cfg["reference_crop"])
    thr = cfg["thresholds"]

    results = []
    for test in cfg["tests"]:
        print(f"\n--- {test['name']} ({test['photo']}) ---")
        r = run_test(test, ref_cells, thr, root)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            results.append(r)
            continue
        print(f"  expected={r['expected']}  actual={r['actual']}  -> {'PASS' if r['pass'] else 'FAIL'}")
        print(f"  per-slot: MATCH={r['match']}  WRONG_FACE={r['wrong_face']}  EMPTY={r['empty']}")
        for s in r["per_slot"]:
            row, col = divmod(s["slot"], 3)
            bm_r, bm_c = divmod(s["best_idx"], 3)
            print(f"    r{row}c{col}  exp={s['expected_inliers']:>3}  best={s['best_inliers']:>3} (r{bm_r}c{bm_c})  {s['verdict']}")
        results.append(r)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for r in results if r.get("pass"))
    print(f"PASS: {n_pass}/{len(results)}")
    for r in results:
        status = "PASS" if r.get("pass") else "FAIL"
        if "error" in r:
            status = "ERROR"
            extra = r["error"]
        else:
            extra = f"expected={r['expected']} actual={r['actual']} (match={r['match']}, wrong={r['wrong_face']}, empty={r['empty']})"
        print(f"  [{status}] {r['name']}  {extra}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

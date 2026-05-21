"""
6-puzzle batch validation with homography-based auto-alignment.

No manual slot calibration. For each (photo, reference):
  1. Find the homography that maps reference -> photo via ORB matching.
  2. Project the 9 reference cell quads into the photo.
  3. Warp each photo quad to a square slot, match against the corresponding
     reference cell.
  4. Verdict per slot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from homography_align import find_homography, project_cells, warp_quad, draw_quads
from pharaon_cv import orb_ransac_inliers, verdict_for


def split_reference(reference: np.ndarray) -> list[np.ndarray]:
    h, w = reference.shape[:2]
    ch, cw = h / 3, w / 3
    cells = []
    for r in range(3):
        for c in range(3):
            y1, y2 = int(round(r * ch)), int(round((r + 1) * ch))
            x1, x2 = int(round(c * cw)), int(round((c + 1) * cw))
            cells.append(reference[y1:y2, x1:x2].copy())
    return cells


def run_photo(photo_path: Path, ref_path: Path, thr: dict, debug_root: Path) -> dict:
    photo = cv2.imread(str(photo_path))
    ref = cv2.imread(str(ref_path))
    if photo is None or ref is None:
        return {"name": photo_path.stem, "error": "could not load images"}

    H, h_inliers, h_good = find_homography(photo, ref)
    if H is None:
        return {"name": photo_path.stem, "error": "homography failed"}

    quads = project_cells(H, ref)
    photo_slots = [warp_quad(photo, q, 256) for q in quads]
    ref_cells = split_reference(ref)

    per_slot = []
    labels = []
    for i, (p_slot, r_cell) in enumerate(zip(photo_slots, ref_cells)):
        per_ref = [orb_ransac_inliers(p_slot, rc) for rc in ref_cells]
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
            br, bc = divmod(best_idx, 3)
            labels.append(f"r{i // 3}c{i % 3} ->r{br}c{bc}")
        else:
            labels.append(f"r{i // 3}c{i % 3} {v}")

    # debug/<puzzle_name>/<photo_stem>_alignment.png
    puzzle_dir = debug_root / photo_path.parent.name.replace(" ", "_")
    puzzle_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_quads(photo, quads, labels)
    cv2.imwrite(str(puzzle_dir / f"{photo_path.stem}_alignment.png"), overlay)

    n_match = sum(1 for s in per_slot if s["verdict"] == "MATCH")
    return {
        "name": photo_path.stem,
        "puzzle": photo_path.parent.name,
        "photo": str(photo_path),
        "h_inliers": h_inliers,
        "h_good": h_good,
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
    debug_root = root / "debug"
    debug_root.mkdir(exist_ok=True)

    all_results = []
    for puzzle in cfg["puzzles"]:
        ref_path = root / puzzle["ref"]
        print(f"\n=== {puzzle['name']} ===")
        for photo_cfg in puzzle["photos"]:
            photo_path = root / photo_cfg["photo"]
            r = run_photo(photo_path, ref_path, thr, debug_root)
            r["puzzle_name"] = puzzle["name"]
            if "error" in r:
                print(f"  [{r['name']}] ERROR: {r['error']}")
                all_results.append(r)
                continue
            tag = "PASS" if r["pass"] else "FAIL"
            print(f"  [{r['name']}] {tag}  H_inliers={r['h_inliers']}  MATCH={r['match']}/9  WRONG_FACE={r['wrong_face']}  EMPTY={r['empty']}")
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
        extra = "" if "error" in r else f"  MATCH={r.get('match',0)}/9  H_inliers={r.get('h_inliers', 0)}"
        print(f"  [{tag}] {r.get('puzzle_name', '?')}/{r['name']}{extra}")
    print(f"\nOVERALL: {n_pass}/{len(all_results)} photos passed")
    return 0 if n_pass == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())

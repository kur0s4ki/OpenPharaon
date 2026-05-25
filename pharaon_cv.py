"""
Pharaon puzzle verification — markerless OpenCV prototype.

Pipeline:
  1. Load puzzle photo (camera frame) + reference papyrus image.
  2. Crop the reference papyrus square and split it into a 3x3 grid of "expected" cells.
  3. Crop the 9 cube-face slots from the puzzle photo using a configurable frame rect.
  4. For each slot pair (photo_slot, reference_cell), compute three similarity signals:
       - ORB feature matches (Lowe ratio test, inlier count)
       - Perceptual hash hamming distance (pHash)
       - HSV histogram correlation
  5. Combine into a per-slot verdict at configurable thresholds.
  6. Emit debug images and a CSV/table of scores.

Runtime target: Raspberry Pi (CPU only, no ML dependencies beyond OpenCV).
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


# ---------- config / data ----------

@dataclass
class SlotScores:
    row: int
    col: int
    expected_inliers: int       # score against the cell that SHOULD be here
    best_inliers: int           # highest score across all 9 reference cells
    best_match_index: int       # which ref cell this slot matches best (0..8)
    verdict: str                # MATCH / WRONG_FACE / EMPTY


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- geometry ----------

def compute_slot_rects(frame: dict, rows: int, cols: int, margin: int) -> list[tuple[int, int, int, int]]:
    """Return row-major list of (x1, y1, x2, y2) for each cell."""
    fx1, fy1, fx2, fy2 = frame["x1"], frame["y1"], frame["x2"], frame["y2"]
    cell_w = (fx2 - fx1) / cols
    cell_h = (fy2 - fy1) / rows
    rects: list[tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(round(fx1 + c * cell_w + margin))
            y1 = int(round(fy1 + r * cell_h + margin))
            x2 = int(round(fx1 + (c + 1) * cell_w - margin))
            y2 = int(round(fy1 + (r + 1) * cell_h - margin))
            rects.append((x1, y1, x2, y2))
    return rects


def crop(img: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = rect
    return img[y1:y2, x1:x2].copy()


# ---------- similarity signals ----------

def _preprocess(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Normalize for matching: aspect-preserving letterbox + grayscale + CLAHE.

    Letterboxing matters when comparing patches whose source aspect ratios
    differ (e.g. rectangular reference cell vs nearly-square cube face).
    Stretching to a fixed square would distort features asymmetrically.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_t = (size - nh) // 2
    pad_b = size - nh - pad_t
    pad_l = (size - nw) // 2
    pad_r = size - nw - pad_l
    g = cv2.copyMakeBorder(g, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(g)


def orb_good_matches(a: np.ndarray, b: np.ndarray, n_features: int = 2000) -> int:
    """Count Lowe-ratio-test "good" matches between two image patches."""
    ga = _preprocess(a)
    gb = _preprocess(b)
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 2 or len(kb) < 2:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(da, db, k=2)
    good = 0
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good += 1
    return good


# Use USAC_FAST when available (OpenCV >= 4.5). Falls back to RANSAC on older
# versions. USAC_FAST runs the same problem 2-3x faster than classical RANSAC
# at comparable quality for our small per-slot point sets.
_HOMOG_METHOD = getattr(cv2, "USAC_FAST", cv2.RANSAC)


def precompute_orb(img: np.ndarray, n_features: int = 600, size: int = 256):
    """Preprocess `img` and run ORB. Returns (keypoints, descriptors).

    Designed to be cached: call once on each reference cell at startup,
    re-use across many photo-slot matchings.

    n_features default lowered to 600 — at 256x256 the extra 400 features
    that 1000 would extract are mostly redundant and slow down knn matching.
    """
    g = _preprocess(img, size=size)
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8)
    return orb.detectAndCompute(g, None)


def ransac_inliers_from_descriptors(ka, da, kb, db) -> int:
    """Count RANSAC inliers given already-computed ORB keypoints+descriptors.

    Hot path on Pi: skips the preprocess+ORB step that dominates runtime,
    so callers can amortize feature extraction across many comparisons.
    Uses USAC_FAST (modern OpenCV) for ~2-3x speedup over classical RANSAC.
    """
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(da, db, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 4:
        return 0
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, _HOMOG_METHOD, 5.0)
    if mask is None:
        return 0
    return int(mask.sum())


def orb_ransac_inliers(a: np.ndarray, b: np.ndarray, n_features: int = 600) -> int:
    """Convenience: compute features for both sides then call RANSAC matcher.
    Slow path — kept for callers that don't have a cached side."""
    ka, da = precompute_orb(a, n_features=n_features)
    kb, db = precompute_orb(b, n_features=n_features)
    return ransac_inliers_from_descriptors(ka, da, kb, db)


def phash_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Perceptual hash hamming distance (0=identical, 64=opposite)."""
    hasher = cv2.img_hash.PHash_create()
    ha = hasher.compute(a)
    hb = hasher.compute(b)
    return int(cv2.norm(ha, hb, cv2.NORM_HAMMING))


def hist_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """HSV histogram correlation in [-1, 1]; 1.0 = identical distribution."""
    ah = _hsv_hist(a)
    bh = _hsv_hist(b)
    return float(cv2.compareHist(ah, bh, cv2.HISTCMP_CORREL))


def _hsv_hist(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def ncc_multiscale(slot: np.ndarray, ref_cell: np.ndarray) -> float:
    """Best normalized cross-correlation across small scales.

    Resizes the ref cell to slightly smaller than the slot and slides it
    via cv2.matchTemplate(TM_CCOEFF_NORMED). Returns the max correlation
    score in [-1, 1]. Robust to small alignment drift.
    """
    s = _preprocess(slot, size=256)
    best = -1.0
    for scale in (0.80, 0.85, 0.90, 0.95):
        tpl_size = int(round(256 * scale))
        tpl = _preprocess(ref_cell, size=tpl_size)
        res = cv2.matchTemplate(s, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, _ = cv2.minMaxLoc(res)
        if mx > best:
            best = float(mx)
    return best


# ---------- verdict ----------

def verdict_for(
    expected_inliers: int,
    best_inliers: int,
    best_idx: int,
    slot_idx: int,
    thr: dict,
    context_strong: bool = False,
) -> str:
    """Verdict for a single slot.

    Three outcomes, in order:
      1. WRONG_FACE — clear off-cell win: best > expected + wrong_face_margin
         AND best >= wrong_face_floor. The cube is at this slot but its face
         clearly belongs in another slot. Independent of context.
      2. MATCH — the cube IS at this slot showing the right face. Requires:
          * Diagonal-best: best_idx == slot_idx AND expected >= floor, OR
          * (context_strong only) tied off-diagonal at noise floor: when the
            puzzle is overall mostly correct (6+ slots clearly match), accept
            a weak tied slot as MATCH. In a partially-scrambled puzzle this
            lenient rule is OFF, so noise-floor ties go to EMPTY rather than
            falsely declaring MATCH on a clearly wrong cube.
      3. EMPTY — neither MATCH nor WRONG_FACE.
    """
    floor = thr["orb_inliers_min"]
    wrong_face_floor = thr.get("wrong_face_min", 10)
    wrong_face_margin = thr.get("wrong_face_margin", 5)

    if (
        best_idx != slot_idx
        and best_inliers > expected_inliers + wrong_face_margin
        and best_inliers >= wrong_face_floor
    ):
        return "WRONG_FACE"

    if best_idx == slot_idx and expected_inliers >= floor:
        return "MATCH"

    if context_strong and expected_inliers == best_inliers and expected_inliers >= floor:
        return "MATCH"

    return "EMPTY"


# ---------- debug rendering ----------

def draw_overlay(img: np.ndarray, rects: list[tuple[int, int, int, int]], labels: list[str]) -> np.ndarray:
    out = img.copy()
    color_for = {"MATCH": (0, 200, 0), "WRONG_FACE": (0, 165, 255), "EMPTY": (0, 0, 200)}
    for (x1, y1, x2, y2), label in zip(rects, labels):
        verdict = label.split()[-1]
        color = color_for.get(verdict, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, label, (x1 + 6, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def montage_pairs(photo_slots: list[np.ndarray], ref_cells: list[np.ndarray], scores: list[SlotScores], cell_px: int = 220) -> np.ndarray:
    """Return a single image: 3 rows x 3 cols, each cell shows photo|ref side-by-side."""
    pair_w = cell_px * 2 + 4
    pair_h = cell_px + 60
    grid_w = pair_w * 3 + 8
    grid_h = pair_h * 3 + 8
    canvas = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)
    for i, (p, r, s) in enumerate(zip(photo_slots, ref_cells, scores)):
        row, col = s.row, s.col
        ox = col * pair_w + 4
        oy = row * pair_h + 4
        p_r = cv2.resize(p, (cell_px, cell_px))
        r_r = cv2.resize(r, (cell_px, cell_px))
        canvas[oy:oy + cell_px, ox:ox + cell_px] = p_r
        canvas[oy:oy + cell_px, ox + cell_px + 4:ox + cell_px + 4 + cell_px] = r_r
        text = f"r{row}c{col} exp={s.expected_inliers} best={s.best_inliers} {s.verdict}"
        color = (0, 220, 0) if s.verdict == "MATCH" else (0, 80, 255)
        cv2.putText(canvas, text, (ox, oy + cell_px + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


# ---------- main ----------

def main(config_path: Path) -> int:
    cfg = load_config(config_path)
    root = config_path.parent

    puzzle = cv2.imread(str(root / cfg["puzzle_image"]))
    reference = cv2.imread(str(root / cfg["reference_image"]))
    if puzzle is None or reference is None:
        print("ERROR: could not read images.", file=sys.stderr)
        return 2

    # 1. Crop reference papyrus square + split into 3x3
    rc = cfg["reference_crop"]
    rx2 = rc["x2"] if rc["x2"] is not None else reference.shape[1]
    ry2 = rc["y2"] if rc["y2"] is not None else reference.shape[0]
    ref_square = reference[rc["y1"]:ry2, rc["x1"]:rx2].copy()
    ref_h, ref_w = ref_square.shape[:2]
    cell_h, cell_w = ref_h / 3, ref_w / 3
    ref_cells: list[np.ndarray] = []
    for r in range(3):
        for c in range(3):
            y1 = int(round(r * cell_h))
            y2 = int(round((r + 1) * cell_h))
            x1 = int(round(c * cell_w))
            x2 = int(round((c + 1) * cell_w))
            ref_cells.append(ref_square[y1:y2, x1:x2].copy())

    # 2. Compute photo slot rects + crops
    ps = cfg["puzzle_slots"]
    slot_rects = compute_slot_rects(ps["frame"], ps["rows"], ps["cols"], ps["inner_margin_px"])
    photo_slots = [crop(puzzle, r) for r in slot_rects]

    # 3. Score each slot against ALL 9 reference cells (full confusion).
    #    Required to distinguish MATCH from WRONG_FACE (correct cube, wrong slot).
    thr = cfg["thresholds"]
    scores: list[SlotScores] = []
    for i, p_slot in enumerate(photo_slots):
        row, col = divmod(i, 3)
        per_ref = [orb_ransac_inliers(p_slot, r_cell) for r_cell in ref_cells]
        best_idx = int(np.argmax(per_ref))
        scores.append(SlotScores(
            row=row,
            col=col,
            expected_inliers=per_ref[i],
            best_inliers=per_ref[best_idx],
            best_match_index=best_idx,
            verdict=verdict_for(per_ref[i], per_ref[best_idx], best_idx, i, thr),
        ))

    # 4. Print results
    print(f"\n{'slot':<6}{'expected':>10}{'best':>6}  {'best=':<6}  verdict")
    print("-" * 50)
    for s in scores:
        bm_row, bm_col = divmod(s.best_match_index, 3)
        print(f"r{s.row}c{s.col}  {s.expected_inliers:>10}{s.best_inliers:>6}  r{bm_row}c{bm_col}    {s.verdict}")
    n_match = sum(1 for s in scores if s.verdict == "MATCH")
    n_wrong = sum(1 for s in scores if s.verdict == "WRONG_FACE")
    n_empty = sum(1 for s in scores if s.verdict == "EMPTY")
    print(f"\nMATCH: {n_match}/9   WRONG_FACE: {n_wrong}/9   EMPTY: {n_empty}/9")
    if n_match == 9:
        print("PUZZLE SOLVED")
    else:
        print("PUZZLE NOT SOLVED")

    # 5. Write debug outputs
    labels = []
    for s in scores:
        if s.verdict == "WRONG_FACE":
            bm_row, bm_col = divmod(s.best_match_index, 3)
            labels.append(f"r{s.row}c{s.col} ->r{bm_row}c{bm_col} WRONG_FACE")
        else:
            labels.append(f"r{s.row}c{s.col} {s.verdict}")
    overlay = draw_overlay(puzzle, slot_rects, labels)
    cv2.imwrite(str(root / "debug_overlay.png"), overlay)

    ref_overlay = reference.copy()
    cv2.rectangle(ref_overlay, (rc["x1"], rc["y1"]), (rx2, ry2), (0, 255, 0), 4)
    for k in range(1, 3):
        x = int(round(rc["x1"] + k * (rx2 - rc["x1"]) / 3))
        y = int(round(rc["y1"] + k * (ry2 - rc["y1"]) / 3))
        cv2.line(ref_overlay, (x, rc["y1"]), (x, ry2), (0, 255, 0), 2)
        cv2.line(ref_overlay, (rc["x1"], y), (rx2, y), (0, 255, 0), 2)
    cv2.imwrite(str(root / "debug_reference.png"), ref_overlay)

    cv2.imwrite(str(root / "debug_montage.png"), montage_pairs(photo_slots, ref_cells, scores))

    with (root / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row", "col", "expected_inliers", "best_inliers", "best_match_index", "verdict"])
        w.writeheader()
        for s in scores:
            w.writerow(asdict(s))

    print("\nWrote: debug_overlay.png, debug_reference.png, debug_montage.png, scores.csv")
    return 0


if __name__ == "__main__":
    cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "config.json"
    sys.exit(main(cfg))

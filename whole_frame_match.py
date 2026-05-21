"""
Whole-frame matching: robust SOLVED/NOT_SOLVED detection without precise slot calibration.

Idea:
  - Slot-level matching needs precise per-slot calibration. When the frame box
    is off by 50 pixels, ORB sees mostly background and we get noise.
  - For binary solved/not-solved, we don't need slot-level precision. We can
    match the entire cropped puzzle region against the entire reference image.
  - If the puzzle is solved, the whole region matches the reference with
    hundreds of consistent feature inliers under one homography.
  - If scrambled, features don't fit a single transformation -> low inliers.

This complements the slot-level matcher: use whole-frame for the binary verdict,
fall back to per-slot only when the whole-frame says SOLVED-and-we-want-hints,
or for the WRONG_FACE diagnostic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from pharaon_cv import _preprocess


def whole_frame_inliers(photo_crop: np.ndarray, reference: np.ndarray) -> tuple[int, int]:
    """ORB + RANSAC homography on the whole frame vs the whole reference.

    Returns (inlier_count, good_match_count). Inliers under a consistent
    homography are the strong signal; good matches alone include outliers.
    """
    a = _preprocess(photo_crop, size=512)  # larger size for whole-frame detail
    b = _preprocess(reference, size=512)

    orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return 0, 0

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
        return 0, len(good)

    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return 0, len(good)
    return int(mask.sum()), len(good)


def main() -> int:
    root = Path(__file__).parent
    cfg = json.loads((root / "test_batch.json").read_text(encoding="utf-8"))
    reference = cv2.imread(str(root / cfg["reference_image"]))
    if reference is None:
        print("ERROR: cannot load reference", file=sys.stderr)
        return 2

    print(f"{'name':<32}{'inliers':>10}{'good':>8}  verdict  expected")
    print("-" * 72)
    n_pass = 0
    threshold = 30  # whole-frame uses ~10x more features, so threshold scales up

    for test in cfg["tests"]:
        photo = cv2.imread(str(root / test["photo"]))
        if photo is None:
            print(f"{test['name']:<32}  ERROR: cannot load")
            continue
        f = test["frame"]
        crop = photo[f["y1"]:f["y2"], f["x1"]:f["x2"]].copy()
        inliers, good = whole_frame_inliers(crop, reference)
        verdict = "SOLVED" if inliers >= threshold else "NOT_SOLVED"
        ok = verdict == test["expected"]
        n_pass += int(ok)
        flag = "PASS" if ok else "FAIL"
        print(f"{test['name']:<32}{inliers:>10}{good:>8}  {verdict:<12} expected={test['expected']}  [{flag}]")

    print(f"\nthreshold = {threshold} inliers")
    print(f"PASS: {n_pass}/{len(cfg['tests'])}")
    return 0 if n_pass == len(cfg["tests"]) else 1


if __name__ == "__main__":
    sys.exit(main())

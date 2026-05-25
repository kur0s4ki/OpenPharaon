"""
Core ORB matching primitives used by the OpenPharaon verifier.

Three small functions, in order of how they fit the pipeline:

  precompute_orb(img)
      Preprocess an image (resize-and-letterbox to 256x256, grayscale,
      CLAHE) and run ORB. Returns (keypoints, descriptors). Designed to
      be called ONCE per reference cell at startup and cached, so that
      per-check matching only needs to compute features for the photo
      side.

  ransac_inliers_from_descriptors(ka, da, kb, db)
      Brute-force ORB matching with Lowe ratio test + RANSAC (USAC_FAST
      when available) homography. Returns the inlier count. This is the
      per-cell similarity score the verifier uses.

  verdict_for(expected, best, best_idx, slot_idx, thr, context_strong)
      Turns a slot's (own-cell score, best-cell-across-9 score) into one
      of MATCH / WRONG_FACE / EMPTY according to a few thresholds and a
      "context strong" flag (set by the caller when most of the grid has
      strong matches — see app.py).

Runtime target: Raspberry Pi (CPU only, no ML).
"""

from __future__ import annotations

import cv2
import numpy as np


# Use USAC_FAST when available (OpenCV >= 4.5); 2-3x faster than RANSAC
# on small point sets at comparable robustness.
_HOMOG_METHOD = getattr(cv2, "USAC_FAST", cv2.RANSAC)


def _preprocess(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Resize-and-letterbox to `size`x`size`, grayscale, CLAHE-normalise."""
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


def precompute_orb(img: np.ndarray, n_features: int = 600, size: int = 256):
    """Preprocess `img` and run ORB. Returns (keypoints, descriptors)."""
    g = _preprocess(img, size=size)
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8)
    return orb.detectAndCompute(g, None)


def ransac_inliers_from_descriptors(ka, da, kb, db) -> int:
    """Brute-force ORB match (Lowe ratio 0.75) + RANSAC homography.
    Returns the number of geometrically-consistent inliers."""
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
    _H, mask = cv2.findHomography(src, dst, _HOMOG_METHOD, 5.0)
    if mask is None:
        return 0
    return int(mask.sum())


def verdict_for(
    expected_inliers: int,
    best_inliers: int,
    best_idx: int,
    slot_idx: int,
    thr: dict,
    context_strong: bool = False,
) -> str:
    """Verdict for a single slot.

    Rules, in order:
      1. WRONG_FACE — a different cell wins by a confident margin
         (best > expected + wrong_face_margin AND best >= wrong_face_min).
         Catches "this cube belongs in slot X" cases with real signal.
      2. MATCH — diagonal is the winner OR tied (expected == best),
         and expected >= floor. When context_strong is set, ties are
         accepted even on off-diagonal (handles weak-signal solved
         puzzles where most cubes are clearly correct).
      3. EMPTY — anything else (the cube is showing a face the matcher
         doesn't recognise against this puzzle).
    """
    floor = thr["orb_inliers_min"]
    wf_floor = thr.get("wrong_face_min", 10)
    wf_margin = thr.get("wrong_face_margin", 5)

    diagonal_is_winner_or_tied = (
        (best_idx == slot_idx)
        or (context_strong and expected_inliers == best_inliers)
    )
    if expected_inliers >= floor and diagonal_is_winner_or_tied:
        return "MATCH"
    if (best_inliers > expected_inliers + wf_margin
            and best_inliers >= wf_floor
            and best_idx != slot_idx):
        return "WRONG_FACE"
    return "EMPTY"

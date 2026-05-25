"""
Homography-based auto-alignment.

Given a photo of a solved puzzle and the reference image, find the
homography that maps reference coordinates -> photo coordinates by
matching ORB features across the two images. Then project the 9
reference cell quadrilaterals into the photo to get exact slot
positions in the photo, no manual calibration required.

This handles arbitrary camera angle, distance, lens distortion, and
even moderate rotation — as long as enough of the reference is visible
in the photo.

Usage:
  python homography_align.py <photo.png> <reference.png>
    -> writes debug_homography_<name>.png and prints inliers
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from pharaon_cv import _preprocess, orb_ransac_inliers


_HOMOG_ORB = cv2.ORB_create(nfeatures=8000, scaleFactor=1.2, nlevels=10)

# Module-level cache of reference-side ORB features keyed by id(reference_ndarray).
# Avoids recomputing 8000-feature ORB on the reference for every request.
_REF_FEATURE_CACHE: dict[int, tuple] = {}


def _reference_features(reference: np.ndarray):
    key = id(reference)
    cached = _REF_FEATURE_CACHE.get(key)
    if cached is not None:
        return cached
    ga = _preprocess(reference, size=800)
    ka, da = _HOMOG_ORB.detectAndCompute(ga, None)
    _REF_FEATURE_CACHE[key] = (ka, da)
    return ka, da


def find_homography(photo: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray | None, int, int]:
    """Return (H, inliers, good_matches) mapping reference -> photo coords."""
    # Reference-side ORB is cached across calls — only the photo side is computed each time.
    ka, da = _reference_features(reference)
    gb = _preprocess(photo, size=800)
    kb, db = _HOMOG_ORB.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return None, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(da, db, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return None, 0, len(good)

    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None, 0, len(good)

    # H maps preprocessed-reference (800x800) -> preprocessed-photo (800x800).
    # We need to compose with the scale transforms to map original-ref -> original-photo.
    h_ref, w_ref = reference.shape[:2]
    h_ph, w_ph = photo.shape[:2]

    # Letterbox scale factor used by _preprocess for each image
    s_ref = 800 / max(h_ref, w_ref)
    s_ph = 800 / max(h_ph, w_ph)

    # Padding offset (letterbox centers the image in the 800x800 canvas)
    nh_ref, nw_ref = int(round(h_ref * s_ref)), int(round(w_ref * s_ref))
    nh_ph, nw_ph = int(round(h_ph * s_ph)), int(round(w_ph * s_ph))
    pad_l_ref = (800 - nw_ref) // 2
    pad_t_ref = (800 - nh_ref) // 2
    pad_l_ph = (800 - nw_ph) // 2
    pad_t_ph = (800 - nh_ph) // 2

    # T_ref: original_ref_pixel -> preprocessed_ref_pixel
    T_ref = np.array([
        [s_ref, 0, pad_l_ref],
        [0, s_ref, pad_t_ref],
        [0, 0, 1],
    ], dtype=np.float64)
    # T_ph: original_photo_pixel -> preprocessed_photo_pixel
    T_ph = np.array([
        [s_ph, 0, pad_l_ph],
        [0, s_ph, pad_t_ph],
        [0, 0, 1],
    ], dtype=np.float64)

    # H_full maps original_ref_pixel -> original_photo_pixel
    H_full = np.linalg.inv(T_ph) @ H @ T_ref

    return H_full, int(mask.sum()), len(good)


def project_cells(H: np.ndarray, reference: np.ndarray) -> list[np.ndarray]:
    """Return the 9 cell quadrilaterals (4 corners each) projected into photo coords."""
    h, w = reference.shape[:2]
    ch, cw = h / 3, w / 3
    quads = []
    for r in range(3):
        for c in range(3):
            corners_ref = np.array([
                [c * cw, r * ch],
                [(c + 1) * cw, r * ch],
                [(c + 1) * cw, (r + 1) * ch],
                [c * cw, (r + 1) * ch],
            ], dtype=np.float64).reshape(-1, 1, 2)
            corners_ph = cv2.perspectiveTransform(corners_ref, H)
            quads.append(corners_ph.reshape(-1, 2))
    return quads


def warp_quad(photo: np.ndarray, quad: np.ndarray, out_size: int = 256) -> np.ndarray:
    """Warp a quadrilateral region from the photo into a square output of `out_size` pixels."""
    dst = np.array([[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]], dtype=np.float32)
    src = quad.astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(photo, M, (out_size, out_size))


def draw_quads(img: np.ndarray, quads: list[np.ndarray], labels: list[str] | None = None) -> np.ndarray:
    out = img.copy()
    for i, q in enumerate(quads):
        pts = q.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 220, 0), 3)
        if labels:
            cx, cy = q.mean(axis=0).astype(int)
            cv2.putText(out, labels[i], (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        else:
            r, c = divmod(i, 3)
            cx, cy = q.mean(axis=0).astype(int)
            cv2.putText(out, f"r{r}c{c}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python homography_align.py <photo> <reference> [debug_dir]", file=sys.stderr)
        return 2
    photo_p, ref_p = Path(sys.argv[1]), Path(sys.argv[2])
    photo = cv2.imread(str(photo_p))
    ref = cv2.imread(str(ref_p))
    if photo is None or ref is None:
        print("ERROR loading images", file=sys.stderr)
        return 2

    H, inliers, good = find_homography(photo, ref)
    print(f"H found, inliers={inliers}, good_matches={good}")
    if H is None:
        return 1

    # debug/<puzzle_name>/<photo_stem>_homography.png
    debug_root = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).parent / "debug"
    puzzle_dir = debug_root / photo_p.parent.name.replace(" ", "_")
    puzzle_dir.mkdir(parents=True, exist_ok=True)

    quads = project_cells(H, ref)
    out = draw_quads(photo, quads)
    out_path = puzzle_dir / f"{photo_p.stem}_homography.png"
    cv2.imwrite(str(out_path), out)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

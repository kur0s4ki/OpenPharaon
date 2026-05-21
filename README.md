# OpenPharaon — Markerless CV Puzzle Solver

Computer-vision detection of "is the 9-cube Pharaon puzzle solved?" using OpenCV only — no markers, no ML, no GPU. Designed to replace an RFID-based detection system that was suffering from card-fall reliability issues.

## What it does

Given a reference image of a solved puzzle (papyrus painting) and a camera photo of the physical 9-cube board:

1. **Auto-aligns** the photo to the reference via ORB feature matching + RANSAC homography (no manual slot calibration).
2. **Projects** the 9 reference cell corners into the photo to get exact slot quadrilaterals.
3. **Warps** each slot to a canonical square and matches it against the reference cell using ORB+RANSAC inliers.
4. **Reports** per-slot verdict — `MATCH` (correct face in correct slot), `WRONG_FACE` (right face, wrong slot — gives a free hint system), or `EMPTY` (cube showing a face not in this puzzle).

100% pass rate on 7 solved photos across 6 different puzzles at the time of writing.

## Pipeline at a glance

```
photo ──┐
        ├─► ORB+RANSAC ──► H (homography)
ref  ───┘                    │
                             ▼
                   project 9 ref cell quads into photo
                             │
                             ▼
                   warp each photo quad → 256×256 slot
                             │
                             ▼
                   per-slot ORB-RANSAC vs all 9 ref cells
                             │
                             ▼
                   verdict: MATCH | WRONG_FACE | EMPTY
```

## Files

| File | Purpose |
|---|---|
| `pharaon_cv.py` | Core ORB-RANSAC slot matcher + verdict logic |
| `homography_align.py` | Reference→photo homography + slot projection |
| `puzzles_batch_h.py` | Batch test runner using homography alignment (the production matcher) |
| `puzzles_batch.json` | Test catalogue: (photo, reference) pairs |
| `confusion.py` | Diagnostic: 9×9 photo-slot × ref-cell similarity matrix |
| `tests/puzzle N/` | Reference + solved prod photos for each puzzle |
| `debug/` | Auto-generated per-puzzle debug overlays (gitignored) |

## Run the batch validator

```bash
python puzzles_batch_h.py
```

Expected output: `OVERALL: 7/7 photos passed`. Debug overlays land in `debug/puzzle_N/<photo>_alignment.png`.

## Requirements

- Python 3.10+
- `opencv-contrib-python` (`img_hash` module is used in some older diagnostic scripts)
- `numpy`

## Production target

Raspberry Pi 4 (CPU-only). Pure OpenCV — no model files, no GPU. Per-frame runtime: under 1 second on Pi 4.

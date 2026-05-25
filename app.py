"""
OpenPharaon — single-file Flask web UI for testing the puzzle matcher.

Pick a puzzle, upload or capture a photo, get a SOLVED/NOT_SOLVED verdict
with a per-slot breakdown and an alignment overlay.

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import platform
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template_string, request, send_from_directory

from homography_align import (
    draw_quads,
    find_homography,
    project_cells,
    warp_quad,
)
from pharaon_cv import orb_ransac_inliers, precompute_orb, ransac_inliers_from_descriptors, verdict_for
from puzzles_batch_h import split_reference  # canonical splitter (rounded float)

ROOT = Path(__file__).parent
TESTS_DIR = ROOT / "tests"
UPLOADS_DIR = ROOT / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
CALIBRATION_PATH = ROOT / "calibration.json"

ORB_INLIERS_MIN = 3           # per-slot match threshold; leaves margin above RANSAC noise
H_INLIERS_MIN_RECOGNIZE = 8   # method B: minimum inliers to claim "this is puzzle X"
H_INLIERS_MIN_ALIGN = 30      # method A: minimum inliers for the projected quads to be trusted.
                              # Below this, the homography is degenerate and projecting cells
                              # produces nonsense quadrilaterals (zero-area, off-image, etc.).
                              # Solved photos observed at 64-225, scrambled / wrong-puzzle at 5-14.

# Make every check deterministic: same photo always returns the same per-slot
# scores, no run-to-run RANSAC variance.
cv2.setRNGSeed(42)

app = Flask(__name__)


# ---------- USB camera (Raspberry Pi target) ----------

class Camera:
    """Lazy-initialised USB camera wrapper.

    Opens /dev/video0 on first use, runs a background capture thread that
    keeps the latest frame in memory. Multiple HTTP requests share that
    single capture loop (only one process can own a camera at a time).
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 15):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: cv2.VideoCapture | None = None
        self.frame: np.ndarray | None = None
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.running = False
        self.error: str | None = None
        self.read_failures = 0
        self.read_successes = 0

    def _try_open(self):
        """Try multiple (backend, device-index) combinations, return the first
        combination that opens AND actually delivers a frame.

        On Raspberry Pi, the internal CSI/ISP subsystem grabs video0..video9-ish
        numbering, so a plugged-in USB webcam can land at /dev/video1 or
        higher. Trying indices 0..3 covers the common cases.
        """
        system = platform.system().lower()
        backends: list[tuple[int, str]] = []
        if system == "windows":
            backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY")]
        elif system == "linux":
            backends = [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "ANY")]
        else:
            backends = [(cv2.CAP_ANY, "ANY")]

        indices_to_try = [self.index] if self.index != 0 else [0, 1, 2, 3]

        tried = []
        for idx in indices_to_try:
            for backend, name in backends:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    tried.append(f"idx{idx}/{name}=not-open")
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.index = idx  # remember which one worked
                    self.error = None
                    return cap, f"idx{idx}/{name}"
                tried.append(f"idx{idx}/{name}=no-frame")
                cap.release()
        self.error = "tried " + ", ".join(tried)
        return None, None

    def ensure_started(self) -> bool:
        if self.running:
            return True
        with self.lock:
            if self.running:
                return True
            cap, backend = self._try_open()
            if cap is None:
                return False
            self.cap = cap
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            self.error = None
            return True

    def _loop(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self.lock:
                    self.frame = frame
                self.read_successes += 1
            else:
                self.read_failures += 1
                time.sleep(0.02)

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()


CAMERA = Camera()


def _mjpeg_stream():
    """Yield JPEG-encoded frames in multipart/x-mixed-replace format."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        frame = CAMERA.latest()
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            time.sleep(0.05)
            continue
        yield boundary + buf.tobytes() + b"\r\n"
        time.sleep(0.066)  # ~15 fps cap


# Cache all reference images on first access so we don't re-read from disk on every request.
_REF_CACHE: dict[str, np.ndarray] = {}

# Cache ORB features for the 9 reference cells of each puzzle. Computed once on
# first use, reused for every check. On a Pi this is the main perf win — slot
# matching now only computes features for the 9 photo slots (not also the 9 ref
# cells × N requests).
_REF_CELL_FEATURES: dict[str, list[tuple]] = {}


def get_reference_cell_features(puzzle_id: str, reference: np.ndarray) -> list[tuple]:
    cached = _REF_CELL_FEATURES.get(puzzle_id)
    if cached is not None:
        return cached
    cells = split_reference(reference)
    feats = [precompute_orb(c) for c in cells]
    _REF_CELL_FEATURES[puzzle_id] = feats
    return feats


def load_all_references() -> dict[str, np.ndarray]:
    if not _REF_CACHE:
        if TESTS_DIR.exists():
            for d in sorted(TESTS_DIR.iterdir()):
                if not d.is_dir():
                    continue
                ref_path = d / "ref.png"
                if ref_path.exists():
                    img = cv2.imread(str(ref_path))
                    if img is not None:
                        _REF_CACHE[d.name] = img
    return _REF_CACHE


# ---------- puzzle catalogue ----------

def discover_puzzles() -> list[dict]:
    """Find every tests/puzzle*/ref.png and report its id + sample photos."""
    out = []
    if not TESTS_DIR.exists():
        return out
    for d in sorted(TESTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        ref = d / "ref.png"
        if not ref.exists():
            continue
        sample_photos = sorted(p.name for p in d.glob("*.png") if p.name != "ref.png")
        out.append({"id": d.name, "ref": ref.name, "samples": sample_photos})
    return out


# ---------- matcher pipeline ----------

CLEAR_MATCH_FLOOR = 10  # per-slot inlier count above which a diagonal-best is considered "unambiguous"
CLEAR_MATCH_COUNT_FOR_LENIENT = 6  # how many clear matches required to switch to lenient mode


def _score_slots(photo_slots: list[np.ndarray], ref_cell_features: list[tuple]) -> list[dict]:
    """Two-pass per-slot scoring using PRECOMPUTED reference-cell ORB features.

    Pass 1: compute raw per-slot scores (own + best across all 9 ref cells).
    Determine "context_strong": how many slots have unambiguously correct
    matches (own >= CLEAR_MATCH_FLOOR AND diagonal-best). If at least 6/9,
    the puzzle is overall mostly solved, so verdict_for can apply its
    lenient tied-off-diagonal rule. In a partially-scrambled scene this
    count is low and the lenient rule stays OFF — preventing false MATCH
    on cubes whose noise-level scores happen to tie an off-cell.
    """
    # Compute photo-slot features ONCE each (instead of 9× per slot inside the loop).
    photo_features = [precompute_orb(s) for s in photo_slots]

    raw = []
    for i, (ka, da) in enumerate(photo_features):
        per_ref = [
            ransac_inliers_from_descriptors(ka, da, kb, db)
            for (kb, db) in ref_cell_features
        ]
        best_idx = int(np.argmax(per_ref))
        raw.append((i, int(per_ref[i]), int(per_ref[best_idx]), best_idx))

    clear_matches = sum(1 for (i, exp, _b, bi) in raw if exp >= CLEAR_MATCH_FLOOR and bi == i)
    context_strong = clear_matches >= CLEAR_MATCH_COUNT_FOR_LENIENT

    thr = {
        "orb_inliers_min": ORB_INLIERS_MIN,
        "wrong_face_min": 10,
        "wrong_face_margin": 5,
    }

    results = []
    for (i, exp, best, bi) in raw:
        v = verdict_for(exp, best, bi, i, thr, context_strong=context_strong)
        results.append({
            "row": i // 3,
            "col": i % 3,
            "expected_inliers": exp,
            "best_inliers": best,
            "best_match_index": bi,
            "best_match_label": f"r{bi // 3}c{bi % 3}",
            "verdict": v,
        })
    return results


def _draw_verdict_overlay(photo: np.ndarray, quads, slot_results) -> np.ndarray:
    color_for = {"MATCH": (0, 200, 0), "WRONG_FACE": (0, 165, 255), "EMPTY": (60, 60, 220)}
    out = photo.copy()
    for q, s in zip(quads, slot_results):
        col = color_for.get(s["verdict"], (200, 200, 200))
        pts = q.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, col, 4)
        cx, cy = q.mean(axis=0).astype(int)
        label = f"r{s['row']}c{s['col']} {s['verdict']}"
        if s["verdict"] == "WRONG_FACE":
            label += f" -> {s['best_match_label']}"
        cv2.putText(out, label, (cx - 70, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    return out


def _summarise(slot_results: list[dict]) -> dict:
    n_match = sum(1 for s in slot_results if s["verdict"] == "MATCH")
    n_wrong = sum(1 for s in slot_results if s["verdict"] == "WRONG_FACE")
    n_empty = sum(1 for s in slot_results if s["verdict"] == "EMPTY")
    return {
        "match_count": n_match,
        "wrong_face_count": n_wrong,
        "empty_count": n_empty,
        "verdict": "SOLVED" if n_match == 9 else "NOT_SOLVED",
    }


def _quads_are_sane(quads: list[np.ndarray]) -> tuple[bool, str]:
    """Reject pathological homographies before downstream processing.

    Catches mirror-flipped (reflected) homographies and severely degenerate
    grids — both happen when RANSAC finds a 'valid' homography against
    unrelated background content (wall paintings, decorations, etc.) that
    happens to share enough features with the reference.
    """
    if len(quads) != 9:
        return False, "expected 9 quads"

    signed_areas = []
    for q in quads:
        x = q[:, 0]
        y = q[:, 1]
        # Shoelace formula: positive for CCW, negative for CW (= reflection)
        sa = 0.5 * ((x[0]*y[1] - x[1]*y[0]) +
                    (x[1]*y[2] - x[2]*y[1]) +
                    (x[2]*y[3] - x[3]*y[2]) +
                    (x[3]*y[0] - x[0]*y[3]))
        signed_areas.append(sa)

    # All cells must have the same winding as the reference (positive area
    # under CCW vertex order). If any flipped, the homography is reflected.
    pos = sum(1 for s in signed_areas if s > 0)
    neg = sum(1 for s in signed_areas if s < 0)
    if neg > 0:
        return False, f"homography is reflected/flipped ({neg}/9 cells have inverted winding)"

    # Cell sizes should be roughly equal (real cubes are uniform).
    abs_areas = [abs(s) for s in signed_areas]
    if min(abs_areas) <= 10:
        return False, "one or more cells collapsed to near-zero area"
    if max(abs_areas) / min(abs_areas) > 4.0:
        return False, f"cell sizes vary too much (max/min = {max(abs_areas)/min(abs_areas):.1f}x)"

    return True, ""


def run_method_a(photo: np.ndarray, reference: np.ndarray, puzzle_id: str) -> dict:
    """Method A: homography-based reference→photo alignment, then per-slot match.
    puzzle_id is used as the cache key for precomputed reference-cell ORB features."""
    H, h_inliers, h_good = find_homography(photo, reference)
    if H is None or h_inliers < H_INLIERS_MIN_ALIGN:
        return {
            "ok": False,
            "method": "A",
            "name": "Homography (reference-aligned)",
            "error": (
                f"Could not align photo to reference (only {h_inliers} homography inliers, "
                f"need at least {H_INLIERS_MIN_ALIGN}). The cubes likely don't show this "
                f"puzzle's content."
            ),
            "h_inliers": h_inliers,
            "h_good_matches": h_good,
        }
    quads = project_cells(H, reference)
    sane, reason = _quads_are_sane(quads)
    if not sane:
        return {
            "ok": False,
            "method": "A",
            "name": "Homography (reference-aligned)",
            "error": (
                f"Found {h_inliers} matched features but the resulting alignment is "
                f"geometrically invalid ({reason}). The matcher likely locked onto "
                f"non-puzzle content (a wall painting or decoration). Re-frame the "
                f"camera so the cube grid fills most of the image."
            ),
            "h_inliers": h_inliers,
            "h_good_matches": h_good,
        }
    ref_cell_features = get_reference_cell_features(puzzle_id, reference)
    photo_slots = [warp_quad(photo, q, out_size=256) for q in quads]
    slots = _score_slots(photo_slots, ref_cell_features)
    overlay = _draw_verdict_overlay(photo, quads, slots)
    return {
        "ok": True,
        "method": "A",
        "name": "Homography (reference-aligned)",
        "h_inliers": h_inliers,
        "h_good_matches": h_good,
        "slots": slots,
        **_summarise(slots),
        "_overlay": overlay,
    }


def run_method_b(photo: np.ndarray, selected_puzzle_id: str, selected_reference: np.ndarray) -> dict:
    """Method B: multi-reference homography auto-detect.

    Tries homography against all 6 puzzle references. The reference with the
    most inliers identifies which puzzle is *physically* in the photo. We use
    that detected puzzle's projected quads as the 9 cube ROIs (so the overlay
    is always correct), then run per-slot matching against the user's
    *selected* reference. If the detected puzzle differs from the selected
    puzzle, we report NOT_SOLVED with a "wrong puzzle" diagnostic regardless
    of per-slot scores.
    """
    all_refs = load_all_references()

    per_puzzle: list[dict] = []
    best_id: str | None = None
    best_H = None
    best_inliers = 0
    best_good = 0
    for pid, ref_img in all_refs.items():
        H, inliers, good = find_homography(photo, ref_img)
        per_puzzle.append({"puzzle": pid, "h_inliers": inliers, "good_matches": good})
        if H is not None and inliers > best_inliers:
            best_id, best_H = pid, H
            best_inliers, best_good = inliers, good

    per_puzzle.sort(key=lambda r: -r["h_inliers"])

    if best_H is None or best_inliers < H_INLIERS_MIN_RECOGNIZE:
        return {
            "ok": False,
            "method": "B",
            "name": "Multi-reference homography (auto-detect)",
            "error": "Could not recognize any of the known puzzles in this photo.",
            "per_puzzle": per_puzzle,
        }

    # Project the 9 ROI quads from the DETECTED puzzle's reference
    detected_ref = all_refs[best_id]
    quads = project_cells(best_H, detected_ref)

    # Per-slot match against the SELECTED puzzle's reference (cached features).
    ref_cell_features = get_reference_cell_features(selected_puzzle_id, selected_reference)
    photo_slots = [warp_quad(photo, q, out_size=256) for q in quads]
    slots = _score_slots(photo_slots, ref_cell_features)
    summary = _summarise(slots)

    is_wrong_puzzle = best_id != selected_puzzle_id
    if is_wrong_puzzle:
        # Force NOT_SOLVED regardless of slot scores when the wrong puzzle is selected.
        summary["verdict"] = "NOT_SOLVED"

    overlay = _draw_verdict_overlay(photo, quads, slots)

    return {
        "ok": True,
        "method": "B",
        "name": "Multi-reference homography (auto-detect)",
        "detected_puzzle": best_id,
        "detected_h_inliers": best_inliers,
        "detected_good_matches": best_good,
        "is_wrong_puzzle": is_wrong_puzzle,
        "selected_puzzle": selected_puzzle_id,
        "per_puzzle": per_puzzle,
        "slots": slots,
        **summary,
        "_overlay": overlay,
    }


# ---------- routes ----------

@app.get("/")
def index():
    return render_template_string(INDEX_HTML, puzzles=discover_puzzles())


def _build_response(photo: np.ndarray, puzzle_id: str, reference: np.ndarray, photo_label: str, run_b: bool, t0: float) -> dict:
    """Shared matcher pipeline used by /check (file upload) and /check-camera (live frame)."""
    cv2.setRNGSeed(42)
    run_id = uuid.uuid4().hex[:10]
    method_a = run_method_a(photo, reference, puzzle_id)
    method_b = run_method_b(photo, puzzle_id, reference) if run_b else None

    if method_a.get("ok"):
        overlay_path = UPLOADS_DIR / f"{run_id}_a_overlay.jpg"
        cv2.imwrite(str(overlay_path), method_a.pop("_overlay"), [cv2.IMWRITE_JPEG_QUALITY, 85])
        method_a["overlay_url"] = f"/uploads/{overlay_path.name}"
    if method_b is not None and method_b.get("ok"):
        overlay_path = UPLOADS_DIR / f"{run_id}_b_overlay.jpg"
        cv2.imwrite(str(overlay_path), method_b.pop("_overlay"), [cv2.IMWRITE_JPEG_QUALITY, 85])
        method_b["overlay_url"] = f"/uploads/{overlay_path.name}"

    response = {
        "ok": True,
        "puzzle": puzzle_id,
        "photo_label": photo_label,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "method_a": method_a,
    }
    if method_b is not None:
        response["method_b"] = method_b
    return response


def _load_reference(puzzle_id: str):
    """Return (reference_ndarray, error_response_or_None)."""
    if not puzzle_id:
        return None, (jsonify(ok=False, error="puzzle is required"), 400)
    ref_path = TESTS_DIR / puzzle_id / "ref.png"
    if not ref_path.exists():
        return None, (jsonify(ok=False, error=f"reference for '{puzzle_id}' not found"), 404)
    reference = cv2.imread(str(ref_path))
    if reference is None:
        return None, (jsonify(ok=False, error="failed to read reference image"), 500)
    return reference, None


@app.post("/check")
def check():
    t0 = time.perf_counter()
    puzzle_id = request.form.get("puzzle", "").strip()
    reference, err = _load_reference(puzzle_id)
    if err:
        return err

    # Photo source: either uploaded file or a named sample from tests/
    sample = request.form.get("sample", "").strip()
    file = request.files.get("photo")
    if file and file.filename:
        photo_bytes = np.frombuffer(file.read(), np.uint8)
        photo = cv2.imdecode(photo_bytes, cv2.IMREAD_COLOR)
        photo_label = file.filename
    elif sample:
        sample_path = TESTS_DIR / puzzle_id / sample
        if not sample_path.exists():
            return jsonify(ok=False, error=f"sample '{sample}' not found"), 404
        photo = cv2.imread(str(sample_path))
        photo_label = sample
    else:
        return jsonify(ok=False, error="provide a photo file or sample name"), 400

    if photo is None:
        return jsonify(ok=False, error="failed to decode photo"), 400

    methods = request.form.get("methods", "a").lower()
    run_b = "b" in methods
    return jsonify(_build_response(photo, puzzle_id, reference, photo_label, run_b, t0))


@app.get("/camera/status")
def camera_status():
    available = CAMERA.ensure_started()
    w = h = None
    if available and CAMERA.cap is not None:
        # Actual delivered resolution (may differ from what we requested if
        # the cam can't match e.g. 1280x720).
        w = int(CAMERA.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(CAMERA.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return jsonify(
        available=available,
        error=CAMERA.error,
        read_successes=CAMERA.read_successes,
        read_failures=CAMERA.read_failures,
        frame_width=w,
        frame_height=h,
        requested_width=CAMERA.width,
        requested_height=CAMERA.height,
    )


@app.get("/stream")
def stream():
    if not CAMERA.ensure_started():
        return Response(f"camera unavailable: {CAMERA.error}", status=503)
    return Response(_mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/check-camera")
def check_camera():
    t0 = time.perf_counter()
    puzzle_id = request.form.get("puzzle", "").strip()
    reference, err = _load_reference(puzzle_id)
    if err:
        return err

    if not CAMERA.ensure_started():
        return jsonify(ok=False, error=f"camera unavailable: {CAMERA.error}"), 503
    photo = CAMERA.latest()
    if photo is None:
        return jsonify(ok=False, error="no frame from camera yet — try again in a moment"), 503

    methods = request.form.get("methods", "a").lower()
    run_b = "b" in methods
    return jsonify(_build_response(photo, puzzle_id, reference, "live camera capture", run_b, t0))


# ---------- Locked-ROI calibration (install-time, separate from /check) ----------

def _load_calibration() -> dict | None:
    if not CALIBRATION_PATH.exists():
        return None
    try:
        return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save_calibration(data: dict) -> None:
    CALIBRATION_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


@app.get("/snap")
def snap():
    """Return ONE still JPEG from the camera. Used by /locked's calibration canvas."""
    if not CAMERA.ensure_started():
        return Response(f"camera unavailable: {CAMERA.error}", status=503)
    frame = CAMERA.latest()
    if frame is None:
        return Response("no frame yet", status=503)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return Response("encode failed", status=500)
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.get("/calibration")
def get_calibration():
    return jsonify(_load_calibration() or {"boxes": []})


@app.post("/calibration")
def post_calibration():
    data = request.get_json(silent=True)
    if not data or not isinstance(data.get("boxes"), list) or len(data["boxes"]) != 9:
        return jsonify(ok=False, error="need exactly 9 boxes"), 400
    cleaned = []
    for b in data["boxes"]:
        if not all(k in b for k in ("x1", "y1", "x2", "y2")):
            return jsonify(ok=False, error="each box must have x1,y1,x2,y2"), 400
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        if x2 <= x1 or y2 <= y1:
            return jsonify(ok=False, error=f"invalid box: ({x1},{y1})-({x2},{y2})"), 400
        cleaned.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    out = {"boxes": cleaned, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if "frame_size" in data:
        out["frame_size"] = data["frame_size"]
    _save_calibration(out)
    return jsonify(ok=True, saved_at=out["saved_at"])


@app.post("/check-locked")
def check_locked():
    """Per-slot match using the saved calibration boxes. No homography, no
    projection — direct crop of fixed pixel rects. Fast and deterministic."""
    t0 = time.perf_counter()
    puzzle_id = request.form.get("puzzle", "").strip()
    reference, err = _load_reference(puzzle_id)
    if err:
        return err

    cal = _load_calibration()
    if not cal or not cal.get("boxes"):
        return jsonify(
            ok=False,
            error="No calibration saved. Open /locked and draw the 9 cube ROIs first.",
        ), 400

    if not CAMERA.ensure_started():
        return jsonify(ok=False, error=f"camera unavailable: {CAMERA.error}"), 503
    photo = CAMERA.latest()
    if photo is None:
        return jsonify(ok=False, error="no frame available yet"), 503

    cv2.setRNGSeed(42)
    h_photo, w_photo = photo.shape[:2]

    photo_slots = []
    quads = []
    for b in cal["boxes"]:
        x1 = max(0, min(int(b["x1"]), w_photo - 1))
        y1 = max(0, min(int(b["y1"]), h_photo - 1))
        x2 = max(x1 + 1, min(int(b["x2"]), w_photo))
        y2 = max(y1 + 1, min(int(b["y2"]), h_photo))
        crop = photo[y1:y2, x1:x2]
        photo_slots.append(cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA))
        quads.append(np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))

    ref_cell_features = get_reference_cell_features(puzzle_id, reference)
    slots = _score_slots(photo_slots, ref_cell_features)
    summary = _summarise(slots)

    overlay = _draw_verdict_overlay(photo, quads, slots)
    run_id = uuid.uuid4().hex[:10]
    overlay_path = UPLOADS_DIR / f"{run_id}_locked_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return jsonify({
        "ok": True,
        "method": "locked",
        "puzzle": puzzle_id,
        "photo_label": "live camera capture (locked ROI)",
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "slots": slots,
        **summary,
        "overlay_url": f"/uploads/{overlay_path.name}",
    })


@app.get("/locked")
def locked_page():
    return render_template_string(LOCKED_HTML, puzzles=discover_puzzles())


@app.get("/uploads/<path:name>")
def serve_upload(name: str):
    return send_from_directory(UPLOADS_DIR, name)


@app.get("/tests/<path:p>")
def serve_test_asset(p: str):
    full = (TESTS_DIR / p).resolve()
    if TESTS_DIR.resolve() not in full.parents and full != TESTS_DIR.resolve():
        abort(404)
    if not full.exists():
        abort(404)
    return send_from_directory(full.parent, full.name)


# ---------- HTML ----------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenPharaon — puzzle check</title>
  <style>
    :root {
      --bg: #14110d;
      --panel: #1f1b14;
      --panel-2: #2a241a;
      --ink: #f3e6c8;
      --muted: #a89671;
      --gold: #d8a84b;
      --ok: #4ac779;
      --warn: #f0a531;
      --bad: #e15252;
      --border: #3a3122;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 24px;
      font: 15px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      background: var(--bg); color: var(--ink);
      min-height: 100vh;
    }
    h1 { font-size: 22px; margin: 0 0 16px; color: var(--gold); letter-spacing: 0.5px; }
    .wrap { max-width: 1080px; margin: 0 auto; }
    .panel {
      background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
      padding: 18px; margin-bottom: 16px;
    }
    label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.6px; }
    select, input[type=file] {
      background: var(--panel-2); color: var(--ink); border: 1px solid var(--border);
      padding: 10px 12px; border-radius: 8px; font-size: 15px; width: 100%;
    }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 720px) { .row { grid-template-columns: 1fr; } }
    button {
      background: var(--gold); color: #1a1408; border: 0; border-radius: 8px;
      padding: 12px 18px; font-size: 15px; font-weight: 600; cursor: pointer;
    }
    button.secondary { background: var(--panel-2); color: var(--ink); border: 1px solid var(--border); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }

    .preview { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }
    .preview img { max-height: 140px; border-radius: 6px; border: 1px solid var(--border); }

    .verdict {
      font-size: 28px; font-weight: 700; padding: 14px 18px; border-radius: 10px;
      text-align: center; letter-spacing: 1px;
    }
    .verdict.solved { background: rgba(74, 199, 121, 0.12); color: var(--ok); border: 1px solid rgba(74, 199, 121, 0.4); }
    .verdict.notsolved { background: rgba(225, 82, 82, 0.10); color: var(--bad); border: 1px solid rgba(225, 82, 82, 0.4); }
    .verdict.error { background: rgba(240, 165, 49, 0.10); color: var(--warn); border: 1px solid rgba(240, 165, 49, 0.4); font-size: 18px; }

    .stats { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; color: var(--muted); font-size: 13px; }
    .stats b { color: var(--ink); }

    table.slots { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }
    table.slots th, table.slots td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }
    table.slots th { color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; font-size: 11px; }
    .badge { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; letter-spacing: 0.4px; }
    .badge.MATCH { background: rgba(74, 199, 121, 0.18); color: var(--ok); }
    .badge.WRONG_FACE { background: rgba(240, 165, 49, 0.18); color: var(--warn); }
    .badge.EMPTY { background: rgba(225, 82, 82, 0.10); color: var(--bad); }

    .overlay-wrap { margin-top: 12px; }
    .overlay-wrap img { max-width: 100%; border-radius: 8px; border: 1px solid var(--border); }

    .methods-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 980px) { .methods-row { grid-template-columns: 1fr; } }
    .method-card { background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
    .method-card h3 { margin: 0 0 4px; color: var(--gold); font-size: 14px; letter-spacing: 0.5px; text-transform: uppercase; }
    .method-card .sub { color: var(--muted); font-size: 12px; margin-bottom: 10px; }

    .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.2); border-top-color: var(--ink); border-radius: 50%; animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 8px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .hidden { display: none !important; }
    .hint { color: var(--muted); font-size: 12px; margin-top: 6px; }

    .live-cam-wrap { position: relative; background: #000; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); max-width: 100%; }
    .live-cam-wrap img { display: block; width: 100%; max-height: 360px; object-fit: contain; background: #000; }
    .live-cam-wrap .dot { position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,0.6); color: #fff; padding: 4px 10px; border-radius: 4px; font-size: 11px; letter-spacing: 0.4px; }
    .live-cam-wrap .dot::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #e15252; margin-right: 6px; vertical-align: middle; animation: pulse 1.4s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  </style>
</head>
<body>
<div class="wrap">
  <h1>OpenPharaon — puzzle solver check</h1>

  <div class="panel">
    <div class="row">
      <div>
        <label for="puzzle">Puzzle</label>
        <select id="puzzle">
          {% for p in puzzles %}
          <option value="{{ p.id }}" data-samples='{{ p.samples|tojson }}' data-ref="{{ p.ref }}">{{ p.id }}</option>
          {% endfor %}
        </select>
        <div id="refPreview" class="preview"></div>
      </div>
      <div>
        <label for="photo">Photo — upload or capture</label>
        <input type="file" id="photo" accept="image/*" capture="environment">
        <div class="hint">On mobile this opens the camera. On desktop it picks a file.</div>
        <div id="filePreview" class="preview"></div>
      </div>
    </div>

    <div style="margin-top: 18px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
      <button id="run">Check uploaded photo</button>
      <button id="reset" class="secondary">Clear</button>
      <span id="status" class="hint"></span>
    </div>
  </div>

  <div class="panel" id="cameraPanel">
    <label>Live USB camera</label>
    <div id="cameraLoading" class="hint">Checking for connected camera…</div>
    <div id="cameraUnavailable" class="hint hidden">
      USB camera not detected. <span id="cameraError"></span> Use the upload above instead.
    </div>
    <div id="cameraView" class="hidden">
      <div class="live-cam-wrap">
        <img id="liveStream" alt="live camera feed">
        <span class="dot">LIVE</span>
      </div>
      <div style="margin-top: 12px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
        <button id="captureBtn">Capture &amp; check</button>
        <span id="cameraStatus" class="hint"></span>
      </div>
    </div>
  </div>

  <div id="resultPanel" class="panel hidden">
    <div id="topStats" class="stats" style="margin-bottom: 12px;"></div>
    <div id="methodsRow" class="methods-row"></div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const puzzleSel = $("puzzle");
const photoInput = $("photo");
const refPreview = $("refPreview");
const filePreview = $("filePreview");
const statusEl = $("status");
const resultPanel = $("resultPanel");
const topStatsEl = $("topStats");
const methodsRow = $("methodsRow");

function refreshPuzzleUI() {
  const opt = puzzleSel.selectedOptions[0];
  const id = opt.value;
  refPreview.innerHTML = `<img src="/tests/${encodeURIComponent(id)}/${opt.dataset.ref}" alt="reference">`;
}

photoInput.addEventListener("change", () => {
  filePreview.innerHTML = "";
  const f = photoInput.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  filePreview.innerHTML = `<img src="${url}" alt="uploaded">`;
});

puzzleSel.addEventListener("change", refreshPuzzleUI);
refreshPuzzleUI();

$("reset").onclick = () => {
  photoInput.value = "";
  filePreview.innerHTML = "";
  resultPanel.classList.add("hidden");
  statusEl.textContent = "";
};

// ---------- Live camera ----------
async function initCamera() {
  try {
    const res = await fetch("/camera/status");
    const data = await res.json();
    $("cameraLoading").classList.add("hidden");
    if (data.available) {
      $("cameraView").classList.remove("hidden");
      $("liveStream").src = "/stream";
    } else {
      $("cameraUnavailable").classList.remove("hidden");
      if (data.error) $("cameraError").textContent = "(" + data.error + ")";
    }
  } catch (e) {
    $("cameraLoading").classList.add("hidden");
    $("cameraUnavailable").classList.remove("hidden");
    $("cameraError").textContent = "(" + e.message + ")";
  }
}
initCamera();

$("captureBtn") && ($("captureBtn").onclick = async () => {
  $("cameraStatus").innerHTML = '<span class="spinner"></span>Capturing and matching…';
  $("captureBtn").disabled = true;
  resultPanel.classList.add("hidden");
  try {
    const fd = new FormData();
    fd.append("puzzle", puzzleSel.value);
    const res = await fetch("/check-camera", { method: "POST", body: fd });
    const data = await res.json();
    renderResult(data);
  } catch (e) {
    renderError("Network error: " + e.message);
  } finally {
    $("captureBtn").disabled = false;
    $("cameraStatus").textContent = "";
  }
});

$("run").onclick = async () => {
  const f = photoInput.files[0];
  if (!f) {
    statusEl.textContent = "Pick a photo first.";
    return;
  }
  statusEl.innerHTML = '<span class="spinner"></span>Running matcher…';
  $("run").disabled = true;
  resultPanel.classList.add("hidden");

  const fd = new FormData();
  fd.append("puzzle", puzzleSel.value);
  fd.append("photo", f);

  try {
    const res = await fetch("/check", { method: "POST", body: fd });
    const data = await res.json();
    renderResult(data);
  } catch (e) {
    renderError("Network error: " + e.message);
  } finally {
    $("run").disabled = false;
    statusEl.textContent = "";
  }
};

function renderResult(d) {
  resultPanel.classList.remove("hidden");
  if (!d.ok) { renderError(d.error || "Unknown error"); return; }

  topStatsEl.innerHTML = `
    <span>Photo: <b>${d.photo_label}</b></span>
    <span>Puzzle: <b>${d.puzzle}</b></span>
    <span>Total time: <b>${d.elapsed_ms} ms</b></span>
  `;

  methodsRow.innerHTML = "";
  methodsRow.appendChild(renderMethodCard(d.method_a));
  if (d.method_b) {
    methodsRow.appendChild(renderMethodCard(d.method_b));
    methodsRow.style.gridTemplateColumns = "1fr 1fr";
  } else {
    methodsRow.style.gridTemplateColumns = "1fr";
  }
}

function renderMethodCard(m) {
  const card = document.createElement("div");
  card.className = "method-card";
  let header = `<h3>Method ${m.method} — ${m.name}</h3>`;

  if (!m.ok) {
    card.innerHTML = header + `
      <div class="verdict error" style="font-size:16px;margin-top:8px;">${m.error || "failed"}</div>
    `;
    return card;
  }

  const verdictCls = m.verdict === "SOLVED" ? "solved" : "notsolved";
  let extra = "";
  if (m.method === "A") {
    extra = `<span>Homography inliers: <b>${m.h_inliers}</b> / ${m.h_good_matches}</span>`;
  } else if (m.method === "B") {
    let detectedNote = `<span>Detected: <b>${m.detected_puzzle}</b> (${m.detected_h_inliers} inliers)</span>`;
    if (m.is_wrong_puzzle) {
      detectedNote += ` <span style="color:var(--warn);">— differs from selected (${m.selected_puzzle})</span>`;
    }
    extra = detectedNote;
  }

  let body = `
    ${header}
    <div class="verdict ${verdictCls}" style="font-size:20px;margin-bottom:8px;">${m.verdict}</div>
    <div class="stats">
      <span>Match / Wrong / Empty: <b>${m.match_count} / ${m.wrong_face_count} / ${m.empty_count}</b></span>
      ${extra}
    </div>`;

  if (m.overlay_url) {
    body += `<div class="overlay-wrap"><img src="${m.overlay_url}" alt="overlay"></div>`;
  }

  body += `<table class="slots"><thead><tr><th>Slot</th><th>Own</th><th>Best</th><th>Best @</th><th>Verdict</th></tr></thead><tbody>`;
  m.slots.forEach(s => {
    const slot = `r${s.row}c${s.col}`;
    body += `<tr>
      <td><b>${slot}</b></td>
      <td>${s.expected_inliers}</td>
      <td>${s.best_inliers}</td>
      <td>${s.best_match_label}</td>
      <td><span class="badge ${s.verdict}">${s.verdict}</span></td>
    </tr>`;
  });
  body += "</tbody></table>";
  card.innerHTML = body;
  return card;
}

function renderError(msg) {
  topStatsEl.innerHTML = `<span class="verdict error" style="font-size:14px;">${msg}</span>`;
  methodsRow.innerHTML = "";
}
</script>
</body>
</html>
"""


LOCKED_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenPharaon — Locked ROI</title>
<style>
  :root {
    --bg: #14110d; --panel: #1f1b14; --panel-2: #2a241a;
    --ink: #f3e6c8; --muted: #a89671; --gold: #d8a84b;
    --ok: #4ac779; --warn: #f0a531; --bad: #e15252;
    --border: #3a3122;
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:24px; font:15px/1.45 system-ui, sans-serif; background:var(--bg); color:var(--ink); }
  h1 { font-size:22px; color:var(--gold); margin:0 0 12px; }
  .wrap { max-width:1080px; margin:0 auto; }
  .panel { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:18px; margin-bottom:16px; }
  label { display:block; font-size:12px; color:var(--muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:0.6px; }
  select { background:var(--panel-2); color:var(--ink); border:1px solid var(--border); padding:10px 12px; border-radius:8px; width:100%; max-width:300px; }
  button { background:var(--gold); color:#1a1408; border:0; border-radius:8px; padding:10px 16px; font-weight:600; cursor:pointer; margin-right:8px; }
  button.secondary { background:var(--panel-2); color:var(--ink); border:1px solid var(--border); }
  button:disabled { opacity:0.5; cursor:not-allowed; }
  .canvas-wrap { position:relative; max-width:100%; margin-top:12px; border:1px solid var(--border); border-radius:8px; overflow:hidden; }
  .canvas-wrap img, .canvas-wrap canvas { display:block; max-width:100%; width:100%; height:auto; }
  .canvas-wrap canvas { position:absolute; top:0; left:0; cursor:crosshair; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
  @media (max-width: 1100px) { .two-col { grid-template-columns: 1fr; } }
  .hint { color:var(--muted); font-size:13px; margin-top:8px; }
  .verdict { font-size:22px; font-weight:700; padding:12px 16px; border-radius:10px; text-align:center; letter-spacing:0.6px; margin-top:12px; }
  .verdict.solved { background:rgba(74,199,121,0.12); color:var(--ok); border:1px solid rgba(74,199,121,0.4); }
  .verdict.notsolved { background:rgba(225,82,82,0.10); color:var(--bad); border:1px solid rgba(225,82,82,0.4); }
  .verdict.error { background:rgba(240,165,49,0.10); color:var(--warn); border:1px solid rgba(240,165,49,0.4); font-size:16px; }
  table { width:100%; border-collapse:collapse; margin-top:10px; font-size:13px; }
  th, td { padding:6px 10px; border-bottom:1px solid var(--border); text-align:left; }
  th { color:var(--muted); text-transform:uppercase; font-size:11px; }
  .badge { padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; }
  .badge.MATCH { background:rgba(74,199,121,0.18); color:var(--ok); }
  .badge.WRONG_FACE { background:rgba(240,165,49,0.18); color:var(--warn); }
  .badge.EMPTY { background:rgba(225,82,82,0.10); color:var(--bad); }
  .row { display:flex; gap:8px; align-items:center; margin-top:10px; flex-wrap:wrap; }
  a { color:var(--gold); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Locked-ROI mode (install-time calibration)</h1>
  <div class="hint">Camera is fixed. Draw 9 boxes <b>once</b> over the cube grid; every check after that crops those exact pixels — no homography, no auto-detection. <a href="/">← back to main</a></div>

  <div class="two-col">
    <div class="panel">
      <label for="puzzle">Puzzle</label>
      <select id="puzzle">
        {% for p in puzzles %}<option value="{{ p.id }}">{{ p.id }}</option>{% endfor %}
      </select>

      <div class="row">
        <button id="snapBtn">Snap & Calibrate</button>
        <button id="resumeBtn" class="secondary">Resume live</button>
        <button id="clearBtn" class="secondary">Clear boxes</button>
        <button id="saveBtn" class="secondary" disabled>Save calibration</button>
        <button id="checkBtn">Check now (locked)</button>
      </div>
      <div class="hint" id="boxCounter">Boxes drawn: 0 / 9 — drag to draw, row-major order (r0c0, r0c1, r0c2, r1c0, …)</div>
      <div class="hint" id="status"></div>

      <div class="canvas-wrap">
        <img id="bg" src="/stream" alt="camera">
        <canvas id="overlay"></canvas>
      </div>
    </div>

    <div id="resultPanel" class="panel" style="opacity:0.4; min-height: 200px;">
      <div id="verdict" class="verdict" style="background: var(--panel-2); color: var(--muted); border-color: var(--border);">No check yet</div>
      <div id="stats" class="hint" style="margin-top:8px"></div>
      <div id="tableWrap"></div>
    </div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const bg = $("bg"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const statusEl = $("status"), counterEl = $("boxCounter");
let boxes = [];      // saved boxes in IMAGE-pixel coords {x1,y1,x2,y2}
let dragging = false, startPt = null, drawingEnabled = false;
let imgNaturalSize = null;

function fitCanvasToImage() {
  if (!imgNaturalSize) return;
  canvas.width  = imgNaturalSize.w;
  canvas.height = imgNaturalSize.h;
  redraw();
}

bg.addEventListener("load", () => {
  imgNaturalSize = { w: bg.naturalWidth, h: bg.naturalHeight };
  fitCanvasToImage();
});

function clientToImage(ev) {
  const rect = bg.getBoundingClientRect();
  const sx = imgNaturalSize.w / rect.width;
  const sy = imgNaturalSize.h / rect.height;
  return {
    x: Math.round((ev.clientX - rect.left) * sx),
    y: Math.round((ev.clientY - rect.top) * sy),
  };
}

function redraw(preview) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "bold 22px sans-serif";
  boxes.forEach((b, i) => {
    ctx.strokeStyle = "#4ac779";
    ctx.lineWidth = 4;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    const r = i / 3 | 0, c = i % 3;
    const label = `r${r}c${c}`;
    const padding = 6;
    ctx.fillRect(b.x1 + 4, b.y1 + 4, 70, 28);
    ctx.fillStyle = "#4ac779";
    ctx.fillText(label, b.x1 + 8, b.y1 + 26);
  });
  if (preview) {
    ctx.strokeStyle = "#d8a84b";
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 6]);
    ctx.strokeRect(preview.x1, preview.y1, preview.x2 - preview.x1, preview.y2 - preview.y1);
    ctx.setLineDash([]);
  }
  counterEl.textContent = `Boxes drawn: ${boxes.length} / 9 — drag to draw next, row-major (r0c0, r0c1, r0c2, r1c0, …)`;
  $("saveBtn").disabled = boxes.length !== 9;
}

canvas.addEventListener("mousedown", (ev) => {
  if (!drawingEnabled || boxes.length >= 9) return;
  dragging = true;
  startPt = clientToImage(ev);
});
canvas.addEventListener("mousemove", (ev) => {
  if (!dragging) return;
  const p = clientToImage(ev);
  redraw({
    x1: Math.min(startPt.x, p.x), y1: Math.min(startPt.y, p.y),
    x2: Math.max(startPt.x, p.x), y2: Math.max(startPt.y, p.y),
  });
});
canvas.addEventListener("mouseup", (ev) => {
  if (!dragging) return;
  dragging = false;
  const p = clientToImage(ev);
  const b = {
    x1: Math.min(startPt.x, p.x), y1: Math.min(startPt.y, p.y),
    x2: Math.max(startPt.x, p.x), y2: Math.max(startPt.y, p.y),
  };
  if ((b.x2 - b.x1) < 10 || (b.y2 - b.y1) < 10) { redraw(); return; }
  boxes.push(b);
  redraw();
});

$("snapBtn").onclick = () => {
  // Freeze the bg on the latest snapshot (so user can draw on a still image)
  bg.src = "/snap?ts=" + Date.now();
  drawingEnabled = true;
  boxes = [];
  statusEl.textContent = "Drawing mode: drag 9 boxes in row-major order.";
};
$("resumeBtn").onclick = () => {
  bg.src = "/stream";
  drawingEnabled = false;
  statusEl.textContent = "Live preview resumed.";
};
$("clearBtn").onclick = () => { boxes = []; redraw(); };

$("saveBtn").onclick = async () => {
  if (boxes.length !== 9) return;
  statusEl.textContent = "Saving…";
  const res = await fetch("/calibration", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ boxes, frame_size: imgNaturalSize })
  });
  const data = await res.json();
  if (data.ok) {
    statusEl.textContent = `Calibration saved (${data.saved_at}).`;
  } else {
    statusEl.textContent = "Save failed: " + (data.error || "unknown");
  }
};

$("checkBtn").onclick = async () => {
  $("checkBtn").disabled = true;
  $("resultPanel").style.opacity = "1";
  statusEl.textContent = "Capturing & matching…";
  const fd = new FormData();
  fd.append("puzzle", $("puzzle").value);
  try {
    const res = await fetch("/check-locked", { method: "POST", body: fd });
    const d = await res.json();
    renderResult(d);
  } catch (e) {
    renderError("Network error: " + e.message);
  } finally {
    $("checkBtn").disabled = false;
    statusEl.textContent = "";
  }
};

function renderResult(d) {
  $("resultPanel").style.opacity = "1";
  const v = $("verdict");
  if (!d.ok) { v.className = "verdict error"; v.textContent = d.error || "error"; v.style = ""; $("stats").textContent = ""; $("tableWrap").innerHTML = ""; return; }
  v.className = "verdict " + (d.verdict === "SOLVED" ? "solved" : "notsolved");
  v.textContent = d.verdict;
  v.style = "";
  $("stats").textContent = `Match ${d.match_count} / Wrong ${d.wrong_face_count} / Empty ${d.empty_count}   |   ${d.elapsed_ms} ms`;
  let html = '<table><thead><tr><th>Slot</th><th>Own</th><th>Best</th><th>Best @</th><th>Verdict</th></tr></thead><tbody>';
  d.slots.forEach(s => {
    html += `<tr><td><b>r${s.row}c${s.col}</b></td><td>${s.expected_inliers}</td><td>${s.best_inliers}</td><td>${s.best_match_label}</td><td><span class="badge ${s.verdict}">${s.verdict}</span></td></tr>`;
  });
  html += '</tbody></table>';
  if (d.overlay_url) html += `<div style="margin-top:12px"><img src="${d.overlay_url}" style="max-width:100%; border-radius:8px"></div>`;
  $("tableWrap").innerHTML = html;
}
function renderError(m) { $("resultPanel").style.opacity = "1"; const v=$("verdict"); v.className="verdict error"; v.textContent=m; v.style = ""; $("stats").textContent=""; $("tableWrap").innerHTML=""; }

// Load existing calibration on page load
(async () => {
  try {
    const r = await fetch("/calibration");
    const d = await r.json();
    if (d.boxes && d.boxes.length === 9) {
      boxes = d.boxes;
      statusEl.textContent = "Loaded saved calibration (" + (d.saved_at || "") + "). Click Snap & Calibrate to redraw.";
      // Wait for image to load before drawing on it
      if (imgNaturalSize) redraw();
    }
  } catch (e) {}
})();
</script>
</body></html>
"""


if __name__ == "__main__":
    print(f"Discovered puzzles: {[p['id'] for p in discover_puzzles()]}")
    # host=0.0.0.0 so the Pi-deployed UI is reachable from other devices on the LAN.
    # threaded=True so MJPEG streaming on /stream doesn't block other requests.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

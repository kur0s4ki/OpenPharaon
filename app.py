"""
OpenPharaon — locked-camera puzzle verifier.

One workflow:
  1. Camera is bolted in front of the cube grid (does not move).
  2. Operator calibrates ONCE by drawing 9 rectangles over the cubes
     in the camera image (`/` page, "Snap & Calibrate").
  3. Every check captures the latest frame, crops those 9 fixed pixel
     rectangles, and matches each crop against the chosen puzzle's
     reference cells.

No homography, no auto-detection, no image-content guessing.
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
from flask import (
    Flask, Response, abort, jsonify, render_template_string,
    request, send_from_directory,
)

from pharaon_cv import (
    precompute_orb,
    ransac_inliers_from_descriptors,
    verdict_for,
)

# ---------- paths & constants ----------

ROOT = Path(__file__).parent
TESTS_DIR = ROOT / "tests"
UPLOADS_DIR = ROOT / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
CALIBRATION_PATH = ROOT / "calibration.json"

ORB_INLIERS_MIN = 3
CLEAR_MATCH_FLOOR = 10
CLEAR_MATCH_COUNT_FOR_LENIENT = 6

# Deterministic RANSAC: same input → same scores every time.
cv2.setRNGSeed(42)

app = Flask(__name__)


# ---------- USB camera ----------

class Camera:
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
        system = platform.system().lower()
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
                    self.index = idx
                    self.error = None
                    return cap
                tried.append(f"idx{idx}/{name}=no-frame")
                cap.release()
        self.error = "tried " + ", ".join(tried)
        return None

    def ensure_started(self) -> bool:
        if self.running:
            return True
        with self.lock:
            if self.running:
                return True
            cap = self._try_open()
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


# ---------- references & caches ----------

_REF_CACHE: dict[str, np.ndarray] = {}
_REF_CELL_FEATURES: dict[str, list[tuple]] = {}


def split_reference(reference: np.ndarray) -> list[np.ndarray]:
    """Split a reference image into 9 cells (row-major) with rounded float
    boundaries so each cell gets the same pixel count regardless of whether
    H/W are divisible by 3."""
    h, w = reference.shape[:2]
    ch, cw = h / 3, w / 3
    cells = []
    for r in range(3):
        for c in range(3):
            y1, y2 = int(round(r * ch)), int(round((r + 1) * ch))
            x1, x2 = int(round(c * cw)), int(round((c + 1) * cw))
            cells.append(reference[y1:y2, x1:x2].copy())
    return cells


def discover_puzzles() -> list[dict]:
    out = []
    if not TESTS_DIR.exists():
        return out
    for d in sorted(TESTS_DIR.iterdir()):
        if d.is_dir() and (d / "ref.png").exists():
            out.append({"id": d.name, "ref": "ref.png"})
    return out


def load_reference(puzzle_id: str) -> np.ndarray | None:
    if puzzle_id in _REF_CACHE:
        return _REF_CACHE[puzzle_id]
    ref_path = TESTS_DIR / puzzle_id / "ref.png"
    if not ref_path.exists():
        return None
    img = cv2.imread(str(ref_path))
    if img is not None:
        _REF_CACHE[puzzle_id] = img
    return img


def get_reference_cell_features(puzzle_id: str, reference: np.ndarray) -> list[tuple]:
    """Cache ORB keypoints+descriptors for the 9 cells of each puzzle.
    Computed once on first access, reused for every check. The single
    biggest perf win on a Pi."""
    cached = _REF_CELL_FEATURES.get(puzzle_id)
    if cached is not None:
        return cached
    feats = [precompute_orb(c) for c in split_reference(reference)]
    _REF_CELL_FEATURES[puzzle_id] = feats
    return feats


# ---------- calibration ----------

def load_calibration() -> dict | None:
    if not CALIBRATION_PATH.exists():
        return None
    try:
        return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_calibration(data: dict) -> None:
    CALIBRATION_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------- matcher pipeline ----------

def score_slots(photo_slots: list[np.ndarray], ref_cell_features: list[tuple]) -> list[dict]:
    """Match each of 9 photo crops against the 9 reference cells, return
    verdict records. Uses cached reference features for speed."""
    photo_features = [precompute_orb(s) for s in photo_slots]

    raw = []
    for i, (ka, da) in enumerate(photo_features):
        per_ref = [
            ransac_inliers_from_descriptors(ka, da, kb, db)
            for (kb, db) in ref_cell_features
        ]
        best_idx = int(np.argmax(per_ref))
        raw.append((i, int(per_ref[i]), int(per_ref[best_idx]), best_idx))

    clear_matches = sum(1 for (i, exp, _b, bi) in raw
                        if exp >= CLEAR_MATCH_FLOOR and bi == i)
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


def draw_overlay(photo: np.ndarray, quads, slot_results) -> np.ndarray:
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
        cv2.putText(out, label, (cx - 70, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    return out


def summarise(slot_results: list[dict]) -> dict:
    n_match = sum(1 for s in slot_results if s["verdict"] == "MATCH")
    n_wrong = sum(1 for s in slot_results if s["verdict"] == "WRONG_FACE")
    n_empty = sum(1 for s in slot_results if s["verdict"] == "EMPTY")
    return {
        "match_count": n_match,
        "wrong_face_count": n_wrong,
        "empty_count": n_empty,
        "verdict": "SOLVED" if n_match == 9 else "NOT_SOLVED",
    }


# ---------- routes ----------

@app.get("/")
def index():
    return render_template_string(INDEX_HTML, puzzles=discover_puzzles())


@app.get("/calibrate")
def calibrate_page():
    return render_template_string(CALIBRATE_HTML)


@app.get("/stream")
def stream():
    if not CAMERA.ensure_started():
        return Response(f"camera unavailable: {CAMERA.error}", status=503)
    return Response(_mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/snap")
def snap():
    if not CAMERA.ensure_started():
        return Response(f"camera unavailable: {CAMERA.error}", status=503)
    frame = CAMERA.latest()
    if frame is None:
        return Response("no frame yet", status=503)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return Response("encode failed", status=500)
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.get("/camera/status")
def camera_status():
    available = CAMERA.ensure_started()
    w = h = None
    if available and CAMERA.cap is not None:
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


@app.get("/calibration")
def get_calibration():
    return jsonify(load_calibration() or {"boxes": []})


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
    save_calibration(out)
    return jsonify(ok=True, saved_at=out["saved_at"])


@app.post("/check")
def check():
    """Capture latest camera frame, crop the 9 saved ROIs, match each against
    the selected puzzle's reference cells, return per-slot verdicts."""
    t0 = time.perf_counter()

    puzzle_id = request.form.get("puzzle", "").strip()
    if not puzzle_id:
        return jsonify(ok=False, error="puzzle is required"), 400
    reference = load_reference(puzzle_id)
    if reference is None:
        return jsonify(ok=False, error=f"reference for '{puzzle_id}' not found"), 404

    cal = load_calibration()
    if not cal or not cal.get("boxes"):
        return jsonify(
            ok=False,
            error="No calibration saved. Draw the 9 ROIs first."
        ), 400

    if not CAMERA.ensure_started():
        return jsonify(ok=False, error=f"camera unavailable: {CAMERA.error}"), 503
    photo = CAMERA.latest()
    if photo is None:
        return jsonify(ok=False, error="no frame from camera yet"), 503

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
    slots = score_slots(photo_slots, ref_cell_features)
    summary = summarise(slots)

    overlay = draw_overlay(photo, quads, slots)
    run_id = uuid.uuid4().hex[:10]
    overlay_path = UPLOADS_DIR / f"{run_id}_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return jsonify({
        "ok": True,
        "puzzle": puzzle_id,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "slots": slots,
        **summary,
        "overlay_url": f"/uploads/{overlay_path.name}",
    })


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

_SHARED_CSS = r"""
  :root {
    --bg:#14110d; --panel:#1f1b14; --panel-2:#2a241a;
    --ink:#f3e6c8; --muted:#a89671; --gold:#d8a84b;
    --ok:#4ac779; --warn:#f0a531; --bad:#e15252;
    --border:#3a3122;
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:20px 28px; font:15px/1.45 system-ui, sans-serif; background:var(--bg); color:var(--ink); }
  .topbar { display:flex; justify-content:space-between; align-items:center; margin-bottom:18px; }
  h1 { font-size:22px; color:var(--gold); margin:0; }
  .subtle-link { color:var(--muted); text-decoration:none; font-size:13px; padding:6px 12px; border:1px solid var(--border); border-radius:6px; }
  .subtle-link:hover { color:var(--gold); border-color:var(--gold); }
  .wrap { max-width:1600px; margin:0 auto; }
  .panel { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:18px; }
  .panel.tight { display:flex; flex-direction:column; }
  label { display:block; font-size:11px; color:var(--muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:0.6px; }
  select { background:var(--panel-2); color:var(--ink); border:1px solid var(--border); padding:9px 12px; border-radius:8px; font-size:14px; min-width:140px; }
  button { background:var(--gold); color:#1a1408; border:0; border-radius:6px; padding:9px 16px; font-weight:600; font-size:14px; cursor:pointer; white-space:nowrap; }
  button.secondary { background:var(--panel-2); color:var(--ink); border:1px solid var(--border); }
  button:disabled { opacity:0.5; cursor:not-allowed; }
  .canvas-wrap { position:relative; width:100%; border:1px solid var(--border); border-radius:8px; overflow:hidden; background:#000; flex:1; }
  .canvas-wrap img, .canvas-wrap canvas { display:block; width:100%; height:auto; }
  .canvas-wrap canvas { position:absolute; top:0; left:0; }
  .canvas-wrap canvas.draw-mode { cursor:crosshair; }
  .two-col { display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:stretch; margin-bottom:16px; }
  @media (max-width:1100px) { .two-col { grid-template-columns:1fr; } }
  .hint { color:var(--muted); font-size:12px; margin-top:8px; }
  .verdict { font-size:28px; font-weight:700; padding:16px 20px; border-radius:10px; text-align:center; letter-spacing:1px; margin-bottom:12px; }
  .verdict.solved { background:rgba(74,199,121,0.15); color:var(--ok); border:1px solid rgba(74,199,121,0.5); }
  .verdict.notsolved { background:rgba(225,82,82,0.12); color:var(--bad); border:1px solid rgba(225,82,82,0.5); }
  .verdict.error { background:rgba(240,165,49,0.10); color:var(--warn); border:1px solid rgba(240,165,49,0.4); font-size:16px; line-height:1.5; }
  .verdict.placeholder { background:var(--panel-2); color:var(--muted); border:1px dashed var(--border); font-size:18px; font-weight:500; letter-spacing:0.4px; }
  .toolbar-group { display:inline-flex; gap:6px; padding:4px; background:var(--panel-2); border:1px solid var(--border); border-radius:8px; }
  .toolbar-group button { padding:7px 12px; font-size:13px; }
  .ref-frame { width:100%; aspect-ratio:1/1; background:#000; border:1px solid var(--border); border-radius:8px; overflow:hidden; display:flex; align-items:center; justify-content:center; }
  .ref-frame img { max-width:100%; max-height:100%; display:block; }
"""


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenPharaon</title>
<style>""" + _SHARED_CSS + r"""</style>
</head><body>
<div class="wrap">
  <div class="topbar">
    <h1>OpenPharaon</h1>
    <a class="subtle-link" href="/calibrate">⚙ Calibrate</a>
  </div>

  <div class="panel" style="margin-bottom:16px;">
    <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
      <div>
        <label for="puzzle">Puzzle</label>
        <select id="puzzle">
          {% for p in puzzles %}<option value="{{ p.id }}">{{ p.id }}</option>{% endfor %}
        </select>
      </div>
      <div style="flex:1;"></div>
      <button id="checkBtn">Check now</button>
      <button id="liveBtn" class="secondary" style="display:none;">Back to live preview</button>
      <span id="status" class="hint" style="margin:0;"></span>
    </div>
  </div>

  <div class="two-col">
    <!-- LEFT: reference image for the chosen puzzle -->
    <div class="panel tight">
      <label>Reference (what a SOLVED puzzle looks like)</label>
      <div class="ref-frame" id="refFrame">
        <img id="refImg" alt="reference">
      </div>
    </div>

    <!-- RIGHT: live cam feed (or analyzed photo after check) -->
    <div class="panel tight">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <label id="rightLabel" style="margin:0;">Live camera (saved ROIs overlaid)</label>
        <span id="elapsed" class="hint" style="margin:0;"></span>
      </div>
      <div id="verdict" class="verdict placeholder" style="margin-top:10px;">Ready — press Check now</div>
      <div class="canvas-wrap" style="margin-top:0;">
        <img id="feed" src="/stream" alt="feed">
        <canvas id="overlay"></canvas>
      </div>
    </div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const feed = $("feed"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const refImg = $("refImg"), puzzleSel = $("puzzle"), statusEl = $("status");

let savedBoxes = [];
let imgNaturalSize = null;
let liveMode = true;

function refreshRef() {
  refImg.src = "/tests/" + encodeURIComponent(puzzleSel.value) + "/ref.png";
}
puzzleSel.addEventListener("change", refreshRef);
refreshRef();

function drawSavedBoxes() {
  if (!imgNaturalSize) return;
  canvas.width = imgNaturalSize.w;
  canvas.height = imgNaturalSize.h;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!liveMode || !savedBoxes || savedBoxes.length !== 9) return;
  ctx.font = "bold 22px sans-serif";
  savedBoxes.forEach((b, i) => {
    ctx.strokeStyle = "#4ac779";
    ctx.lineWidth = 4;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    const r = i / 3 | 0, c = i % 3;
    ctx.fillRect(b.x1 + 4, b.y1 + 4, 70, 28);
    ctx.fillStyle = "#4ac779";
    ctx.fillText(`r${r}c${c}`, b.x1 + 8, b.y1 + 26);
  });
}

feed.addEventListener("load", () => {
  imgNaturalSize = { w: feed.naturalWidth, h: feed.naturalHeight };
  drawSavedBoxes();
});

(async () => {
  try {
    const r = await fetch("/calibration");
    const d = await r.json();
    if (d.boxes && d.boxes.length === 9) {
      savedBoxes = d.boxes;
      drawSavedBoxes();
    } else {
      statusEl.innerHTML = 'No calibration yet — <a href="/calibrate" style="color:var(--gold)">click here to draw the 9 ROIs</a>.';
    }
  } catch (e) {}
})();

$("checkBtn").onclick = async () => {
  $("checkBtn").disabled = true;
  statusEl.textContent = "Capturing & matching…";
  $("verdict").className = "verdict placeholder";
  $("verdict").textContent = "Running…";
  const fd = new FormData();
  fd.append("puzzle", puzzleSel.value);
  try {
    const res = await fetch("/check", { method: "POST", body: fd });
    const d = await res.json();
    renderResult(d);
  } catch (e) {
    renderError("Network error: " + e.message);
  } finally {
    $("checkBtn").disabled = false;
    statusEl.textContent = "";
  }
};

$("liveBtn").onclick = () => {
  liveMode = true;
  feed.src = "/stream";
  $("rightLabel").textContent = "Live camera (saved ROIs overlaid)";
  $("verdict").className = "verdict placeholder";
  $("verdict").textContent = "Ready — press Check now";
  $("elapsed").textContent = "";
  $("liveBtn").style.display = "none";
  drawSavedBoxes();
};

function renderResult(d) {
  const v = $("verdict");
  if (!d.ok) { renderError(d.error || "error"); return; }
  liveMode = false;
  feed.src = d.overlay_url + "?ts=" + Date.now();
  $("rightLabel").textContent = "Captured frame with per-slot result";
  v.className = "verdict " + (d.verdict === "SOLVED" ? "solved" : "notsolved");
  v.textContent = d.verdict;
  $("elapsed").textContent = `${d.elapsed_ms} ms`;
  $("liveBtn").style.display = "";
  // hide the canvas overlay; the result image already has its own coloured quads
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function renderError(m) {
  const v = $("verdict");
  v.className = "verdict error"; v.textContent = m;
  $("elapsed").textContent = "";
}
</script>
</body></html>
"""


CALIBRATE_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenPharaon — Calibrate</title>
<style>""" + _SHARED_CSS + r"""</style>
</head><body>
<div class="wrap">
  <div class="topbar">
    <h1>OpenPharaon — Calibrate</h1>
    <a class="subtle-link" href="/">← back to check page</a>
  </div>

  <div class="panel">
    <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;">
      <div>
        <label style="margin:0">Draw the 9 cube ROIs</label>
        <div class="hint" style="margin-top:2px;">Camera is locked. Same 9 boxes work for every puzzle.</div>
      </div>
      <div class="toolbar-group">
        <button id="snapBtn">Snap & Calibrate</button>
        <button id="resumeBtn" class="secondary">Resume live</button>
        <button id="clearBtn" class="secondary">Clear</button>
        <button id="saveBtn" class="secondary" disabled>Save</button>
      </div>
    </div>
    <div class="hint" id="boxCounter" style="margin-top:10px;">Boxes drawn: 0 / 9 — drag to draw, row-major (r0c0, r0c1, r0c2, r1c0, …)</div>
    <div class="hint" id="status"></div>

    <div class="canvas-wrap" style="margin-top:12px;">
      <img id="bg" src="/stream" alt="camera">
      <canvas id="overlay" class="draw-mode"></canvas>
    </div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const bg = $("bg"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const statusEl = $("status"), counterEl = $("boxCounter");
let boxes = [];
let dragging = false, startPt = null, drawingEnabled = false;
let imgNaturalSize = null;

function fitCanvasToImage() {
  if (!imgNaturalSize) return;
  canvas.width = imgNaturalSize.w;
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
    ctx.fillRect(b.x1 + 4, b.y1 + 4, 70, 28);
    ctx.fillStyle = "#4ac779";
    ctx.fillText(`r${r}c${c}`, b.x1 + 8, b.y1 + 26);
  });
  if (preview) {
    ctx.strokeStyle = "#d8a84b"; ctx.lineWidth = 3;
    ctx.setLineDash([10, 6]);
    ctx.strokeRect(preview.x1, preview.y1, preview.x2 - preview.x1, preview.y2 - preview.y1);
    ctx.setLineDash([]);
  }
  counterEl.textContent = `Boxes drawn: ${boxes.length} / 9 — drag to draw next, row-major (r0c0, r0c1, r0c2, r1c0, …)`;
  $("saveBtn").disabled = boxes.length !== 9;
}

canvas.addEventListener("mousedown", (ev) => {
  if (!drawingEnabled || boxes.length >= 9) return;
  dragging = true; startPt = clientToImage(ev);
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
  boxes.push(b); redraw();
});

$("snapBtn").onclick = () => {
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
  statusEl.textContent = data.ok
    ? `Calibration saved (${data.saved_at}). You can now return to the check page.`
    : "Save failed: " + (data.error || "unknown");
};

(async () => {
  try {
    const r = await fetch("/calibration");
    const d = await r.json();
    if (d.boxes && d.boxes.length === 9) {
      boxes = d.boxes;
      statusEl.textContent = "Loaded saved calibration (" + (d.saved_at || "") + "). Drag-draw will replace them.";
      if (imgNaturalSize) redraw();
    } else {
      statusEl.textContent = "No calibration yet. Click Snap & Calibrate to draw 9 ROIs.";
    }
  } catch (e) {}
})();
</script>
</body></html>
"""


if __name__ == "__main__":
    print(f"Discovered puzzles: {[p['id'] for p in discover_puzzles()]}")
    cal = load_calibration()
    print(f"Calibration: {'loaded (' + str(len(cal['boxes'])) + ' boxes)' if cal and cal.get('boxes') else 'none'}")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

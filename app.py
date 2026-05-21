"""
OpenPharaon — single-file Flask web UI for testing the puzzle matcher.

Pick a puzzle, upload or capture a photo, get a SOLVED/NOT_SOLVED verdict
with a per-slot breakdown and an alignment overlay.

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template_string, request, send_from_directory

from homography_align import (
    draw_quads,
    find_homography,
    project_cells,
    warp_quad,
)
from pharaon_cv import orb_ransac_inliers, verdict_for
from puzzles_batch_h import split_reference  # canonical splitter (rounded float)

ROOT = Path(__file__).parent
TESTS_DIR = ROOT / "tests"
UPLOADS_DIR = ROOT / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

ORB_INLIERS_MIN = 5          # per-slot match threshold (calibrated on 9 stress-test solved photos)
H_INLIERS_MIN_RECOGNIZE = 8  # minimum homography inliers to claim "this is puzzle X"

app = Flask(__name__)

# Cache all reference images on first access so we don't re-read from disk on every request.
_REF_CACHE: dict[str, np.ndarray] = {}


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

def _score_slots(photo_slots: list[np.ndarray], ref_cells: list[np.ndarray]) -> list[dict]:
    """Run the per-slot ORB-RANSAC matcher and return verdict records."""
    results = []
    for i, slot in enumerate(photo_slots):
        per_ref = [orb_ransac_inliers(slot, rc) for rc in ref_cells]
        best_idx = int(np.argmax(per_ref))
        verdict = verdict_for(
            per_ref[i], per_ref[best_idx], best_idx, i, {"orb_inliers_min": ORB_INLIERS_MIN}
        )
        results.append({
            "row": i // 3,
            "col": i % 3,
            "expected_inliers": int(per_ref[i]),
            "best_inliers": int(per_ref[best_idx]),
            "best_match_index": best_idx,
            "best_match_label": f"r{best_idx // 3}c{best_idx % 3}",
            "verdict": verdict,
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


def run_method_a(photo: np.ndarray, reference: np.ndarray) -> dict:
    """Method A: homography-based reference→photo alignment, then per-slot match."""
    H, h_inliers, h_good = find_homography(photo, reference)
    if H is None or h_inliers < 8:
        return {
            "ok": False,
            "method": "A",
            "name": "Homography (reference-aligned)",
            "error": "Could not align photo to reference. Wrong puzzle or unclear photo.",
            "h_inliers": h_inliers,
            "h_good_matches": h_good,
        }
    quads = project_cells(H, reference)
    ref_cells = split_reference(reference)
    photo_slots = [warp_quad(photo, q, out_size=256) for q in quads]
    slots = _score_slots(photo_slots, ref_cells)
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

    # Per-slot match against the SELECTED puzzle's reference
    ref_cells = split_reference(selected_reference)
    photo_slots = [warp_quad(photo, q, out_size=256) for q in quads]
    slots = _score_slots(photo_slots, ref_cells)
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


@app.post("/check")
def check():
    t0 = time.perf_counter()
    puzzle_id = request.form.get("puzzle", "").strip()
    if not puzzle_id:
        return jsonify(ok=False, error="puzzle is required"), 400

    ref_path = TESTS_DIR / puzzle_id / "ref.png"
    if not ref_path.exists():
        return jsonify(ok=False, error=f"reference for '{puzzle_id}' not found"), 404
    reference = cv2.imread(str(ref_path))
    if reference is None:
        return jsonify(ok=False, error="failed to read reference image"), 500

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

    # By default, only run Method A (the production path). Method B is a
    # diagnostic; users can opt in via methods=ab.
    methods = request.form.get("methods", "a").lower()
    run_b = "b" in methods

    run_id = uuid.uuid4().hex[:10]
    method_a = run_method_a(photo, reference)
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
    return jsonify(response)


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
    .samples { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    .sample-chip {
      background: var(--panel-2); padding: 6px 10px; border-radius: 6px;
      border: 1px solid var(--border); cursor: pointer; font-size: 13px;
    }
    .sample-chip:hover { border-color: var(--gold); }
    .sample-chip.active { background: var(--gold); color: #1a1408; border-color: var(--gold); }

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

        <label style="margin-top: 14px;">Or pick a bundled sample</label>
        <div id="samples" class="samples"></div>
      </div>
    </div>

    <div style="margin-top: 18px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
      <button id="run">Check puzzle</button>
      <button id="reset" class="secondary">Clear</button>
      <label style="display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font-size: 13px; margin-bottom: 0; text-transform: none; letter-spacing: 0;">
        <input type="checkbox" id="debugMode"> Also run Method B (diagnostic, ~6&times; slower)
      </label>
      <span id="status" class="hint"></span>
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
const samplesDiv = $("samples");
const refPreview = $("refPreview");
const filePreview = $("filePreview");
const statusEl = $("status");
const resultPanel = $("resultPanel");
const topStatsEl = $("topStats");
const methodsRow = $("methodsRow");

let selectedSample = null;

function refreshPuzzleUI() {
  const opt = puzzleSel.selectedOptions[0];
  const id = opt.value;
  refPreview.innerHTML = `<img src="/tests/${encodeURIComponent(id)}/${opt.dataset.ref}" alt="reference">`;
  const samples = JSON.parse(opt.dataset.samples || "[]");
  samplesDiv.innerHTML = "";
  samples.forEach(s => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "sample-chip";
    b.textContent = s;
    b.onclick = () => {
      photoInput.value = "";
      filePreview.innerHTML = "";
      selectedSample = s;
      document.querySelectorAll(".sample-chip").forEach(el => el.classList.remove("active"));
      b.classList.add("active");
    };
    samplesDiv.appendChild(b);
  });
  selectedSample = null;
}

photoInput.addEventListener("change", () => {
  filePreview.innerHTML = "";
  document.querySelectorAll(".sample-chip").forEach(el => el.classList.remove("active"));
  selectedSample = null;
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
  selectedSample = null;
  document.querySelectorAll(".sample-chip").forEach(el => el.classList.remove("active"));
  resultPanel.classList.add("hidden");
  statusEl.textContent = "";
};

$("run").onclick = async () => {
  const f = photoInput.files[0];
  if (!f && !selectedSample) {
    statusEl.textContent = "Pick a photo or a sample first.";
    return;
  }
  statusEl.innerHTML = '<span class="spinner"></span>Running matcher…';
  $("run").disabled = true;
  resultPanel.classList.add("hidden");

  const fd = new FormData();
  fd.append("puzzle", puzzleSel.value);
  if (f) fd.append("photo", f);
  if (selectedSample) fd.append("sample", selectedSample);
  if ($("debugMode").checked) fd.append("methods", "ab");

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


if __name__ == "__main__":
    print(f"Discovered puzzles: {[p['id'] for p in discover_puzzles()]}")
    app.run(host="127.0.0.1", port=5000, debug=False)

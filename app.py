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
from frame_detect import detect_cube_quads

ROOT = Path(__file__).parent
TESTS_DIR = ROOT / "tests"
UPLOADS_DIR = ROOT / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

ORB_INLIERS_MIN = 10  # per-slot match threshold

app = Flask(__name__)


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


def run_method_b(photo: np.ndarray, reference: np.ndarray) -> dict:
    """Method B: detect 9 cube quads directly in the photo (no reference needed
    for localization), then match each cube against the reference's 9 cells."""
    quads, info = detect_cube_quads(photo)
    if quads is None:
        return {
            "ok": False,
            "method": "B",
            "name": "Frame detect (markerless grid)",
            "error": f"Frame detection failed: {info.get('fail_reason', 'unknown')}",
            "detector_info": info,
        }
    ref_cells = split_reference(reference)
    photo_slots = [warp_quad(photo, q, out_size=256) for q in quads]
    slots = _score_slots(photo_slots, ref_cells)
    overlay = _draw_verdict_overlay(photo, quads, slots)
    return {
        "ok": True,
        "method": "B",
        "name": "Frame detect (markerless grid)",
        "detector_info": info,
        "slots": slots,
        **_summarise(slots),
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

    run_id = uuid.uuid4().hex[:10]
    method_a = run_method_a(photo, reference)
    method_b = run_method_b(photo, reference)

    for tag, result in (("a", method_a), ("b", method_b)):
        if result.get("ok"):
            overlay_path = UPLOADS_DIR / f"{run_id}_{tag}_overlay.jpg"
            cv2.imwrite(str(overlay_path), result.pop("_overlay"), [cv2.IMWRITE_JPEG_QUALITY, 85])
            result["overlay_url"] = f"/uploads/{overlay_path.name}"

    return jsonify({
        "ok": True,
        "puzzle": puzzle_id,
        "photo_label": photo_label,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "method_a": method_a,
        "method_b": method_b,
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

    <div style="margin-top: 18px; display: flex; gap: 10px; align-items: center;">
      <button id="run">Check puzzle</button>
      <button id="reset" class="secondary">Clear</button>
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
  methodsRow.appendChild(renderMethodCard(d.method_b));
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
    const di = m.detector_info || {};
    extra = `<span>Detector: cols=${JSON.stringify(di.col_xs || [])} rows=${JSON.stringify(di.row_ys || [])}</span>`;
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

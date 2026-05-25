# OpenPharaon — Locked-camera puzzle verifier

Detects whether a 9-cube Egyptian Pharaon puzzle is solved using a fixed USB
camera and OpenCV. Replaces an RFID-based detection that was unreliable due
to cards falling inside the cubes.

## How it works

The camera does not move. At install time, an operator draws 9 rectangles
once over the cube grid in the live preview ("Snap & Calibrate" on `/`).
Every check after that:

1. Grabs the latest camera frame.
2. Crops the 9 saved pixel rectangles.
3. Matches each crop against the selected puzzle's 9 reference cells using
   ORB + RANSAC homography (USAC_FAST when available).
4. Returns a per-slot verdict (`MATCH` / `WRONG_FACE` / `EMPTY`) and an
   overall `SOLVED` / `NOT_SOLVED`.

No homography auto-alignment, no image-content guessing — the ROIs are
fixed by the operator. If the operator hasn't calibrated yet, the page
prompts for it and disables the check button.

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask web app: UI + camera + calibration + check endpoint |
| `pharaon_cv.py` | Core ORB + RANSAC primitives + verdict rules |
| `tests/puzzle N/ref.png` | Reference image for each of the 6 puzzles |
| `calibration.json` | 9 ROI boxes for this installation (gitignored) |
| `uploads/` | Generated debug overlays per check (gitignored) |

## Endpoints

| Method + path | Purpose |
|---|---|
| `GET /` | Single-page UI: live preview, calibration canvas, check panel |
| `GET /stream` | MJPEG live camera feed |
| `GET /snap` | Single still JPEG from the camera (used by the calibration canvas) |
| `GET /camera/status` | JSON: availability, error, frame size, read counts |
| `GET /calibration` | Load saved 9 boxes |
| `POST /calibration` | Save 9 boxes from the UI |
| `POST /check` | Capture latest frame, crop the 9 saved boxes, return verdict |

## Run locally (Windows / dev)

```powershell
python -m pip install opencv-contrib-python numpy flask
python app.py
```

Open `http://127.0.0.1:5000`.

## Deploy on Raspberry Pi

```bash
sudo apt install -y git python3-pip python3-venv libatlas-base-dev libgl1
git clone <repo>
cd OpenPharaon
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install --upgrade pip
pip install 'flask>=2.3'
sudo usermod -aG video $USER     # logout + login required
python app.py
```

The systemd unit (suggested):

```ini
[Unit]
Description=OpenPharaon puzzle verifier
After=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/OpenPharaon
ExecStart=/home/pi/OpenPharaon/.venv/bin/python /home/pi/OpenPharaon/app.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## Tunable thresholds

In `app.py`:

| Constant | Default | Meaning |
|---|---|---|
| `ORB_INLIERS_MIN` | `3` | Per-slot inlier floor for a MATCH to register |
| `CLEAR_MATCH_FLOOR` | `10` | A slot at this many inliers (and diagonal-best) counts as "clearly correct" |
| `CLEAR_MATCH_COUNT_FOR_LENIENT` | `6` | Need this many clear matches to enable the lenient tied-MATCH rule |

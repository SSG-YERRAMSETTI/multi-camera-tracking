# 📘 Complete Setup & Run Guide
## Multi-Camera Live Object Tracking v3.0
### VS Code Edition

---

## ✅ System Requirements

| Item | Requirement |
|------|-------------|
| **Python** | **3.10.x** (recommended) — also works on 3.9 / 3.11 |
| **OS** | Windows 10/11, macOS 12+, Ubuntu 20.04+ |
| **RAM** | Minimum 8 GB (16 GB recommended for 2 cameras) |
| **GPU** | Optional — NVIDIA GPU with CUDA gives 3–5× speedup |
| **VS Code** | Latest version from https://code.visualstudio.com |
| **Git** | Optional — for cloning original repo |
| **Webcam** | At least 1 (USB webcam or built-in) |

> ⚠️ **DO NOT use Python 3.12+** — PyTorch and some OpenCV wheels have not fully caught up yet.
> Use Python 3.10.x for best compatibility.

---

## STEP 1 — Install Python 3.10

### Windows
1. Go to https://www.python.org/downloads/release/python-31011/
2. Download **Windows installer (64-bit)**
3. Run installer — check ✅ **"Add Python to PATH"** before clicking Install
4. Verify: open Command Prompt and type:
   ```
   python --version
   ```
   You should see: `Python 3.10.x`

### macOS
```bash
brew install python@3.10
```

### Ubuntu / Linux
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-pip
```

---

## STEP 2 — Install VS Code + Extensions

1. Download VS Code from https://code.visualstudio.com
2. Install it with default options
3. Open VS Code
4. Press `Ctrl+Shift+X` (Extensions panel)
5. Install these extensions one by one (search by name):
   - **Python** (by Microsoft) ← most important
   - **Pylance** (by Microsoft)
   - **Thunder Client** (for testing API endpoints)
   - **SQLite Viewer** (to browse the database)

> 💡 When you open the project folder, VS Code will auto-suggest installing
> all recommended extensions from `.vscode/extensions.json` — click **Install All**.

---

## STEP 3 — Open the Project in VS Code

1. Unzip the downloaded file `Multi_Camera_Tracking_v3.zip`
2. Open VS Code
3. Click **File → Open Folder**
4. Select the unzipped folder `Multi_Camera_Tracking_v3`
5. VS Code will open the project

---

## STEP 4 — Create a Virtual Environment

Open the **VS Code Terminal** (`Ctrl+` ` ` or Terminal → New Terminal)

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```
You should see `(venv)` appear at the start of your terminal line.

### macOS / Linux
```bash
python3.10 -m venv venv
source venv/bin/activate
```

> 💡 VS Code will detect the venv and ask "Do you want to use this environment?"
> Click **Yes**. The Python interpreter in the bottom-left will switch to `venv`.

---

## STEP 5 — Install All Libraries

With your venv activated, run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install everything: FastAPI, YOLOv8, Deep SORT, OpenCV, SQLite, etc.

**Expected time:** 5–15 minutes (downloads ~500 MB of PyTorch + model files)

### If you have an NVIDIA GPU (optional but recommended):
After the above, run this to get GPU-accelerated PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Then open `config.json` and change `"device": "cpu"` to `"device": "cuda"`.

### Verify the install:
```bash
python -c "import ultralytics; import fastapi; import cv2; print('All libraries OK')"
```
You should see: `All libraries OK`

---

## STEP 6 — Verify Project Structure

Your folder should look exactly like this:

```
Multi_Camera_Tracking_v3/
│
├── .vscode/
│   ├── settings.json        ← VS Code project config
│   ├── launch.json          ← Run/Debug buttons
│   └── extensions.json      ← Recommended extensions
│
├── backend/
│   ├── __init__.py
│   ├── app.py               ← FastAPI server (main entry point)
│   ├── configuration.py     ← Settings Manager (Phase 3 NEW)
│   ├── database.py          ← SQLite database layer (Phase 3 NEW)
│   ├── yolo_detector.py     ← YOLOv8 inference engine
│   ├── tracker.py           ← Deep SORT tracker
│   ├── traffic_counter.py   ← Line-crossing counter
│   ├── analytics.py         ← Real-time analytics (Phase 3 NEW)
│   ├── camera_manager.py    ← Multi-camera pipeline
│   └── status_monitor.py    ← Health monitor (Phase 3 NEW)
│
├── frontend/
│   └── dashboard.html       ← Full control dashboard (served at /)
│
├── logs/
│   ├── counts/              ← CSV logs written here
│   └── objects/             ← TXT logs written here
│
├── models/
│   └── README.md            ← Model download instructions
│
├── .vscode/                 ← VS Code debug configs
├── camera_client.py         ← Run on camera device
├── config.json              ← All settings (edit this)
├── requirements.txt         ← All Python dependencies
├── SETUP_GUIDE.md           ← This file
└── README.md                ← Quick reference
```

---

## STEP 7 — Start the Server

### Option A — Using VS Code Run Button (Easiest)
1. Press `F5` or click the **Run** icon in the sidebar
2. Select **"▶ Run Server (app.py)"** from the dropdown
3. The server starts in the Debug Console

### Option B — Using the Terminal
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### Option C — Direct Python
```bash
python backend/app.py
```

**Expected output:**
```
[App] Starting up...
[YOLODetector] Loading model: yolov8n.pt on cpu
[App] Ready — visit http://localhost:8000
INFO:     Uvicorn running on http://0.0.0.0:8000
```

> 💡 First run downloads YOLOv8n (~6 MB). This happens automatically.

---

## STEP 8 — Open the Dashboard

Open your browser and go to:
```
http://localhost:8000
```

You will see the **Multi-Camera Tracking Dashboard**. It will show in demo mode
until a real camera connects.

---

## STEP 9 — Connect a Camera

Open a **second terminal** in VS Code (`+` icon in terminal panel), activate venv again, then:

### If using a webcam:
```bash
python camera_client.py --camera 0 --server tcp://localhost:5555
```

### If using a video file for testing:
```bash
python camera_client.py --camera test_video.mp4 --server tcp://localhost:5555
```

### If using an IP camera (RTSP):
```bash
python camera_client.py --camera rtsp://192.168.1.10/stream --server tcp://localhost:5555
```

### If connecting a second camera:
Open a **third terminal** and run:
```bash
python camera_client.py --camera 1 --server tcp://localhost:5566 --name camera_1
```

> 💡 You can also use VS Code's **"📷 Camera Client 0 (Webcam)"** launch config from F5 menu.

---

## STEP 10 — Using the Dashboard

| Control | What it does |
|---------|-------------|
| **▶ Start / ■ Stop** | Turns YOLOv8 detection on or off live |
| **Mode: Objects / Traffic** | Switches between object counting and traffic counting |
| **Classes checkboxes** | Choose which objects to detect (Car, Truck, Bus, Person) |
| **Confidence slider** | Adjust detection sensitivity (0.0 = detect everything, 1.0 = very strict) |
| **Counting line** | Drag to set where vehicles are counted (traffic mode only) |
| **↓ CSV** | Download traffic counting log |
| **↓ TXT** | Download object counting log |
| **↺ Refresh config** | Reload current settings from server |

---

## API Endpoints (for testing with Thunder Client in VS Code)

```
GET  http://localhost:8000/api/status
GET  http://localhost:8000/api/analytics
GET  http://localhost:8000/api/config

POST http://localhost:8000/api/set_yolo_mode
     Body: {"enabled": true}

POST http://localhost:8000/api/set_confidence
     Body: {"threshold": 0.5}

POST http://localhost:8000/api/set_classes
     Body: {"classes": ["car", "truck", "bus", "person"]}

POST http://localhost:8000/api/set_mode
     Body: {"mode": "traffic_counting"}

POST http://localhost:8000/api/set_counting_line
     Body: {"position": 0.5}

GET  http://localhost:8000/api/logs/traffic.csv
GET  http://localhost:8000/api/logs/objects.txt
```

Auto-generated API docs:
```
http://localhost:8000/docs
```

---

## Common Errors & Fixes

### Error: `ModuleNotFoundError: No module named 'ultralytics'`
**Fix:** Your venv is not activated.
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Then install again
pip install -r requirements.txt
```

### Error: `Address already in use` / Port 8000 taken
**Fix:**
```bash
# Windows — find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

### Error: Camera not opening (`Cannot open source: 0`)
**Fix:** Check your webcam is connected and not in use by another app (Zoom, Teams, etc.)
Try changing `--camera 0` to `--camera 1` (some laptops number cameras differently).

### Error: `CUDA out of memory`
**Fix:** Switch back to CPU mode. Edit `config.json`:
```json
"device": "cpu"
```

### Error: `deep_sort_realtime` import error
**Fix:**
```bash
pip install deep-sort-realtime --upgrade
```

### Dashboard shows "demo mode" but server is running
**Fix:** Make sure you run `camera_client.py` in a separate terminal.
The dashboard needs a camera client sending frames.

### VS Code doesn't show the venv Python
**Fix:**
1. Press `Ctrl+Shift+P`
2. Type `Python: Select Interpreter`
3. Choose the one that says `./venv/Scripts/python.exe` (Windows) or `./venv/bin/python` (Mac/Linux)

---

## What Was Upgraded from the Original GitHub Project

| Original (Phase 1) | Upgraded (Phase 3) |
|---|---|
| YOLOv4 via Darknet + Keras | **YOLOv8** via Ultralytics |
| TensorFlow 1.14 | **PyTorch 2.2** |
| Keras 2.3.1 | Removed — not needed |
| Flask 1.1.1 | **FastAPI 0.111** |
| Raw MJPEG streams | **WebSocket + base64 frames** |
| No control API | **REST API** — change settings live |
| .txt and .csv flat files | **SQLite database** |
| No analytics | **Real-time analytics module** |
| No error handling | **Status monitor + auto-reconnect** |
| Two separate apps | **One unified app** with mode toggle |
| Settings hardcoded | **config.json** + hot-reload |
| Old dashboard (2 boxes) | **Full control room dashboard** |

---

## Quick Start Cheatsheet

```bash
# 1. Activate venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # Mac/Linux

# 2. Start server
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# 3. Open dashboard
# Browser → http://localhost:8000

# 4. Connect camera (new terminal, activate venv first)
python camera_client.py --camera 0 --server tcp://localhost:5555
```

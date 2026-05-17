<div align="center">

# Multi-Camera Live Object Tracking — v3.0

**Real-time vehicle detection, tracking, and intelligence across multiple camera feeds**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Deep SORT](https://img.shields.io/badge/Deep_SORT-Realtime-8A2BE2?style=flat-square)](https://github.com/levan92/deep_sort_realtime)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

> The original system worked — but setting it up meant compiling Darknet from source, wrestling with TensorFlow 1.14, and hoping your GPU drivers cooperated. v3.0 replaces all of that with a pure-Python stack you can install with a single pip command.

This is a real-time multi-camera tracking system. Point any number of cameras at a scene and it detects vehicles, assigns each one a persistent ID, counts how many cross a virtual line, and optionally tells you the color, type, and license plate of each vehicle — all streaming live to a browser dashboard.

---

## What It Does

**Multi-Camera Object Detection**
Each camera feed runs through YOLOv8 independently. Frames are sent from camera devices to the backend over ZMQ, so your cameras can be anywhere on the network — not just attached to the server. Detection runs at configurable confidence thresholds and you can switch which object classes you're tracking (car, truck, bus, person) from the dashboard without restarting anything.

**Persistent Vehicle Tracking with Deep SORT**
Every detected vehicle gets a unique track ID that stays with it across frames — even when it temporarily leaves the frame or gets occluded. Each camera gets its own Deep SORT instance, which fixes a concurrency bug in the original system where a shared tracker caused ID collisions across multiple feeds.

**Traffic Counting**
A virtual counting line sits across the frame. When a vehicle's tracked path crosses it, the system logs the crossing — direction, class, timestamp. The crossing detection uses a cross-product sign-change algorithm which is more robust than the angle math in the original. Each vehicle ID is counted exactly once, so restarts and re-identification gaps don't inflate the numbers.

**Vehicle Intelligence**
For each tracked vehicle, three things run in the background:
- **Color detection** — HSV color space matching against 12 color profiles
- **Type classification** — EfficientNet backbone mapping to ImageNet vehicle classes (ambulance, pickup, school bus, sports car, etc.)
- **License plate OCR** — EasyOCR with graceful fallback if not installed

**Live Dashboard + REST API**
Everything streams to a browser dashboard over WebSocket — live video frames, counts, track IDs, FPS, and recent events. All settings (confidence threshold, tracked classes, counting line position, YOLO on/off) are changeable at runtime through the dashboard or REST API. No restart required.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  camera_client.py  (runs on any device with a camera)         │
│  Captures frames → ZMQ → backend port 5555 / 5566             │
└───────────────────────┬───────────────────────────────────────┘
                        │  ZMQ (imagezmq)
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  camera_manager.py  — one processing thread per camera        │
│                                                               │
│  Frame → YOLODetector     (YOLOv8, thread-safe singleton)     │
│       → CameraTracker     (Deep SORT, per-camera instance)    │
│       → TrafficCounter    (line-crossing, cross-product)      │
│       → VehicleIntel      (color + OCR + type, async)         │
│       → AnalyticsModule   (thread-safe stats aggregator)      │
└───────────────────────┬───────────────────────────────────────┘
                        │  WebSocket broadcast (30 fps)
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  FastAPI server  (app.py)                                     │
│  Serves dashboard + REST API + WebSocket stream               │
└───────────────────────┬───────────────────────────────────────┘
                        │  HTTP + WebSocket
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  Browser Dashboard  (frontend/dashboard.html)                 │
│  Live video · Track IDs · Counts · Controls · Activity feed   │
└───────────────────────────────────────────────────────────────┘
```

---

## What Changed from v1

The original project (by [LeonLok](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking)) was a solid foundation but painful to set up and maintain. Here's the full picture of what v3.0 changed:

| Original | v3.0 |
|---|---|
| YOLOv4 via Darknet (C library, manual compilation) | **YOLOv8** via Ultralytics — pure Python, `pip install` |
| TensorFlow 1.14 + Keras 2.3 | **PyTorch 2.2** |
| Flask 1.1 + MJPEG streams | **FastAPI 0.111 + WebSocket** |
| Hardcoded settings, restart required | **Hot-reload config** — change anything live |
| Flat `.txt` / `.csv` logs | **SQLite** with structured tables |
| No analytics layer | **Real-time analytics** — FPS, counts, IDs, events |
| No error recovery | **Status monitor** with 5-second auto-reconnect |
| Shared YOLO instance (concurrency bug) | **Per-camera tracker instances** |
| No vehicle analysis | **Vehicle Intelligence** — color, type, plate OCR |

---

## Project Structure

```
multi-camera-tracking/
│
├── backend/
│   ├── app.py                 ← FastAPI server + REST endpoints + WebSocket
│   ├── yolo_detector.py       ← YOLOv8 thread-safe singleton, FP16 support
│   ├── tracker.py             ← Deep SORT per-camera, track history
│   ├── traffic_counter.py     ← Line-crossing counter, cross-product method
│   ├── vehicle_intel.py       ← Color detection, OCR, type classifier
│   ├── analytics.py           ← Thread-safe stats aggregator
│   ├── camera_manager.py      ← Multi-camera pipeline threads
│   ├── configuration.py       ← Hot-reloadable settings (observer pattern)
│   ├── database.py            ← SQLite schema + async queries
│   └── status_monitor.py      ← Health monitor + auto-reconnect (5s)
│
├── frontend/
│   └── dashboard.html         ← Control room dashboard
│
├── models/
│   └── README.md              ← Model download instructions
│
├── logs/
│   ├── counts/                ← Traffic CSV logs
│   └── objects/               ← Object TXT logs
│
├── camera_client.py           ← Run on any camera device
├── config.json                ← All settings (edit live, no restart needed)
└── requirements.txt
```

---

## Getting Started

### What You'll Need

| Item | Requirement |
|------|-------------|
| **Python** | 3.10.x recommended (3.9–3.11 fine; avoid 3.12+) |
| **RAM** | 8 GB minimum, 16 GB recommended for 2+ cameras |
| **GPU** | Optional — NVIDIA with CUDA gives 3–5× speedup |
| **OS** | Windows 10/11, macOS 12+, Ubuntu 20.04+ |
| **Camera** | USB webcam, IP/RTSP stream, or a video file for testing |

> PyTorch and some OpenCV wheels haven't fully caught up to Python 3.12 yet. Stick with 3.10.x unless you know what you're doing.

---

### 1. Clone the Repository

```bash
git clone https://github.com/SSG-YERRAMSETTI/multi-camera-tracking.git
cd multi-camera-tracking
```

---

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This pulls in PyTorch, YOLOv8, Deep SORT, FastAPI, OpenCV, and everything else (~500 MB, 5–15 minutes).

**For NVIDIA GPU acceleration (optional but recommended):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Then set `"device": "cuda"` in `config.json`.

**Verify the install:**
```bash
python -c "import ultralytics; import fastapi; import cv2; print('All libraries OK')"
```

---

### 4. Start the Server

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000](http://localhost:8000) — the dashboard loads in demo mode until a camera connects.

---

### 5. Connect a Camera

Open a second terminal, activate the venv again, then:

```bash
# USB webcam
python camera_client.py --camera 0 --server tcp://localhost:5555

# Video file (great for testing without hardware)
python camera_client.py --camera test_video.mp4 --server tcp://localhost:5555

# IP / RTSP camera
python camera_client.py --camera rtsp://192.168.1.10/stream --server tcp://localhost:5555
```

**Adding a second camera** — open a third terminal:
```bash
python camera_client.py --camera 1 --server tcp://localhost:5566 --name camera_1
```

---

## Configuration

`config.json` controls everything and reloads live — no restarts needed:

```json
{
  "mode": "object_tracking",
  "detection": {
    "model_path": "yolov8n.pt",
    "confidence_threshold": 0.22,
    "classes_to_track": ["car", "truck", "bus"],
    "device": "cpu"
  },
  "tracking": {
    "max_age": 30,
    "min_hits": 3,
    "embedder": "mobilenet"
  },
  "traffic": {
    "counting_line_position": 0.5,
    "count_direction": "both"
  }
}
```

You can also change any setting through the dashboard controls or the REST API while the system is running.

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | System health + camera states |
| `GET` | `/api/analytics` | Live counts, FPS, track IDs |
| `GET` | `/api/config` | Current settings |
| `POST` | `/api/set_yolo_mode` | Toggle detection on/off |
| `POST` | `/api/set_confidence` | Adjust confidence threshold |
| `POST` | `/api/set_classes` | Change tracked object classes |
| `POST` | `/api/set_mode` | Switch object / traffic mode |
| `POST` | `/api/set_counting_line` | Move counting line |
| `GET` | `/api/logs/traffic.csv` | Download traffic log |
| `GET` | `/api/logs/objects.txt` | Download object log |
| `GET` | `/docs` | Interactive API docs (Swagger) |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'ultralytics'`**
Your venv isn't activated. Run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux), then `pip install -r requirements.txt`.

**Dashboard shows "demo mode" with the server running**
`camera_client.py` needs to be running in a separate terminal. The dashboard waits for a camera client to start sending frames.

**Port 8000 already in use**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

**`CUDA out of memory`**
Switch back to CPU: set `"device": "cpu"` in `config.json`.

**Camera not opening (`Cannot open source: 0`)**
Check that no other app (Zoom, Teams) is holding the webcam. Try `--camera 1` if `--camera 0` doesn't work — some laptops number cameras differently.

---

## Acknowledgements

Built on top of [LeonLok/Multi-Camera-Live-Object-Tracking](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking). The multi-camera ZMQ transport architecture and the original tracking concept come from that work. v3.0 modernizes the entire ML stack and adds the Vehicle Intelligence layer.

---

## Author

**Satya Sai Ganesh Yerramsetti**
MS Computer Science — University of North Texas

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/satya-sai-ganesh-yerramsetti-2a204424b)
[![GitHub](https://img.shields.io/badge/GitHub-SSG--YERRAMSETTI-181717?style=flat-square&logo=github)](https://github.com/SSG-YERRAMSETTI)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat-square&logo=gmail)](mailto:satyasaiganeshyerramsetti@my.unt.edu)

---

<div align="center">
  <sub>If this was useful, a ⭐ goes a long way.</sub>
</div>

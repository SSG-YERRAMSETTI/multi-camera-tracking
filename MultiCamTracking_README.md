<div align="center">

# Multi-Camera Live Object Tracking — v3.0

**Real-time multi-camera vehicle detection, tracking, and intelligence using YOLOv8 and Deep SORT**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Deep SORT](https://img.shields.io/badge/Deep_SORT-Realtime-8A2BE2?style=flat-square)](https://github.com/levan92/deep_sort_realtime)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

> Built on top of [LeonLok/Multi-Camera-Live-Object-Tracking](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking). v3.0 is a ground-up modernization — replacing Darknet with YOLOv8, Flask with FastAPI, and adding a full Vehicle Intelligence layer that wasn't in the original at all.

This system runs multiple camera feeds simultaneously, detects and tracks vehicles across frames with persistent IDs, counts traffic crossing configurable lines, and optionally identifies vehicle color, type, and license plate — all in real time through a browser dashboard.

---

## What's New in v3.0

The original project was a solid proof of concept. But it was built on TensorFlow 1.14 and Darknet — both painful to set up, and neither of them pip-installable. Here's what changed:

| Original (v1) | v3.0 |
|---|---|
| YOLOv4 via Darknet (C library, manual compilation) | **YOLOv8** via Ultralytics (pure Python, `pip install`) |
| TensorFlow 1.14 + Keras 2.3 | **PyTorch 2.2** |
| Flask + MJPEG streams | **FastAPI + WebSocket** (real-time, bidirectional) |
| Hardcoded settings requiring restart | **Hot-reload config** via `config.json` or REST API |
| Flat `.txt` / `.csv` files | **SQLite** with structured schema |
| No analytics | **Real-time analytics module** (FPS, counts, IDs, events) |
| No error handling | **Status monitor** with 5s auto-reconnect |
| Single shared YOLO instance (concurrency bug) | **Per-camera tracker instances** |
| No vehicle analysis | **Vehicle Intelligence** — color, type, license plate OCR |
| Two separate apps | **One unified system** with mode toggle |

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  camera_client.py  (runs on any device with a camera)       │
│  Captures frames → ZMQ → sends to backend on port 5555/5566 │
└─────────────────┬───────────────────────────────────────────┘
                  │  ZMQ (imagezmq)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  camera_manager.py  — one processing thread per camera      │
│                                                             │
│  Frame → YOLODetector (YOLOv8 inference, thread-safe)       │
│       → CameraTracker (Deep SORT, per-camera instance)      │
│       → TrafficCounter (line-crossing with cross-product)   │
│       → VehicleIntel   (color + OCR + type classification)  │
│       → AnalyticsModule (thread-safe stats aggregator)      │
└─────────────────┬───────────────────────────────────────────┘
                  │  WebSocket broadcast
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI server (app.py)                                    │
│  Serves dashboard.html + REST API + WebSocket stream        │
└─────────────────┬───────────────────────────────────────────┘
                  │  HTTP / WebSocket
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Browser Dashboard  (frontend/dashboard.html)               │
│  Live video feed · Counts · Controls · Activity log         │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Technical Features

**YOLOv8 Inference Engine (`yolo_detector.py`)**
Thread-safe singleton with FP16 half-precision support for 2× GPU speedup. Supports runtime model hot-swap without restarting the server. Falls back to a mock detection mode for testing without a GPU or model file.

**Deep SORT Tracker (`tracker.py`)**
One tracker instance per camera — this fixes a concurrency bug in the original where a shared YOLO instance caused race conditions across multiple streams. Each track maintains a center-point history for trajectory-based line crossing and a confidence history for averaged, stable confidence display.

**Vehicle Intelligence (`vehicle_intel.py`)**
Three layers of analysis running asynchronously on detected vehicles:
- **Color detection** — HSV color space matching across 12 color ranges
- **Type classification** — EfficientNet backbone (via timm) mapped to ImageNet vehicle classes
- **License plate OCR** — EasyOCR with graceful fallback if not installed

**Traffic Counter (`traffic_counter.py`)**
Virtual counting line with cross-product sign-change intersection math (more robust than angle calculations in the original). Each track ID counted exactly once, eliminating double-counts after re-identification gaps. Configurable direction (up/down, left/right) and async flush to SQLite.

**Hot-Reload Configuration (`configuration.py`)**
Change detection confidence, tracked classes, counting line position, or YOLO mode through the dashboard or REST API — no restart needed. Settings broadcast to subscribers via observer pattern.

---

## Project Structure

```
multi-camera-tracking/
│
├── backend/
│   ├── app.py                 ← FastAPI server + WebSocket broadcaster
│   ├── yolo_detector.py       ← YOLOv8 inference engine (thread-safe singleton)
│   ├── tracker.py             ← Deep SORT per-camera tracker
│   ├── traffic_counter.py     ← Line-crossing counter (cross-product method)
│   ├── vehicle_intel.py       ← Color + OCR + type classification
│   ├── analytics.py           ← Real-time stats aggregator
│   ├── camera_manager.py      ← Multi-camera pipeline threads
│   ├── configuration.py       ← Hot-reloadable settings manager
│   ├── database.py            ← SQLite schema + queries
│   └── status_monitor.py      ← Health monitor + auto-reconnect (5s)
│
├── frontend/
│   └── dashboard.html         ← Control room dashboard (WebSocket + REST)
│
├── models/
│   └── README.md              ← Model download instructions
│
├── logs/
│   ├── counts/                ← Traffic CSV logs (auto-generated)
│   └── objects/               ← Object TXT logs (auto-generated)
│
├── camera_client.py           ← Run on any device with a camera
├── config.json                ← All settings (edit live, no restart needed)
└── requirements.txt
```

---

## Getting Started

### Requirements

| Item | Requirement |
|------|------------|
| **Python** | 3.10.x recommended (3.9–3.11 work; avoid 3.12+) |
| **RAM** | 8 GB minimum, 16 GB for 2+ cameras |
| **GPU** | Optional — NVIDIA GPU with CUDA gives 3–5× speedup |
| **OS** | Windows 10/11, macOS 12+, Ubuntu 20.04+ |
| **Camera** | USB webcam, IP camera (RTSP), or video file for testing |

> **Note on Python 3.12+:** PyTorch wheels haven't fully caught up yet. Use 3.10.x for the smoothest experience.

---

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/SSG-YERRAMSETTI/multi-camera-tracking.git
cd multi-camera-tracking
```

**2. Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Expected install time: 5–15 minutes (downloads ~500 MB of PyTorch + model files).

**Optional — GPU acceleration (NVIDIA only)**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Then set `"device": "cuda"` in `config.json`.

**4. Verify the install**
```bash
python -c "import ultralytics; import fastapi; import cv2; print('All libraries OK')"
```

---

### Running the System

**Terminal 1 — Start the server**
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000](http://localhost:8000) — the dashboard will load in demo mode.

**Terminal 2 — Connect a camera**
```bash
# USB webcam
python camera_client.py --camera 0 --server tcp://localhost:5555

# Video file (for testing without a camera)
python camera_client.py --camera test_video.mp4 --server tcp://localhost:5555

# IP camera (RTSP)
python camera_client.py --camera rtsp://192.168.1.10/stream --server tcp://localhost:5555
```

**Adding a second camera (Terminal 3)**
```bash
python camera_client.py --camera 1 --server tcp://localhost:5566 --name camera_1
```

---

## Configuration

Edit `config.json` to change any setting — changes take effect immediately without restarting:

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

You can also change settings live through the dashboard controls or the REST API.

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
| `POST` | `/api/set_mode` | Switch object/traffic mode |
| `POST` | `/api/set_counting_line` | Move counting line position |
| `GET` | `/api/logs/traffic.csv` | Download traffic log |
| `GET` | `/api/logs/objects.txt` | Download object log |
| `GET` | `/docs` | Interactive API docs (Swagger) |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'ultralytics'`**
Your venv isn't activated. Run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux), then `pip install -r requirements.txt`.

**Port 8000 already in use**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

**Dashboard shows "demo mode" with server running**
The dashboard needs `camera_client.py` running in a separate terminal sending frames. Demo mode is the default when no camera is connected.

**`CUDA out of memory`**
Set `"device": "cpu"` in `config.json`.

**Camera not opening (`Cannot open source: 0`)**
Make sure no other app (Zoom, Teams) is using the webcam. Try `--camera 1` if `--camera 0` doesn't work.

---

## Acknowledgements

This project extends [LeonLok/Multi-Camera-Live-Object-Tracking](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking). The multi-camera ZMQ transport architecture and the original tracking concept come from that work. v3.0 modernizes the ML stack, adds the Vehicle Intelligence layer, and rebuilds the backend around FastAPI and WebSocket.

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

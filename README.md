# Multi-Camera Live Object Tracking — v3.0
### YOLOv8 · Deep SORT · FastAPI · WebSocket · SQLite

> Phase 3 modernized system built on top of LeonLok/Multi-Camera-Live-Object-Tracking

---

## Quick Start

```bash
# Python 3.10 required
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt

# Terminal 1 — Start server
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Connect camera
python camera_client.py --camera 0 --server tcp://localhost:5555
```

Open browser: **http://localhost:8000**

---

## 📖 Full instructions → see `SETUP_GUIDE.md`

---

## Project Structure

```
backend/
  app.py              ← FastAPI server + all REST endpoints
  configuration.py    ← Hot-reloadable settings manager
  database.py         ← SQLite: counts + config + events tables
  yolo_detector.py    ← YOLOv8 inference (replaces YOLOv4 + Darknet)
  tracker.py          ← Deep SORT per-camera tracker
  traffic_counter.py  ← Line-crossing counter + object counter
  analytics.py        ← Real-time FPS, counts, IDs aggregator
  camera_manager.py   ← Multi-camera pipeline threads
  status_monitor.py   ← Health monitor + auto-reconnect (5s)

frontend/
  dashboard.html      ← Full control room dashboard (WebSocket + REST)

camera_client.py      ← Run on any device with a camera
config.json           ← Edit settings here (no restart needed)
requirements.txt      ← All dependencies
```

---

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System health |
| `/api/analytics` | GET | Live counts + FPS |
| `/api/config` | GET | Current settings |
| `/api/set_yolo_mode` | POST | Toggle detection on/off |
| `/api/set_confidence` | POST | Change threshold |
| `/api/set_classes` | POST | Change tracked classes |
| `/api/set_mode` | POST | Switch mode |
| `/api/set_counting_line` | POST | Move counting line |
| `/api/logs/traffic.csv` | GET | Download traffic log |
| `/api/logs/objects.txt` | GET | Download object log |
| `/docs` | GET | Auto-generated API docs |

---

## Python Version

**Use Python 3.10.x** — tested and confirmed working.
3.9 and 3.11 also work. Avoid 3.12+.

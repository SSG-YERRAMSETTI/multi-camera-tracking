"""
app.py  —  FastAPI Application
Fixes:
  - VehicleIntel worker .start() called in lifespan
  - POST /api/set_camera_source/{camera_id} endpoint added
  - Serves frontend/dashboard.html at /
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel

from backend.configuration import SettingsManager
from backend.database import db
from backend.analytics import AnalyticsModule
from backend.status_monitor import StatusMonitor
from backend.camera_manager import CameraManager
from backend.vehicle_intel import VehicleIntel


# ─── Lifespan ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[App] Starting up...")

    await db.connect()
    await db.log_event("INFO", "System startup")

    monitor = StatusMonitor.get_instance()
    await monitor.start()

    # START vehicle intel worker thread
    VehicleIntel.get_instance().start()

    cam_manager = CameraManager.get_instance()
    cam_manager.start_all()

    flush_task = asyncio.create_task(_periodic_flush())

    print("[App] Ready — visit http://localhost:8000")
    yield

    print("[App] Shutting down...")
    flush_task.cancel()
    cam_manager.stop_all()
    VehicleIntel.get_instance().stop()
    await monitor.stop()
    await db.log_event("INFO", "System shutdown")
    await db.close()


async def _periodic_flush():
    settings    = SettingsManager.get_instance()
    cam_manager = CameraManager.get_instance()
    while True:
        interval = settings.get_traffic().write_interval_minutes * 60
        await asyncio.sleep(interval)
        for cid in cam_manager.camera_ids:
            pipeline = cam_manager.get_pipeline(cid)
            if pipeline:
                try:
                    await pipeline._traffic_counter.flush_to_db(db)
                    await pipeline._object_counter.flush_to_db(db)
                except Exception as e:
                    print(f"[Flush] cam={cid} error: {e}")


# ─── App ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Camera Live Object Tracking",
    version="3.0.0",
    lifespan=lifespan,
)


# ─── WebSocket manager ────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, data: dict):
        dead = set()
        msg  = json.dumps(data)
        # Iterate over a snapshot copy — prevents RuntimeError when
        # a client disconnects mid-broadcast and modifies self.active
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self.active -= dead


ws_manager = ConnectionManager()


async def analytics_broadcaster():
    analytics   = AnalyticsModule.get_instance()
    status_mon  = StatusMonitor.get_instance()
    cam_manager = CameraManager.get_instance()
    while True:
        await asyncio.sleep(0.5)
        if not ws_manager.active:
            continue
        payload = {
            "type":      "update",
            "analytics": analytics.snapshot(),
            "status":    status_mon.get_status(),
            "frames":    {},
        }
        for cid in cam_manager.camera_ids:
            pipeline = cam_manager.get_pipeline(cid)
            if pipeline:
                frame_b64 = pipeline.get_frame_b64()
                if frame_b64:
                    payload["frames"][str(cid)] = frame_b64
        await ws_manager.broadcast(payload)


# ─── Routes ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = Path("frontend/dashboard.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8")) 
    return HTMLResponse("<h2>Dashboard not found — check frontend/dashboard.html</h2>")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    asyncio.create_task(analytics_broadcaster())
    try:
        while True:
            msg  = await ws.receive_text()
            data = json.loads(msg)
            if data.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


# ── Status & Analytics ───────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    return StatusMonitor.get_instance().get_status()


@app.get("/api/analytics")
async def get_analytics():
    return AnalyticsModule.get_instance().snapshot()


@app.get("/api/config")
async def get_config():
    return SettingsManager.get_instance().to_dict()


# ── Control API ──────────────────────────────────────────────────

class YoloModeRequest(BaseModel):
    enabled: bool

class ConfidenceRequest(BaseModel):
    threshold: float

class ClassesRequest(BaseModel):
    classes: List[str]

class ModeRequest(BaseModel):
    mode: str

class CountingLineRequest(BaseModel):
    position: float
    camera_id: Optional[int] = None   # None = apply to all cameras

class DetectionRequest(BaseModel):
    confidence_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    classes_to_track: Optional[List[str]] = None
    yolo_enabled: Optional[bool] = None

class CameraSourceRequest(BaseModel):
    source: str     # "0", "1", file path, or RTSP URL


@app.post("/api/set_yolo_mode")
async def set_yolo_mode(req: YoloModeRequest):
    SettingsManager.get_instance().set_yolo_mode(req.enabled)
    await db.log_event("INFO", f"YOLO mode set to {req.enabled}")
    return {"success": True, "yolo_enabled": req.enabled}


@app.post("/api/set_confidence")
async def set_confidence(req: ConfidenceRequest):
    if not 0.0 <= req.threshold <= 1.0:
        raise HTTPException(400, "Threshold must be 0.0–1.0")
    SettingsManager.get_instance().set_confidence(req.threshold)
    return {"success": True, "confidence_threshold": req.threshold}


@app.post("/api/set_classes")
async def set_classes(req: ClassesRequest):
    SettingsManager.get_instance().set_classes(req.classes)
    return {"success": True, "classes": req.classes}


@app.post("/api/set_detection")
async def set_detection(req: DetectionRequest):
    params = req.model_dump(exclude_none=True)
    SettingsManager.get_instance().set_detection(params)
    return {"success": True, "updated": params}


@app.post("/api/set_mode")
async def set_mode(req: ModeRequest):
    if req.mode not in ("object_tracking", "traffic_counting"):
        raise HTTPException(400, "mode must be object_tracking or traffic_counting")
    SettingsManager.get_instance().set_mode(req.mode)
    await db.log_event("INFO", f"Mode switched to {req.mode}")
    return {"success": True, "mode": req.mode}


@app.post("/api/set_counting_line")
async def set_counting_line(req: CountingLineRequest):
    if not 0.0 <= req.position <= 1.0:
        raise HTTPException(400, "Position must be 0.0–1.0")
    SettingsManager.get_instance().set_counting_line(req.position)
    return {"success": True, "counting_line_position": req.position}


@app.post("/api/set_camera_source/{camera_id}")
async def set_camera_source(camera_id: int, req: CameraSourceRequest):
    """
    Change the video source for a camera and restart its pipeline.
    source can be:
      "0" or "1"         — webcam device index
      "rtsp://..."       — IP camera RTSP URL
      "myvideo.mp4"      — video file (must be in project root folder)
      "C:/full/path.mp4" — full absolute path to video file
    """
    cam_manager = CameraManager.get_instance()
    try:
        await cam_manager.change_source(camera_id, req.source)
        await db.log_event("INFO", f"Camera {camera_id} source changed to {req.source}")
        return {"success": True, "camera_id": camera_id, "source": req.source}
    except Exception as e:
        raise HTTPException(500, f"Failed to change source: {e}")


# ── Log Downloads ────────────────────────────────────────────────

@app.get("/api/logs/traffic.csv")
async def download_traffic_csv(camera_id: Optional[int] = None):
    csv_data = await db.export_intersections_csv(camera_id=camera_id)
    if not csv_data:
        csv_data = "No data available yet.\n"
    return Response(
        content=csv_data, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=traffic_log.csv"},
    )


@app.get("/api/logs/objects.txt")
async def download_objects_txt(camera_id: Optional[int] = None):
    rows = await db.get_counts(camera_id=camera_id, limit=1000)
    lines = [
        f"{r['timestamp']} | cam={r['camera_id']} | {r['object_class']}: {r['count']}"
        for r in rows
    ]
    content = "\n".join(lines) if lines else "No data available yet."
    return Response(
        content=content, media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=object_log.txt"},
    )


@app.get("/api/logs/analytics")
async def get_analytics_summary():
    return await db.get_analytics_summary()


@app.get("/api/events")
async def get_events(limit: int = 50):
    return await db.get_recent_events(limit=limit)


# ── MJPEG legacy stream ──────────────────────────────────────────

@app.get("/stream/{camera_id}")
async def mjpeg_stream(camera_id: int):
    pipeline = CameraManager.get_instance().get_pipeline(camera_id)
    if not pipeline:
        raise HTTPException(404, f"Camera {camera_id} not found")
    return StreamingResponse(
        pipeline.get_frame_mjpeg(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


# ─── Entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
    )
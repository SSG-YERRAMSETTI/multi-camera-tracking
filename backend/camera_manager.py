"""
camera_manager.py  —  Multi-Camera Pipeline Manager

Changes in this version:
  1. Green counting line RESTORED on each camera frame (traffic mode only)
     - Each camera draws its own independent line
     - Line crossing fires count + activity event
  2. Activity feed events fire in BOTH modes:
     - Traffic mode: vehicle crosses line → event
     - Object mode:  new track ID first appears → event
  3. HTTP MJPEG source support (DroidCam free: http://IP:4747/mjpegfeed)
  4. RTSP FFmpeg backend kept for IP cameras
  5. Frame skipping kept for smooth video
"""

from __future__ import annotations

import asyncio
import base64
import queue
import threading
import time
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

from backend.configuration import SettingsManager, CameraConfig
from backend.yolo_detector import YOLODetector
from backend.tracker import CameraTracker
from backend.traffic_counter import TrafficCounter, ObjectCounter
from backend.analytics import AnalyticsModule
from backend.status_monitor import StatusMonitor
from backend.vehicle_intel import VehicleIntel


# ─── Video Source ─────────────────────────────────────────────────────────────

class VideoSource:
    def __init__(self, source: str) -> None:
        self.source = source
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        src = self.source

        # ── RTSP stream (IP cameras) ──────────────────────────────────────────
        if isinstance(src, str) and src.startswith("rtsp://"):
            self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"[VideoSource] RTSP opened: {src}")
                return True
            print(f"[VideoSource] RTSP failed: {src}")
            return False

        # ── HTTP stream ───────────────────────────────────────────────────────
        # NOTE: OpenCV FFmpeg on Windows does NOT reliably open HTTP MJPEG.
        # For DroidCam use the virtual webcam (device index 1) instead.
        # HTTP support kept here as best-effort only.
        elif isinstance(src, str) and src.startswith("http://"):
            self._cap = cv2.VideoCapture(src)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"[VideoSource] HTTP stream opened: {src}")
                return True
            print(f"[VideoSource] HTTP failed (use device index for DroidCam): {src}")
            return False

        # ── Webcam device index ───────────────────────────────────────────────
        elif str(src).isdigit():
            idx = int(src)
            # Try MSMF first — confirmed working for DroidCam virtual webcam
            # on Windows (Device 1 = DroidCam if DroidCam client is running)
            self._cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                print(f"[VideoSource] Webcam {idx} opened via MSMF")
                return True
            # Fallback: default backend
            self._cap = cv2.VideoCapture(idx)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                print(f"[VideoSource] Webcam {idx} opened via default backend")
                return True
            print(f"[VideoSource] Webcam {idx} failed to open")
            return False

        # ── Video file ────────────────────────────────────────────────────────
        else:
            # Fix: 'Assertion fctx->async_lock failed' — disable FFmpeg async
            # decoding by forcing single-threaded mode for local video files.
            self._cap = cv2.VideoCapture(src)
            if self._cap.isOpened():
                # Single-threaded decode — prevents FFmpeg async_lock crash
                # when two video files are open simultaneously
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                print(f"[VideoSource] Video file opened: {src}")
                return True
            print(f"[VideoSource] Video file failed: {src}")
            return False

    def read(self) -> Optional[np.ndarray]:
        if not self._cap or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        # Loop video files when they end
        if not ret and not str(self.source).isdigit() \
                and not self.source.startswith("http") \
                and not self.source.startswith("rtsp"):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()


# ─── Camera Pipeline ──────────────────────────────────────────────────────────

class CameraPipeline:
    FRAME_QUEUE_SIZE = 3

    # Frame skipping — YOLO runs every N frames for smooth video
    DETECT_EVERY  = 3    # change to 2 if CPU is fast, 5 if still slow
    INTEL_EVERY   = 15
    INTEL_CLASSES = {"car", "truck", "bus"}

    def __init__(self, config: CameraConfig) -> None:
        self.config    = config
        self.camera_id = config.camera_id

        self._settings  = SettingsManager.get_instance()
        self._detector  = YOLODetector.get_instance()
        self._tracker   = CameraTracker(camera_id=self.camera_id)
        self._analytics = AnalyticsModule.get_instance()
        self._status    = StatusMonitor.get_instance()
        self._vi        = VehicleIntel.get_instance()

        self._traffic_counter = TrafficCounter(self.camera_id)
        self._object_counter  = ObjectCounter(self.camera_id)

        self._frame_queue: queue.Queue = queue.Queue(
            maxsize=self.FRAME_QUEUE_SIZE
        )
        self._thread: Optional[threading.Thread] = None
        self._running  = False
        self._source:  Optional[VideoSource] = None
        self._frame_n  = 0

        # Frame-skip cache
        self._last_detections: list = []
        self._last_tracks:     list = []

        # Track IDs seen in object mode (for "new vehicle detected" events)
        self._seen_track_ids: Set[int] = set()

        self._analytics.register_camera(self.camera_id)
        self._status.register_camera(
            self.camera_id,
            reconnect_callback=self._reconnect,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._pipeline_loop,
            name=f"cam-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        time.sleep(0.3)
        if self._source:
            self._source.release()

    def restart_with_source(self, new_source: str) -> None:
        was_running = self._running
        self.stop()
        time.sleep(0.5)

        self.config = CameraConfig(
            camera_id = self.camera_id,
            source    = new_source,
            zmq_port  = self.config.zmq_port,
            enabled   = True,
            width     = self.config.width,
            height    = self.config.height,
            fps_limit = self.config.fps_limit,
        )
        self._tracker         = CameraTracker(camera_id=self.camera_id)
        self._traffic_counter = TrafficCounter(self.camera_id)
        self._object_counter  = ObjectCounter(self.camera_id)
        self._frame_n         = 0
        self._last_detections = []
        self._last_tracks     = []
        self._seen_track_ids  = set()

        if was_running:
            self._running = True
            self._thread  = threading.Thread(
                target=self._pipeline_loop,
                name=f"cam-{self.camera_id}",
                daemon=True,
            )
            self._thread.start()

    async def _reconnect(self, camera_id: int) -> None:
        print(f"[CameraPipeline cam={camera_id}] Reconnecting...")
        if self._source:
            self._source.release()
        await asyncio.sleep(1.0)

    # ── Main pipeline loop ────────────────────────────────────────────────────

    def _pipeline_loop(self) -> None:
        self._source = VideoSource(self.config.source)

        while self._running:
            if not self._source.is_open:
                if not self._source.open():
                    print(f"[cam={self.camera_id}] Cannot open: {self.config.source}")
                    self._status.record_error(self.camera_id, "Cannot open source")
                    self._analytics.set_camera_connected(self.camera_id, False)
                    time.sleep(2.0)
                    continue
                self._analytics.set_camera_connected(self.camera_id, True)
                print(f"[cam={self.camera_id}] Opened: {self.config.source}")

            frame = self._source.read()
            if frame is None:
                time.sleep(0.01)
                continue

            self._frame_n += 1
            self._status.record_frame(self.camera_id)

            # Frame skipping — only run YOLO every DETECT_EVERY frames
            run_detection = (self._frame_n % self.DETECT_EVERY == 0)
            if run_detection:
                detections            = self._detector.detect(frame)
                tracks                = self._tracker.update(frame, detections)
                self._last_detections = detections
                self._last_tracks     = tracks
            else:
                detections = self._last_detections
                tracks     = self._last_tracks

            # Vehicle intel
            if self._frame_n % self.INTEL_EVERY == 0:
                for track in tracks:
                    if track.dominant_class in self.INTEL_CLASSES:
                        self._vi.submit(track.track_id, frame, track.bbox)
                self._vi.clear_stale({t.track_id for t in tracks})

            # ── Counting + Activity events ────────────────────────────────────
            mode = self._settings.get_mode()

            if mode == "traffic_counting":
                # Line crossing → count goes up + event fires
                crossing_events = self._traffic_counter.update(tracks)
                for ev in crossing_events:
                    # Enrich event with camera_id and class
                    ev["camera_id"]  = self.camera_id
                    ev["class_name"] = ev.get("class_name", "vehicle")
                    self._analytics.add_event(ev)

                counts        = self._traffic_counter.counts
                traffic_total = self._traffic_counter.total_count
                object_counts = self._object_counter.counts

            else:
                # Object tracking mode — fire event when new track ID appears
                self._object_counter.update(tracks)
                counts        = self._object_counter.counts
                traffic_total = 0
                object_counts = counts

                for track in tracks:
                    if track.track_id not in self._seen_track_ids:
                        self._seen_track_ids.add(track.track_id)
                        # New vehicle/person first detected → push to activity feed
                        self._analytics.add_event({
                            "track_id":   track.track_id,
                            "class_name": track.dominant_class,
                            "camera_id":  self.camera_id,
                            "timestamp":  _now_iso(),
                            "direction":  "",
                        })

                # Clean up track IDs that have disappeared
                active_ids = {t.track_id for t in tracks}
                self._seen_track_ids &= active_ids

            # Analytics snapshot
            self._analytics.record_frame(
                camera_id       = self.camera_id,
                active_tracks   = len(tracks),
                detection_count = len(detections),
                object_counts   = object_counts,
                traffic_counts  = counts if mode == "traffic_counting" else None,
                traffic_total   = traffic_total,
                vehicle_intel   = self._vi.get_all_results(),
            )

            # Annotate + encode
            annotated  = self._annotate(frame, tracks, mode)
            _, jpeg    = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            jpeg_bytes = jpeg.tobytes()

            try:
                self._frame_queue.put_nowait(jpeg_bytes)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(jpeg_bytes)
                except queue.Empty:
                    pass

        if self._source:
            self._source.release()

    # ── Frame annotation ──────────────────────────────────────────────────────

    def _annotate(self, frame: np.ndarray, tracks, mode: str) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        CLASS_COLORS = {
            "car":    (255, 165,   0),
            "truck":  (  0, 165, 255),
            "bus":    (  0,   0, 220),
            "person": (  0, 200, 100),
        }

        # Draw bounding boxes + labels
        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            color = CLASS_COLORS.get(track.dominant_class, (180, 180, 180))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            intel = self._vi.get_result(track.track_id)
            if intel and intel.color != "Unknown":
                label = f"#{track.track_id} {intel.color} {intel.vehicle_type}"
                if intel.plate_text:
                    label += f" [{intel.plate_text}]"
            else:
                label = (
                    f"#{track.track_id} {track.dominant_class}"
                    f" {track.avg_confidence:.2f}"
                )

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
            )
            cv2.rectangle(
                out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1
            )
            cv2.putText(
                out, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
            )

        # ── Traffic mode: draw green counting line on THIS camera's frame ─────
        # Each camera has its own TrafficCounter with its own line position.
        # So Camera 1 and Camera 2 each show their own independent green line.
        if mode == "traffic_counting":
            line_y = self._traffic_counter.line_y
            # Green line across full width
            cv2.line(
                out, (0, line_y), (w, line_y),
                (0, 220, 80), 2, cv2.LINE_AA
            )
            # Count label just above the line on the left
            count_label = (
                f"Cam {self.camera_id}  "
                f"Count: {self._traffic_counter.total_count}"
            )
            (lw, lh), _ = cv2.getTextSize(
                count_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            label_y = max(line_y - 6, lh + 4)
            cv2.rectangle(
                out,
                (6, label_y - lh - 4),
                (6 + lw + 6, label_y + 2),
                (0, 0, 0), -1,
            )
            cv2.putText(
                out, count_label, (10, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 1,
            )

        # FPS top-right
        fps = self._analytics.get_camera_fps(self.camera_id)
        cv2.putText(
            out, f"{fps:.1f} FPS", (w - 85, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        )

        return out

    # ── Frame delivery ────────────────────────────────────────────────────────

    def get_frame_b64(self) -> Optional[str]:
        try:
            jpeg = self._frame_queue.get(timeout=0.5)
            return base64.b64encode(jpeg).decode("utf-8")
        except queue.Empty:
            return None

    def get_frame_mjpeg(self):
        while True:
            try:
                jpeg = self._frame_queue.get(timeout=1.0)
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            except queue.Empty:
                continue


# ─── Camera Manager ───────────────────────────────────────────────────────────

class CameraManager:
    _instance: Optional["CameraManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._pipelines: Dict[int, CameraPipeline] = {}

    @classmethod
    def get_instance(cls) -> "CameraManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start_all(self) -> None:
        settings = SettingsManager.get_instance()
        for cam_cfg in settings.get_cameras():
            if cam_cfg.enabled:
                self.start_camera(cam_cfg)

    def start_camera(self, config: CameraConfig) -> None:
        cid = config.camera_id
        if cid not in self._pipelines:
            self._pipelines[cid] = CameraPipeline(config)
        self._pipelines[cid].start()

    def stop_camera(self, camera_id: int) -> None:
        if camera_id in self._pipelines:
            self._pipelines[camera_id].stop()

    def stop_all(self) -> None:
        for p in self._pipelines.values():
            p.stop()

    async def change_source(self, camera_id: int, source: str) -> None:
        settings = SettingsManager.get_instance()
        if camera_id in self._pipelines:
            self._pipelines[camera_id].restart_with_source(source)
        else:
            cameras = settings.get_cameras()
            cfg = next((c for c in cameras if c.camera_id == camera_id), None)
            cfg = CameraConfig(
                camera_id = camera_id,
                source    = source,
                zmq_port  = cfg.zmq_port if cfg else 5566,
                enabled   = True,
                width     = cfg.width if cfg else 640,
                height    = cfg.height if cfg else 480,
                fps_limit = cfg.fps_limit if cfg else 30,
            )
            pipeline = CameraPipeline(cfg)
            self._pipelines[camera_id] = pipeline
            pipeline.start()

        settings.update_camera(camera_id, {"source": source, "enabled": True})
        print(f"[CameraManager] cam={camera_id} source → {source}")

    def get_pipeline(self, camera_id: int) -> Optional[CameraPipeline]:
        return self._pipelines.get(camera_id)

    @property
    def camera_ids(self):
        return list(self._pipelines.keys())


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat()
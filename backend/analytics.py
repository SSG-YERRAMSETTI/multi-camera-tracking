"""
analytics.py  —  Analytics Module
Phase 3 NEW component: real-time FPS, object counts, IDs, system stats

Aggregates data from all camera pipelines and exposes a clean
snapshot for the WebSocket broadcast and REST API.
"""

from __future__ import annotations

import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CameraStats:
    camera_id: int
    fps: float = 0.0
    frame_count: int = 0
    active_tracks: int = 0
    detection_count: int = 0
    object_counts: Dict[str, int] = field(default_factory=dict)
    traffic_counts: Dict[str, int] = field(default_factory=dict)
    traffic_total: int = 0
    vehicle_intel: list = field(default_factory=list)
    is_connected: bool = False
    last_frame_ts: float = field(default_factory=time.time)

    # FPS rolling window
    _frame_times: deque = field(default_factory=lambda: deque(maxlen=30))

    def record_frame(self) -> None:
        now = time.time()
        self._frame_times.append(now)
        self.frame_count += 1
        self.last_frame_ts = now

        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self.fps = round((len(self._frame_times) - 1) / elapsed, 1)

    @property
    def is_stale(self) -> bool:
        return time.time() - self.last_frame_ts > 5.0   # 5s timeout → Phase 3 Availability QA


class AnalyticsModule:
    """
    Central analytics aggregator.
    Thread-safe singleton — written to by camera threads, read by WebSocket broadcaster.
    """

    _instance: Optional["AnalyticsModule"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._cameras: Dict[int, CameraStats] = {}
        self._system_fps_history: deque = deque(maxlen=60)
        self._rw_lock = threading.RLock()
        self._recent_events: deque = deque(maxlen=100)

    @classmethod
    def get_instance(cls) -> "AnalyticsModule":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Camera registration ───────────────────────────────────────────────────

    def register_camera(self, camera_id: int) -> None:
        with self._rw_lock:
            if camera_id not in self._cameras:
                self._cameras[camera_id] = CameraStats(camera_id=camera_id)

    def set_camera_connected(self, camera_id: int, connected: bool) -> None:
        with self._rw_lock:
            self._ensure_camera(camera_id)
            self._cameras[camera_id].is_connected = connected

    def _ensure_camera(self, camera_id: int) -> None:
        if camera_id not in self._cameras:
            self._cameras[camera_id] = CameraStats(camera_id=camera_id)

    # ── Per-frame updates (called from camera processing thread) ──────────────

    def record_frame(
        self,
        camera_id: int,
        active_tracks: int,
        detection_count: int,
        object_counts: dict,
        traffic_counts: Optional[dict] = None,
        traffic_total: int = 0,
        vehicle_intel: Optional[list] = None,
    ) -> None:
        with self._rw_lock:
            self._ensure_camera(camera_id)
            cam = self._cameras[camera_id]
            cam.record_frame()
            cam.active_tracks = active_tracks
            cam.detection_count = detection_count
            cam.object_counts = dict(object_counts)
            if traffic_counts is not None:
                cam.traffic_counts = dict(traffic_counts)
                cam.traffic_total = traffic_total
            if vehicle_intel is not None:
                cam.vehicle_intel = vehicle_intel

    def add_event(self, event: dict) -> None:
        """Push a crossing/detection event for the activity feed."""
        with self._rw_lock:
            self._recent_events.appendleft(event)

    # ── Snapshot for API / WebSocket ─────────────────────────────────────────

    def snapshot(self) -> dict:
        """Returns the complete analytics state for broadcasting."""
        with self._rw_lock:
            cameras = {}
            total_objects = 0
            total_traffic = 0
            all_class_counts: Dict[str, int] = defaultdict(int)
            fps_values = []

            for cid, cam in self._cameras.items():
                cameras[str(cid)] = {
                    "camera_id": cid,
                    "fps": cam.fps if not cam.is_stale else 0.0,
                    "frame_count": cam.frame_count,
                    "active_tracks": cam.active_tracks,
                    "object_counts": cam.object_counts,
                    "traffic_counts": cam.traffic_counts,
                    "traffic_total": cam.traffic_total,
                    "is_connected": cam.is_connected and not cam.is_stale,
                    "vehicle_intel": list(cam.vehicle_intel) if hasattr(cam, 'vehicle_intel') else [],
                }
                total_objects += sum(cam.object_counts.values())
                total_traffic += cam.traffic_total
                fps_values.append(cam.fps)
                for cls, cnt in cam.object_counts.items():
                    all_class_counts[cls] += cnt

            avg_fps = round(sum(fps_values) / max(len(fps_values), 1), 1)

            return {
                "cameras": cameras,
                "totals": {
                    "objects": total_objects,
                    "traffic": total_traffic,
                    "by_class": dict(all_class_counts),
                },
                "system": {
                    "avg_fps": avg_fps,
                    "camera_count": len(self._cameras),
                    "active_cameras": sum(
                        1 for c in self._cameras.values()
                        if c.is_connected and not c.is_stale
                    ),
                },
                "recent_events": list(self._recent_events)[:10],
            }

    def get_camera_fps(self, camera_id: int) -> float:
        with self._rw_lock:
            cam = self._cameras.get(camera_id)
            return cam.fps if cam else 0.0

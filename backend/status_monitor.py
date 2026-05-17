"""
status_monitor.py  —  Status Monitor / Error Handling
Phase 3 NEW component: handles camera failures, dropped frames, system errors

Quality Attributes addressed:
  - Availability: recover within 5 seconds of camera disconnect
  - Reliability: camera failure does not crash other streams
"""

from __future__ import annotations

import asyncio
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


class CameraStatus(Enum):
    CONNECTING   = "connecting"
    ACTIVE       = "active"
    STALE        = "stale"         # no frames for >2s but not yet failed
    FAILED       = "failed"        # no frames for >5s
    RECONNECTING = "reconnecting"
    DISABLED     = "disabled"


@dataclass
class CameraHealth:
    camera_id: int
    status: CameraStatus = CameraStatus.CONNECTING
    last_frame_ts: float = field(default_factory=time.time)
    reconnect_attempts: int = 0
    last_error: Optional[str] = None
    dropped_frames: int = 0
    total_frames: int = 0

    STALE_TIMEOUT  = 2.0   # seconds
    FAILED_TIMEOUT = 5.0   # Phase 3 QA: recover within 5s

    def record_frame(self) -> None:
        self.last_frame_ts = time.time()
        self.total_frames += 1
        if self.status != CameraStatus.ACTIVE:
            self.status = CameraStatus.ACTIVE
            self.reconnect_attempts = 0
            self.last_error = None

    def record_dropped_frame(self) -> None:
        self.dropped_frames += 1

    def check_health(self) -> CameraStatus:
        elapsed = time.time() - self.last_frame_ts
        if elapsed > self.FAILED_TIMEOUT:
            self.status = CameraStatus.FAILED
        elif elapsed > self.STALE_TIMEOUT:
            self.status = CameraStatus.STALE
        return self.status

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "status": self.status.value,
            "reconnect_attempts": self.reconnect_attempts,
            "last_error": self.last_error,
            "dropped_frames": self.dropped_frames,
            "total_frames": self.total_frames,
            "drop_rate": round(
                self.dropped_frames / max(self.total_frames, 1) * 100, 1
            ),
        }


class StatusMonitor:
    """
    Background health monitor.
    Checks all camera health every second.
    Triggers reconnect callbacks when cameras fail.
    Publishes status updates to subscribers (e.g. WebSocket broadcaster).
    """

    _instance: Optional["StatusMonitor"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._cameras: Dict[int, CameraHealth] = {}
        self._reconnect_callbacks: Dict[int, Callable] = {}
        self._status_callbacks: List[Callable] = []
        self._rw_lock = threading.RLock()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @classmethod
    def get_instance(cls) -> "StatusMonitor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_camera(
        self,
        camera_id: int,
        reconnect_callback: Optional[Callable] = None,
    ) -> None:
        with self._rw_lock:
            self._cameras[camera_id] = CameraHealth(camera_id=camera_id)
            if reconnect_callback:
                self._reconnect_callbacks[camera_id] = reconnect_callback

    def record_frame(self, camera_id: int) -> None:
        with self._rw_lock:
            if camera_id in self._cameras:
                self._cameras[camera_id].record_frame()

    def record_error(self, camera_id: int, error: str) -> None:
        with self._rw_lock:
            if camera_id in self._cameras:
                self._cameras[camera_id].last_error = error
                self._cameras[camera_id].status = CameraStatus.FAILED
        print(f"[StatusMonitor] cam={camera_id} ERROR: {error}")

    def subscribe(self, callback: Callable) -> None:
        self._status_callbacks.append(callback)

    # ── Async monitor loop ────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    async def _monitor_loop(self) -> None:
        while self._running:
            await asyncio.sleep(1.0)
            with self._rw_lock:
                for camera_id, health in self._cameras.items():
                    prev_status = health.status
                    new_status = health.check_health()

                    if new_status == CameraStatus.FAILED and prev_status != CameraStatus.FAILED:
                        print(f"[StatusMonitor] cam={camera_id} FAILED — triggering reconnect")
                        health.status = CameraStatus.RECONNECTING
                        health.reconnect_attempts += 1
                        cb = self._reconnect_callbacks.get(camera_id)
                        if cb:
                            asyncio.create_task(self._safe_reconnect(cb, camera_id))

                    if new_status != prev_status:
                        self._broadcast_status()

    async def _safe_reconnect(self, callback: Callable, camera_id: int) -> None:
        try:
            await asyncio.sleep(1.0)   # brief delay before reconnect
            await callback(camera_id)
        except Exception as e:
            with self._rw_lock:
                if camera_id in self._cameras:
                    self._cameras[camera_id].last_error = str(e)

    def _broadcast_status(self) -> None:
        snapshot = self.get_status()
        for cb in self._status_callbacks:
            try:
                cb(snapshot)
            except Exception:
                pass

    # ── Status snapshot ───────────────────────────────────────────────────────

    def get_status(self) -> dict:
        with self._rw_lock:
            cameras = {cid: h.to_dict() for cid, h in self._cameras.items()}
            all_ok = all(
                h.status == CameraStatus.ACTIVE for h in self._cameras.values()
            )
            return {
                "all_ok": all_ok,
                "cameras": cameras,
                "system_message": "All systems operational" if all_ok else "Camera issues detected",
            }

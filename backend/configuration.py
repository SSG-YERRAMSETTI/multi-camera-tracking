"""
configuration.py  —  Settings Manager
Phase 3 NEW component: /set_detection, /set_yolo_mode, /set_classes, /set_confidence
All settings are hot-reloadable at runtime — no server restart required.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ─── Pydantic models for each settings domain ──────────────────────────────────

class CameraConfig(BaseModel):
    camera_id: int = 0
    source: str = "0"                # "0" = webcam, URL = IP cam, path = file
    zmq_port: int = 5555
    enabled: bool = True
    width: int = 640
    height: int = 480
    fps_limit: int = 30


class DetectionConfig(BaseModel):
    model_path: str = "yolov8n.pt"  # YOLOv8 nano by default; swap for v11
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    classes_to_track: List[str] = ["car", "truck", "bus", "person"]
    device: str = "cpu"              # "cuda" if GPU available
    half_precision: bool = False     # FP16 inference for GPU speedup
    yolo_enabled: bool = True


class TrackingConfig(BaseModel):
    enabled: bool = True
    max_age: int = 30                # frames before a lost track is deleted
    min_hits: int = 3                # detections needed before track is confirmed
    iou_threshold: float = 0.3
    embedder: str = "mobilenet"      # Deep SORT appearance model


class TrafficConfig(BaseModel):
    counting_line_position: float = Field(default=0.5, ge=0.0, le=1.0)  # 0–1 of frame height
    count_direction: str = "both"    # "up", "down", or "both"
    write_interval_minutes: int = 5  # how often to flush counts to DB / CSV


class AppConfig(BaseModel):
    mode: str = "object_tracking"    # "object_tracking" | "traffic_counting"
    cameras: List[CameraConfig] = [CameraConfig(camera_id=0), CameraConfig(camera_id=1, zmq_port=5566)]
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    traffic: TrafficConfig = TrafficConfig()
    log_dir: str = "logs"
    db_path: str = "tracking.db"


# ─── Settings Manager  ─────────────────────────────────────────────────────────

class SettingsManager:
    """
    Thread-safe runtime settings manager.
    Persists to config.json so settings survive restarts.
    All changes propagate immediately — no restart needed (Phase 3 Modifiability QA).
    """

    _CONFIG_FILE = Path("config.json")
    _instance: Optional["SettingsManager"] = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        self._config = self._load_or_default()
        self._subscribers: list = []

    # ── Singleton ──────────────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "SettingsManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Load / Save ────────────────────────────────────────────────────────────
    def _load_or_default(self) -> AppConfig:
        if self._CONFIG_FILE.exists():
            try:
                data = json.loads(self._CONFIG_FILE.read_text())
                return AppConfig(**data)
            except Exception:
                pass
        return AppConfig()

    def _save(self) -> None:
        self._CONFIG_FILE.write_text(
            self._config.model_dump_json(indent=2)
        )

    # ── Getters ────────────────────────────────────────────────────────────────
    @property
    def config(self) -> AppConfig:
        with self._lock:
            return self._config

    def get_detection(self) -> DetectionConfig:
        return self.config.detection

    def get_tracking(self) -> TrackingConfig:
        return self.config.tracking

    def get_traffic(self) -> TrafficConfig:
        return self.config.traffic

    def get_cameras(self) -> List[CameraConfig]:
        return self.config.cameras

    def get_mode(self) -> str:
        return self.config.mode

    # ── Setters (REST API endpoints call these) ────────────────────────────────
    def set_yolo_mode(self, enabled: bool) -> None:
        """POST /set_yolo_mode"""
        with self._lock:
            self._config.detection.yolo_enabled = enabled
            self._save()
            self._notify("yolo_mode", enabled)

    def set_confidence(self, threshold: float) -> None:
        """POST /set_confidence"""
        with self._lock:
            self._config.detection.confidence_threshold = max(0.0, min(1.0, threshold))
            self._save()
            self._notify("confidence", threshold)

    def set_classes(self, classes: List[str]) -> None:
        """POST /set_classes"""
        with self._lock:
            self._config.detection.classes_to_track = classes
            self._save()
            self._notify("classes", classes)

    def set_detection(self, params: dict) -> None:
        """POST /set_detection  — bulk update"""
        with self._lock:
            current = self._config.detection.model_dump()
            current.update(params)
            self._config.detection = DetectionConfig(**current)
            self._save()
            self._notify("detection", params)

    def set_mode(self, mode: str) -> None:
        """Switch between object_tracking and traffic_counting"""
        assert mode in ("object_tracking", "traffic_counting"), f"Unknown mode: {mode}"
        with self._lock:
            self._config.mode = mode
            self._save()
            self._notify("mode", mode)

    def set_counting_line(self, position: float) -> None:
        with self._lock:
            self._config.traffic.counting_line_position = position
            self._save()
            self._notify("counting_line", position)

    def update_camera(self, camera_id: int, params: dict) -> None:
        with self._lock:
            for cam in self._config.cameras:
                if cam.camera_id == camera_id:
                    updated = cam.model_dump()
                    updated.update(params)
                    cam.__dict__.update(CameraConfig(**updated).__dict__)
                    break
            self._save()
            self._notify("camera", {"id": camera_id, **params})

    # ── Observer pattern for live updates ─────────────────────────────────────
    def subscribe(self, callback) -> None:
        self._subscribers.append(callback)

    def _notify(self, key: str, value) -> None:
        for cb in self._subscribers:
            try:
                cb(key, value)
            except Exception:
                pass

    # ── Snapshot for API responses ─────────────────────────────────────────────
    def to_dict(self) -> dict:
        return self._config.model_dump()

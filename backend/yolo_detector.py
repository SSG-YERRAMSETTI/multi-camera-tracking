"""
yolo_detector.py  —  YOLOv8 / YOLOv11 Inference Engine
Replaces: yolo.py + Darknet + Keras conversion pipeline from original system
Phase 3: Video Processing Core Engine — Inference Engine component

Key improvements over original:
  - YOLOv8/v11 via Ultralytics (pure Python, no Darknet conversion needed)
  - FP16 half-precision support for 2× GPU speedup
  - Runtime model hot-swap without restart
  - Singleton with thread-safe access
  - Native class filtering by name (not hardcoded indices)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from backend.configuration import SettingsManager, DetectionConfig


# ─── Detection result dataclass ────────────────────────────────────────────────

@dataclass
class Detection:
    """Single object detection from YOLO inference."""
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 (absolute pixels)
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    def to_tlwh(self) -> Tuple[float, float, float, float]:
        """Convert to [top, left, width, height] for Deep SORT."""
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)

    def to_xywh(self) -> List[float]:
        """[x_center, y_center, width, height] — normalised 0-1."""
        return list(self.center) + [self.width, self.height]


# ─── YOLO Detector ─────────────────────────────────────────────────────────────

class YOLODetector:
    """
    Thread-safe YOLOv8/v11 singleton.

    Usage:
        detector = YOLODetector.get_instance()
        detections = detector.detect(frame)
    """

    _instance: Optional["YOLODetector"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model: Optional[object] = None
        self._model_path: str = ""
        self._settings = SettingsManager.get_instance()
        self._inference_lock = threading.Lock()

        # Performance counters
        self._frame_count = 0
        self._total_inference_ms = 0.0
        self._last_fps = 0.0
        self._fps_timer = time.time()

        self._load_model()

        # Subscribe to config changes for hot-reload
        self._settings.subscribe(self._on_config_change)

    @classmethod
    def get_instance(cls) -> "YOLODetector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        cfg: DetectionConfig = self._settings.get_detection()
        if not ULTRALYTICS_AVAILABLE:
            print("[YOLODetector] WARNING: ultralytics not installed — running in mock mode")
            self._model = None
            return

        if self._model_path == cfg.model_path:
            return  # already loaded

        print(f"[YOLODetector] Loading model: {cfg.model_path} on {cfg.device}")
        self._model = YOLO(cfg.model_path)
        if cfg.device != "cpu":
            self._model.to(cfg.device)
        self._model_path = cfg.model_path
        print(f"[YOLODetector] Model ready — classes: {self._model.names}")

    def _on_config_change(self, key: str, value) -> None:
        """Called by SettingsManager when any setting changes."""
        if key in ("detection", "yolo_mode") and key == "detection":
            cfg = self._settings.get_detection()
            if cfg.model_path != self._model_path:
                with self._inference_lock:
                    self._load_model()

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame (OpenCV format).
        Returns list of Detection objects filtered by current config.
        """
        cfg = self._settings.get_detection()

        if not cfg.yolo_enabled:
            return []

        if self._model is None:
            return self._mock_detect(frame)

        t0 = time.perf_counter()

        with self._inference_lock:
            results = self._model(
                frame,
                conf=cfg.confidence_threshold,
                iou=cfg.iou_threshold,
                verbose=False,
                half=cfg.half_precision,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._update_fps_counter(elapsed_ms)

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self._model.names[class_id]

                # Filter to only tracked classes
                if class_name not in cfg.classes_to_track:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                ))

        return detections

    def _mock_detect(self, frame: np.ndarray) -> List[Detection]:
        """Returns synthetic detections for testing without GPU/model."""
        import random
        h, w = frame.shape[:2]
        n = random.randint(1, 4)
        detections = []
        classes = self._settings.get_detection().classes_to_track or ["car"]
        for _ in range(n):
            x1 = random.uniform(0.1, 0.6) * w
            y1 = random.uniform(0.1, 0.6) * h
            x2 = x1 + random.uniform(60, 120)
            y2 = y1 + random.uniform(60, 100)
            detections.append(Detection(
                bbox=(x1, y1, min(x2, w), min(y2, h)),
                confidence=round(random.uniform(0.6, 0.99), 2),
                class_id=0,
                class_name=random.choice(classes),
            ))
        return detections

    # ── Performance tracking ──────────────────────────────────────────────────

    def _update_fps_counter(self, inference_ms: float) -> None:
        self._frame_count += 1
        self._total_inference_ms += inference_ms
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self._last_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = time.time()

    def get_stats(self) -> Dict[str, float]:
        avg_ms = (
            self._total_inference_ms / max(1, self._frame_count)
            if self._frame_count
            else 0.0
        )
        return {
            "fps": round(self._last_fps, 1),
            "avg_inference_ms": round(avg_ms, 1),
            "model": self._model_path,
            "yolo_enabled": self._settings.get_detection().yolo_enabled,
        }

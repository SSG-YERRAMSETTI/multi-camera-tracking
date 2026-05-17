"""
tracker.py  —  Deep SORT Tracker
Phase 3: Feature Extraction & Track Management component

Replaces: nwojke/deep_sort with deep-sort-realtime (modern, pip-installable)
Key improvements:
  - No manual .pb model file management
  - Per-camera tracker instances (fixes the singleton parallelism problem)
  - Low-confidence track filtering (from Phase 2 IEEE paper approach)
  - Track history for trajectory-based counting
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False

from backend.yolo_detector import Detection
from backend.configuration import SettingsManager


# ─── Track dataclass ───────────────────────────────────────────────────────────

@dataclass
class Track:
    """
    Enriched track object — combines Deep SORT output with
    the per-class confidence averaging from Phase 2.
    """
    track_id: int
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2
    class_name: str
    confidence: float

    # History for line-crossing detection
    center_history: deque = field(default_factory=lambda: deque(maxlen=30))

    # For averaged confidence display (Phase 2 feature)
    _confidence_history: deque = field(default_factory=lambda: deque(maxlen=10))
    _class_votes: dict = field(default_factory=dict)

    def update(self, bbox, class_name: str, confidence: float) -> None:
        self.bbox = bbox
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.center_history.append((cx, cy))
        self._confidence_history.append(confidence)
        self._class_votes[class_name] = self._class_votes.get(class_name, 0) + 1

    @property
    def avg_confidence(self) -> float:
        if not self._confidence_history:
            return self.confidence
        return round(sum(self._confidence_history) / len(self._confidence_history), 2)

    @property
    def dominant_class(self) -> str:
        """Most frequently detected class for this track (Phase 2 feature)."""
        if not self._class_votes:
            return self.class_name
        return max(self._class_votes, key=self._class_votes.get)

    @property
    def center(self) -> Optional[Tuple[float, float]]:
        if self.center_history:
            return self.center_history[-1]
        return None

    @property
    def velocity(self) -> Optional[Tuple[float, float]]:
        """Approximate velocity from last two center points."""
        if len(self.center_history) < 2:
            return None
        p1 = self.center_history[-2]
        p2 = self.center_history[-1]
        return (p2[0] - p1[0], p2[1] - p1[1])

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "class_name": self.dominant_class,
            "confidence": self.avg_confidence,
            "center": list(self.center) if self.center else None,
        }


# ─── Per-Camera Tracker ────────────────────────────────────────────────────────

class CameraTracker:
    """
    One Deep SORT tracker instance per camera.
    Fixes the original system's YOLO singleton parallelism problem.
    """

    def __init__(self, camera_id: int) -> None:
        self.camera_id = camera_id
        self._settings = SettingsManager.get_instance()
        self._tracks: Dict[int, Track] = {}
        self._deepsort: Optional[object] = None
        self._init_deepsort()

    def _init_deepsort(self) -> None:
        cfg = self._settings.get_tracking()
        if not cfg.enabled:
            return
        if not DEEPSORT_AVAILABLE:
            print(f"[Tracker cam={self.camera_id}] deep-sort-realtime not installed — ID assignment disabled")
            return
        self._deepsort = DeepSort(
            max_age=cfg.max_age,
            n_init=cfg.min_hits,
            nms_max_overlap=1.0,
            max_cosine_distance=0.4,
            nn_budget=None,
            override_track_class=None,
            embedder=cfg.embedder,
            half=False,
            bgr=True,
            embedder_gpu=False,
        )

    def update(self, frame: np.ndarray, detections: List[Detection]) -> List[Track]:
        """
        Feed new detections into Deep SORT, return active tracks.
        Falls back to detection-only (no IDs) if Deep SORT unavailable.
        """
        cfg = self._settings.get_tracking()

        if not cfg.enabled or self._deepsort is None:
            return self._detection_only(detections)

        # Deep SORT expects [[x,y,w,h], confidence, class_name]
        raw = [
            (list(det.to_tlwh()), det.confidence, det.class_name)
            for det in detections
        ]

        try:
            ds_tracks = self._deepsort.update_tracks(raw, frame=frame)
        except Exception as e:
            print(f"[Tracker cam={self.camera_id}] Deep SORT error: {e}")
            return self._detection_only(detections)

        active: List[Track] = []
        seen_ids = set()

        for ds_track in ds_tracks:
            if not ds_track.is_confirmed():
                continue

            tid = ds_track.track_id
            seen_ids.add(tid)

            ltrb = ds_track.to_ltrb()
            bbox = (ltrb[0], ltrb[1], ltrb[2], ltrb[3])
            class_name = ds_track.get_det_class() or "unknown"
            conf = ds_track.get_det_conf() or 0.0

            if tid not in self._tracks:
                self._tracks[tid] = Track(
                    track_id=tid, bbox=bbox,
                    class_name=class_name, confidence=conf,
                )
            self._tracks[tid].update(bbox, class_name, conf)
            active.append(self._tracks[tid])

        # Clean up old tracks
        for tid in list(self._tracks.keys()):
            if tid not in seen_ids:
                del self._tracks[tid]

        return active

    def _detection_only(self, detections: List[Detection]) -> List[Track]:
        """Fallback: assign temporary IDs based on detection index."""
        tracks = []
        for i, det in enumerate(detections):
            t = Track(
                track_id=i,
                bbox=det.bbox,
                class_name=det.class_name,
                confidence=det.confidence,
            )
            t.center_history.append(t.center)
            tracks.append(t)
        return tracks

    def get_active_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        if self._deepsort:
            self._init_deepsort()

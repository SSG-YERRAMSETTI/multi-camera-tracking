"""
vehicle_intel.py  —  Vehicle Intelligence Module
Fixes:
  - Removed broken Stanford Cars timm internal API calls
  - Worker thread now reliably starts via .start()
  - Color detection always works (pure OpenCV)
  - Vehicle type from EfficientNet ImageNet classes (reliable)
  - EasyOCR plate detection with graceful fallback
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import timm
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[VehicleIntel] timm/torch not found - vehicle type from bbox ratio only")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[VehicleIntel] easyocr not found - plate detection disabled")

# ImageNet vehicle class indices -> display name
IMAGENET_VEHICLE_MAP = {
    407: "Ambulance",
    408: "Taxi / Cab",
    479: "Car",
    511: "Convertible",
    555: "Fire Engine",
    569: "Garbage Truck",
    609: "Jeep / 4x4",
    627: "Limousine",
    656: "Minivan",
    675: "Moving Van",
    717: "Pickup Truck",
    734: "Police Van",
    751: "Race Car",
    757: "RV / Camper",
    779: "School Bus",
    817: "Sports Car",
    820: "Station Wagon",
    867: "Tow Truck",
}

COLOR_RANGES = [
    ("Red",    [  0,  90,  60], [ 10, 255, 255]),
    ("Red",    [160,  90,  60], [180, 255, 255]),
    ("Orange", [ 11,  90,  60], [ 25, 255, 255]),
    ("Yellow", [ 26,  90,  60], [ 34, 255, 255]),
    ("Green",  [ 35,  40,  40], [ 85, 255, 255]),
    ("Blue",   [ 86,  60,  60], [128, 255, 255]),
    ("Purple", [129,  40,  40], [159, 255, 255]),
    ("White",  [  0,   0, 200], [180,  30, 255]),
    ("Silver", [  0,   0, 150], [180,  25, 200]),
    ("Black",  [  0,   0,   0], [180, 255,  50]),
    ("Gray",   [  0,   0,  51], [180,  28, 149]),
    ("Brown",  [  5,  50,  50], [ 20, 200, 150]),
]


@dataclass
class VehicleIntelResult:
    track_id: int
    color: str = "Unknown"
    color_confidence: float = 0.0
    vehicle_type: str = ""
    type_confidence: float = 0.0
    plate_text: str = ""
    plate_confidence: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "track_id":         self.track_id,
            "color":            self.color,
            "color_confidence": round(self.color_confidence, 2),
            "vehicle_type":     self.vehicle_type,
            "type_confidence":  round(self.type_confidence, 2),
            "plate_text":       self.plate_text,
            "plate_confidence": round(self.plate_confidence, 2),
            "last_updated":     self.last_updated,
        }


class VehicleIntel:
    _instance: Optional["VehicleIntel"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._results: Dict[int, VehicleIntelResult] = {}
        self._rw_lock  = threading.RLock()
        self._queue: list = []
        self._q_lock   = threading.Lock()
        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._classifier = None
        self._transform  = None
        self._ocr        = None
        self._device     = "cpu"
        self._load_models()

    @classmethod
    def get_instance(cls) -> "VehicleIntel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_models(self) -> None:
        if TIMM_AVAILABLE:
            try:
                print("[VehicleIntel] Loading EfficientNet-B0 (ImageNet weights)...")
                self._classifier = timm.create_model("efficientnet_b0", pretrained=True)
                self._classifier.eval()
                if torch.cuda.is_available():
                    self._classifier = self._classifier.cuda()
                    self._device = "cuda"
                self._transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
                print("[VehicleIntel] Classifier ready")
            except Exception as e:
                print(f"[VehicleIntel] Classifier load failed: {e}")
                self._classifier = None

        if EASYOCR_AVAILABLE:
            try:
                print("[VehicleIntel] Loading EasyOCR (first run ~100MB download)...")
                gpu = torch.cuda.is_available() if TIMM_AVAILABLE else False
                self._ocr = easyocr.Reader(["en"], gpu=gpu, verbose=False)
                print("[VehicleIntel] OCR ready")
            except Exception as e:
                print(f"[VehicleIntel] OCR failed: {e}")
                self._ocr = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, name="vehicle-intel-worker", daemon=True
        )
        self._thread.start()
        print("[VehicleIntel] Worker thread started")

    def stop(self) -> None:
        self._running = False

    def _worker(self) -> None:
        while self._running:
            job = None
            with self._q_lock:
                if self._queue:
                    job = self._queue.pop(0)
            if job is None:
                time.sleep(0.04)
                continue
            track_id, crop = job
            try:
                self._process(track_id, crop)
            except Exception as e:
                print(f"[VehicleIntel] Error on track {track_id}: {e}")

    def submit(self, track_id: int, frame: np.ndarray,
               bbox: Tuple[float, float, float, float]) -> None:
        with self._rw_lock:
            existing = self._results.get(track_id)
            if existing and (time.time() - existing.last_updated) < 2.0:
                return
        x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
        x2 = min(frame.shape[1], int(bbox[2]))
        y2 = min(frame.shape[0], int(bbox[3]))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return
        with self._q_lock:
            self._queue = [j for j in self._queue if j[0] != track_id]
            if len(self._queue) < 24:
                self._queue.append((track_id, crop.copy()))

    def get_result(self, track_id: int) -> Optional[VehicleIntelResult]:
        with self._rw_lock:
            return self._results.get(track_id)

    def get_all_results(self) -> List[dict]:
        now = time.time()
        with self._rw_lock:
            return [
                r.to_dict() for r in self._results.values()
                if (now - r.last_updated) < 10.0
            ]

    def clear_stale(self, active_ids: set) -> None:
        with self._rw_lock:
            for tid in list(self._results):
                if tid not in active_ids:
                    del self._results[tid]

    def _process(self, track_id: int, crop: np.ndarray) -> None:
        color, color_conf       = self._detect_color(crop)
        vtype, type_conf        = self._classify_type(crop)
        plate, plate_conf       = self._read_plate(crop)
        with self._rw_lock:
            self._results[track_id] = VehicleIntelResult(
                track_id=track_id,
                color=color, color_confidence=color_conf,
                vehicle_type=vtype, type_confidence=type_conf,
                plate_text=plate, plate_confidence=plate_conf,
                last_updated=time.time(),
            )

    def _detect_color(self, crop: np.ndarray) -> Tuple[str, float]:
        try:
            h, w = crop.shape[:2]
            roi = crop[int(h*.15):int(h*.75), int(w*.10):int(w*.90)]
            if roi.size == 0:
                return "Unknown", 0.0
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            total = roi.shape[0] * roi.shape[1]
            best_name, best_n = "Unknown", 0
            for name, lo, hi in COLOR_RANGES:
                n = int(np.sum(cv2.inRange(hsv, np.array(lo), np.array(hi)) > 0))
                if n > best_n:
                    best_n, best_name = n, name
            return best_name, round(min(1.0, best_n / max(total * 0.25, 1)), 2)
        except Exception:
            return "Unknown", 0.0

    def _classify_type(self, crop: np.ndarray) -> Tuple[str, float]:
        h, w = crop.shape[:2]
        ratio = w / max(h, 1)
        if ratio > 2.2:   fallback = "Truck / Van"
        elif ratio > 1.65: fallback = "SUV / Crossover"
        elif ratio > 1.3:  fallback = "Sedan / Hatchback"
        else:              fallback = "Coupe / Sports"

        if self._classifier is None or self._transform is None:
            return fallback, 0.0
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t   = self._transform(rgb).unsqueeze(0)
            if self._device == "cuda":
                t = t.cuda()
            with torch.no_grad():
                probs = F.softmax(self._classifier(t), dim=1)[0]
            top5_vals, top5_idx = probs.topk(5)
            for val, idx in zip(top5_vals.tolist(), top5_idx.tolist()):
                if idx in IMAGENET_VEHICLE_MAP:
                    return IMAGENET_VEHICLE_MAP[idx], round(val, 2)
            return fallback, 0.0
        except Exception:
            return fallback, 0.0

    def _read_plate(self, crop: np.ndarray) -> Tuple[str, float]:
        if self._ocr is None:
            return "", 0.0
        try:
            h, w = crop.shape[:2]
            roi  = crop[int(h * 0.55):h, :]
            if roi.size == 0:
                return "", 0.0
            scale = max(1, 100 // max(roi.shape[0], 1))
            if scale > 1:
                roi = cv2.resize(roi, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC)
            results = self._ocr.readtext(
                roi,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -",
                detail=1, paragraph=False,
            )
            best_text, best_conf = "", 0.0
            for (_, text, conf) in results:
                clean = text.strip().upper().replace(" ", "")
                alpha = sum(c.isalnum() for c in clean) / max(len(clean), 1)
                if 4 <= len(clean) <= 10 and alpha >= 0.7 and conf > best_conf:
                    best_text, best_conf = clean, conf
            return best_text, round(best_conf, 2)
        except Exception:
            return "", 0.0
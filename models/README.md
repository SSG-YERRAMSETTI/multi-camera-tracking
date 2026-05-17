# Model Weights — Download Instructions

Place your YOLO model weights (.pt file) in this folder.

## Option A — Auto Download (Easiest)
The system auto-downloads YOLOv8n on first run. No action needed.
Just start the server and it will download ~6MB automatically.

## Option B — Manual Download
Download from Ultralytics:
  https://github.com/ultralytics/assets/releases

| Model      | Size  | Speed | Accuracy | Best for              |
|------------|-------|-------|----------|-----------------------|
| yolov8n.pt | 6 MB  | ████  | ██       | Testing / low-end CPU |
| yolov8s.pt | 22 MB | ███   | ███      | Balanced              |
| yolov8m.pt | 50 MB | ██    | ████     | Good GPU              |
| yolov8l.pt | 87 MB | █     | █████    | High-end GPU          |
| yolov11n.pt| 5 MB  | ████  | ███      | Latest / recommended  |

## Change the model
Edit config.json:
  "model_path": "models/yolov8s.pt"

Or via REST API (no restart needed):
  POST http://localhost:8000/api/set_detection
  Body: {"model_path": "models/yolov8s.pt"}

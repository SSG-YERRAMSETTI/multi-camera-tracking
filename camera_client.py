"""
camera_client.py  —  Camera Client
Runs on the device with the camera (smartphone, laptop, IP cam).
Captures frames and sends them to the server via ImageZMQ.

Usage:
    python camera_client.py --camera 0 --server tcp://SERVER_IP:5555
    python camera_client.py --camera rtsp://192.168.1.10/stream --server tcp://SERVER_IP:5555
"""

import argparse
import time
import sys

import cv2
import imagezmq


def run_client(camera_source: str, server_address: str, camera_name: str, fps_limit: int) -> None:
    sender = imagezmq.ImageSender(connect_to=server_address)
    cap_src = int(camera_source) if camera_source.isdigit() else camera_source
    cap = cv2.VideoCapture(cap_src)

    if not cap.isOpened():
        print(f"[Client] ERROR: Cannot open camera source: {camera_source}")
        sys.exit(1)

    print(f"[Client] Streaming '{camera_name}' → {server_address}")
    frame_interval = 1.0 / fps_limit
    last_send = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Client] Frame read failed — retrying...")
                time.sleep(0.5)
                continue

            now = time.time()
            if now - last_send >= frame_interval:
                sender.send_image(camera_name, frame)
                last_send = now

    except KeyboardInterrupt:
        print("[Client] Stopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Camera Tracking Client")
    parser.add_argument("--camera", default="0", help="Camera source (0=webcam, URL, or file path)")
    parser.add_argument("--server", default="tcp://localhost:5555", help="ZMQ server address")
    parser.add_argument("--name",   default="camera_0",             help="Camera identifier name")
    parser.add_argument("--fps",    type=int, default=30,           help="Max FPS to send")
    args = parser.parse_args()

    run_client(args.camera, args.server, args.name, args.fps)

import cv2

print("=== Test 1: Default backend (no DSHOW) ===")
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Device {i}: FOUND ({w}x{h})")
        cap.release()
    else:
        print(f"  Device {i}: not available")

print("\n=== Test 2: MSMF backend (Microsoft Media Foundation) ===")
for i in range(4):
    try:
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  Device {i}: FOUND ({w}x{h})")
            cap.release()
        else:
            print(f"  Device {i}: not available")
    except Exception as e:
        print(f"  Device {i}: ERROR {e}")

print("\n=== Test 3: Network - retry after firewall rule ===")
import socket, urllib.request
IP, PORT = "10.183.236.203", 4747
try:
    s = socket.create_connection((IP, PORT), timeout=5)
    s.close()
    print(f"  TCP {IP}:{PORT} -> REACHABLE NOW")
    # Try URL
    for path in ["/mjpegfeed", "/video", "/"]:
        try:
            r = urllib.request.urlopen(f"http://{IP}:{PORT}{path}", timeout=4)
            print(f"  http://{IP}:{PORT}{path} -> {r.status} {r.headers.get('Content-Type','?')}")
        except Exception as e:
            print(f"  http://{IP}:{PORT}{path} -> {e}")
except Exception as e:
    print(f"  TCP {IP}:{PORT} -> STILL BLOCKED: {e}")

print("\n=== Test 4: Device 1 force open with MSMF ===")
try:
    cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
    opened = cap.isOpened()
    print(f"  Device 1 MSMF: {'OPENED' if opened else 'FAILED'}")
    if opened:
        ret, frame = cap.read()
        print(f"  Frame read: {'OK shape=' + str(frame.shape) if ret else 'FAILED'}")
    cap.release()
except Exception as e:
    print(f"  Device 1 MSMF: ERROR {e}")
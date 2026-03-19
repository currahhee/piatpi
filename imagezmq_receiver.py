#!/usr/bin/env python3
"""
imagezmq Receiver / Viewer
============================
Connects to a Pi running pi_camera_stream2.py and displays the stream.

Install (on your PC):
    pip install imagezmq opencv-python pyzmq

Usage:
    python imagezmq_receiver.py
    Press 'q' in the window to quit.
"""

import cv2
import numpy as np
import imagezmq

# ---------------------------------------------------------------------------
# Configuration — update PI_IP to match your Raspberry Pi's address
# ---------------------------------------------------------------------------
PI_IP = "192.168.1.14"       # Raspberry Pi IP
PI_PORT = 5555               # Must match IMAGEZMQ_PORT in pi_camera_stream2.py

# ---------------------------------------------------------------------------
# Connect to the Pi's imagezmq sender (PUB/SUB mode)
# ---------------------------------------------------------------------------
image_hub = imagezmq.ImageHub(
    open_port=f"tcp://{PI_IP}:{PI_PORT}",
    REQ_REP=False
)

print(f"[INFO] Subscribed to tcp://{PI_IP}:{PI_PORT}")
print("[INFO] Waiting for frames... Press 'q' in the window to quit.")

while True:
    # Receive JPEG-compressed frame
    name, jpg_buffer = image_hub.recv_jpg()
    frame = cv2.imdecode(
        np.frombuffer(jpg_buffer, dtype="uint8"), cv2.IMREAD_COLOR
    )

    if frame is None:
        continue

    cv2.imshow(name, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
print("[INFO] Receiver stopped.")

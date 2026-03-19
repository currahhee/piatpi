#!/usr/bin/env python3
"""
Raspberry Pi Camera Streamer (imagezmq version)
=================================================
Streams frames from a Pi Camera Module via:
  1. TCP socket  - to a specific remote server (same as v1)
  2. imagezmq    - viewable by any device on the local network (replaces RTSP)

Installation (on Raspberry Pi):
    sudo apt update
    sudo apt install -y python3-picamera2 python3-opencv python3-numpy python3-zmq
    pip3 install imagezmq

On the receiving PC (viewer):
    pip install imagezmq opencv-python pyzmq

Usage:
    # On the Pi:
    python3 pi_camera_stream2.py

    # On any PC to view the imagezmq stream, run the included receiver or:
    #   python3 imagezmq_receiver.py
    # (see bottom of this file for a minimal receiver example)
"""

import socket
import struct
import time
import signal
import sys
import threading

import cv2
import numpy as np
from picamera2 import Picamera2
import imagezmq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVER_IP = "192.168.1.67"        # Remote TCP server IP (your PC)
SERVER_PORT = 5000                # Remote TCP server port

# imagezmq settings (replaces RTSP)
# PUB/SUB mode: non-blocking, any number of viewers can connect/disconnect freely
# REQ/REP mode: blocking, one viewer at a time, guaranteed delivery
IMAGEZMQ_PORT = 5555              # Port for imagezmq stream
IMAGEZMQ_REQ_REP = False          # False = PUB/SUB (recommended), True = REQ/REP

RESOLUTION = (1920, 1080)
TARGET_FPS = 60
JPEG_QUALITY = 80                 # 1-100 for TCP stream & imagezmq jpg mode
RECONNECT_DELAY = 2               # seconds before TCP reconnect

ENABLE_TCP = True                 # set False to disable TCP streaming
ENABLE_IMAGEZMQ = True            # set False to disable imagezmq streaming

# Whether to send JPEG-compressed frames over imagezmq (lower bandwidth)
# or raw numpy arrays (lower latency, higher bandwidth)
IMAGEZMQ_SEND_JPG = True

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
running = True
picam2 = None
latest_frame_lock = threading.Lock()
latest_frame = None               # shared BGR frame for consumers


def signal_handler(sig, frame):
    """Handle Ctrl+C / SIGTERM for a clean shutdown."""
    global running
    print("\n[INFO] Shutting down...")
    running = False


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def create_camera():
    """Initialise the Pi camera and start capturing."""
    cam = Picamera2()
    config = cam.create_preview_configuration(
        main={"size": RESOLUTION, "format": "RGB888"}
    )
    cam.configure(config)

    cam.start()
    time.sleep(2)  # give AWB time to settle
    print(f"[INFO] Camera started at {RESOLUTION[0]}x{RESOLUTION[1]}")
    return cam


# ---------------------------------------------------------------------------
# TCP streaming (identical to v1)
# ---------------------------------------------------------------------------
def connect_to_server():
    """Create a TCP connection to the server. Returns the socket or None."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((SERVER_IP, SERVER_PORT))
        print(f"[INFO] TCP connected to {SERVER_IP}:{SERVER_PORT}")
        return sock
    except OSError as e:
        print(f"[WARN] TCP connection failed: {e}")
        return None


def send_frame(sock, frame_bgr):
    """
    Encode a frame as JPEG and send it over the socket.

    Protocol:
        1. Encode the frame to JPEG bytes
        2. Pack the frame size as a 4-byte big-endian unsigned long
        3. Send the size header followed by the JPEG data
    """
    ret, jpeg = cv2.imencode(
        ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ret:
        return False

    data = jpeg.tobytes()
    try:
        sock.sendall(struct.pack(">L", len(data)) + data)
        return True
    except OSError:
        return False


def tcp_stream_loop():
    """TCP streaming loop with automatic reconnection (runs in its own thread)."""
    frame_interval = 1.0 / TARGET_FPS

    while running:
        sock = connect_to_server()
        if sock is None:
            if not running:
                break
            print(f"[INFO] TCP retrying in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)
            continue

        try:
            while running:
                t_start = time.monotonic()

                with latest_frame_lock:
                    frame_bgr = latest_frame

                if frame_bgr is None:
                    time.sleep(0.01)
                    continue

                if not send_frame(sock, frame_bgr):
                    print("[WARN] TCP send failed – connection lost")
                    break

                # Throttle to target FPS
                elapsed = time.monotonic() - t_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"[WARN] TCP connection dropped: {e}")
        finally:
            sock.close()

        if running:
            print(f"[INFO] TCP reconnecting in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)


# ---------------------------------------------------------------------------
# imagezmq streaming (replaces RTSP)
# ---------------------------------------------------------------------------
def imagezmq_stream_loop():
    """
    Stream frames via imagezmq. Runs in its own thread.

    PUB/SUB mode (default):
      - Non-blocking – frames are published whether or not a viewer is connected
      - Any number of viewers can subscribe and drop off freely
      - Viewer connects with: ImageHub('tcp://<pi-ip>:5555', REQ_REP=False)

    REQ/REP mode:
      - Blocking – waits for viewer to acknowledge each frame
      - Guaranteed delivery, but only one viewer at a time
      - Viewer connects with: ImageHub('tcp://<pi-ip>:5555')
    """
    # Get the Pi's hostname to tag frames
    pi_name = socket.gethostname()

    # Build the bind address for the sender
    # In PUB/SUB mode the sender binds; viewers connect to it
    bind_addr = f"tcp://*:{IMAGEZMQ_PORT}"

    sender = imagezmq.ImageSender(
        connect_to=bind_addr,
        REQ_REP=IMAGEZMQ_REQ_REP
    )

    mode_str = "REQ/REP" if IMAGEZMQ_REQ_REP else "PUB/SUB"
    print(f"[INFO] imagezmq sender started on port {IMAGEZMQ_PORT} ({mode_str} mode)")
    print(f"[INFO] Viewers connect to: tcp://<pi-ip>:{IMAGEZMQ_PORT}")

    frame_interval = 1.0 / TARGET_FPS

    try:
        while running:
            t_start = time.monotonic()

            with latest_frame_lock:
                frame_bgr = latest_frame

            if frame_bgr is None:
                time.sleep(0.01)
                continue

            # Send either JPEG-compressed or raw frame
            if IMAGEZMQ_SEND_JPG:
                ret, jpeg = cv2.imencode(
                    ".jpg", frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if ret:
                    sender.send_jpg(pi_name, jpeg)
            else:
                sender.send_image(pi_name, frame_bgr)

            # Throttle to target FPS
            elapsed = time.monotonic() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"[WARN] imagezmq error: {e}")
    finally:
        sender.close()
        print("[INFO] imagezmq sender closed")


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------
def capture_loop():
    """Capture frames and store them for TCP and imagezmq consumers."""
    global picam2, latest_frame

    picam2 = create_camera()
    frame_interval = 1.0 / TARGET_FPS

    while running:
        t_start = time.monotonic()

        # Capture frame directly — picamera2 RGB888 outputs BGR-ordered
        # arrays on most Pi builds, which is what OpenCV expects
        frame_bgr = picam2.capture_array()

        with latest_frame_lock:
            latest_frame = frame_bgr

        elapsed = time.monotonic() - t_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start imagezmq sender in a background thread
        if ENABLE_IMAGEZMQ:
            zmq_thread = threading.Thread(target=imagezmq_stream_loop, daemon=True)
            zmq_thread.start()

        # Start TCP sender in a background thread
        if ENABLE_TCP:
            tcp_thread = threading.Thread(target=tcp_stream_loop, daemon=True)
            tcp_thread.start()

        # Capture loop runs on the main thread
        capture_loop()

    finally:
        if picam2 is not None:
            picam2.stop()
            print("[INFO] Camera stopped")
        print("[INFO] Exited cleanly")


if __name__ == "__main__":
    main()


# ===========================================================================
# RECEIVER EXAMPLE (run this on your PC to view the stream)
# ===========================================================================
# Save the code below as "imagezmq_receiver.py" on your PC and run it.
#
# --- imagezmq_receiver.py ---
#
# """
# imagezmq Receiver / Viewer
# Connects to a Pi running pi_camera_stream2.py and displays the stream.
#
# Install:
#     pip install imagezmq opencv-python pyzmq
#
# Usage:
#     python imagezmq_receiver.py
# """
#
# import cv2
# import imagezmq
#
# PI_IP = "192.168.1.XXX"  # <-- Replace with your Pi's IP
# PI_PORT = 5555
#
# # --- PUB/SUB mode (matches default sender config) ---
# image_hub = imagezmq.ImageHub(
#     open_port=f"tcp://{PI_IP}:{PI_PORT}",
#     REQ_REP=False
# )
#
# print(f"[INFO] Subscribed to tcp://{PI_IP}:{PI_PORT}")
# print("[INFO] Press 'q' in the window to quit")
#
# while True:
#     # Use recv_jpg() if sender uses IMAGEZMQ_SEND_JPG = True
#     name, jpg_buffer = image_hub.recv_jpg()
#     frame = cv2.imdecode(
#         np.frombuffer(jpg_buffer, dtype="uint8"), cv2.IMREAD_COLOR
#     )
#
#     # # Use recv_image() if sender uses IMAGEZMQ_SEND_JPG = False
#     # name, frame = image_hub.recv_image()
#
#     cv2.imshow(name, frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cv2.destroyAllWindows()
# ---------------------------------------------------------------------------

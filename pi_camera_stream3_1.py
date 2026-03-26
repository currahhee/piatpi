#!/usr/bin/env python3
"""
Raspberry Pi Camera Streamer v3.1 — MEF + imagezmq
====================================================
Streams frames from a Pi Camera Module via:
  1. TCP socket  — to a specific remote server
  2. imagezmq    — viewable by any device on the local network (PUB/SUB)

Multi-Exposure Bracketing runs in a **separate thread**. A camera_lock
ensures picamera2 is never accessed concurrently — the stream pauses
briefly (~2-4s) during bracket capture for consistent frames.

Changes from v3:
  - RGB888 kept (correct on tested Pi 4B — see comment in create_camera())
  - camera_lock prevents concurrent picamera2 access between threads
  - Settle frames use capture_request()/release() (no wasted numpy alloc)
  - imencode failures no longer spin the CPU
  - Empty bracket sets don't write orphan metadata JSON files
  - Stale frame skip: stream threads don't re-encode the same frozen frame
    during bracket capture (~120 wasted encodes eliminated per bracket set)

Inherited from v3 (bug-fixed):
  - Bracket thread with bracket_lock (no overlap)
  - Nested try/finally guaranteeing AE restore on any exception
  - Atomic capture_request() for frame + metadata
  - TCP socket closed on failed connect
  - Label/speed length mismatch warning
  - bracket_loop survives exceptions

Installation (on Raspberry Pi):
    sudo apt update
    sudo apt install -y python3-picamera2 python3-opencv python3-zmq
    pip3 install imagezmq

On the receiving PC (viewer):
    pip install imagezmq opencv-python pyzmq numpy

Usage:
    python3 pi_camera_stream3_1.py
"""

import os
import json
import socket
import struct
import time
import signal
import threading
from datetime import datetime

import cv2
from picamera2 import Picamera2
import imagezmq

try:
    from libcamera import controls as libcamera_controls
except ImportError:
    libcamera_controls = None

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# --- Network ---
SERVER_IP = "192.168.1.67"
SERVER_PORT = 5000
IMAGEZMQ_PORT = 5555
IMAGEZMQ_REQ_REP = False          # False = PUB/SUB (recommended)

# --- Camera ---
RESOLUTION = (1920, 1080)
TARGET_FPS = 30
JPEG_QUALITY = 90

# --- Noise reduction ---
CAMERA_NOISE_REDUCTION_MODE = "high_quality"   # off | fast | high_quality
CAMERA_SHARPNESS = 0.7                         # lower than 1.0 to avoid sharpening noise
STREAM_DENOISE_FILTER = "median"              # off | median | gaussian | bilateral
STREAM_MEDIAN_KERNEL = 3
STREAM_GAUSSIAN_KERNEL = 3
STREAM_BILATERAL_DIAMETER = 5
STREAM_BILATERAL_SIGMA_COLOR = 30
STREAM_BILATERAL_SIGMA_SPACE = 30

# --- Stream toggles ---
ENABLE_TCP = False
ENABLE_IMAGEZMQ = True
IMAGEZMQ_SEND_JPG = True

# --- TCP reconnection ---
RECONNECT_DELAY = 2               # seconds

# --- Multi-Exposure Bracketing ---
ENABLE_BRACKET = True
BRACKET_SET_INTERVAL = 15         # seconds between bracket sets
BRACKET_SAVE_DIR = os.path.expanduser("~/mef_captures")
BRACKET_JPEG_QUALITY = 95
BRACKET_SETTLE_FRAMES = 4        # frames to discard after exposure change
BRACKET_POST_SETTLE = 0.3        # short pause (s) after settle, before capture
SAVE_METADATA_JSON = True         # write a .json sidecar per bracket set

# Bracket mode: 'fixed' or 'relative'
#   'fixed'    — use BRACKET_SHUTTER_SPEEDS directly (µs)
#   'relative' — multiply auto-exposure by BRACKET_EV_MULTIPLIERS
BRACKET_MODE = "fixed"

# Fixed mode: explicit shutter speeds in µs
BRACKET_SHUTTER_SPEEDS = [
    5000,       # UNDER  — fast shutter, tames specular highlights  (~1/200s)
    33000,      # NORMAL — baseline                                  (~1/30s)
    200000,     # OVER   — slow shutter, reveals shadow detail        (~1/5s)
]
BRACKET_LABELS = ["under", "normal", "over"]

# Relative mode: multiply current auto-exposure by these factors
#   <1.0 = darker, 1.0 = auto, >1.0 = brighter
BRACKET_EV_MULTIPLIERS = [0.15, 1.0, 6.0]

# Hardware limits for the Pi Camera (IMX219/IMX477/IMX708)
SHUTTER_MIN_US = 100
SHUTTER_MAX_US = 2_000_000        # 2s — safe for all Pi Camera modules


# ═══════════════════════════════════════════════════════════════════════════
# Globals & Locks
# ═══════════════════════════════════════════════════════════════════════════
running = True
picam2 = None
latest_frame_lock = threading.Lock()
latest_frame = None
camera_lock = threading.Lock()    # serialises all picamera2 access
bracket_lock = threading.Lock()   # prevents overlapping bracket captures
bracket_count = 0
stream_denoise_warning_shown = False


def signal_handler(sig, frame):
    global running
    print("\n[INFO] Shutting down...")
    running = False


def get_camera_noise_reduction_control():
    mode = CAMERA_NOISE_REDUCTION_MODE.strip().lower()

    if libcamera_controls is not None:
        try:
            enum = libcamera_controls.draft.NoiseReductionModeEnum
            return {
                "off": enum.Off,
                "fast": enum.Fast,
                "high_quality": enum.HighQuality,
            }.get(mode, enum.HighQuality)
        except AttributeError:
            pass

    return {
        "off": 0,
        "fast": 1,
        "high_quality": 2,
    }.get(mode, 2)


def odd_kernel_size(value, fallback):
    size = int(value) if value else fallback
    size = max(1, size)
    if size % 2 == 0:
        size += 1
    return size


def get_frame_duration_limits():
    frame_duration_us = max(1, round(1_000_000 / TARGET_FPS))
    return (frame_duration_us, frame_duration_us)


def reduce_stream_noise(frame_bgr):
    global stream_denoise_warning_shown

    mode = STREAM_DENOISE_FILTER.strip().lower()

    if mode == "off":
        return frame_bgr
    if mode == "median":
        return cv2.medianBlur(
            frame_bgr,
            odd_kernel_size(STREAM_MEDIAN_KERNEL, 3),
        )
    if mode == "gaussian":
        kernel = odd_kernel_size(STREAM_GAUSSIAN_KERNEL, 3)
        return cv2.GaussianBlur(frame_bgr, (kernel, kernel), 0)
    if mode == "bilateral":
        return cv2.bilateralFilter(
            frame_bgr,
            max(1, int(STREAM_BILATERAL_DIAMETER)),
            max(1, int(STREAM_BILATERAL_SIGMA_COLOR)),
            max(1, int(STREAM_BILATERAL_SIGMA_SPACE)),
        )

    if not stream_denoise_warning_shown:
        print(f"[CAM] Unknown STREAM_DENOISE_FILTER='{STREAM_DENOISE_FILTER}', disabling software denoise")
        stream_denoise_warning_shown = True
    return frame_bgr


# ═══════════════════════════════════════════════════════════════════════════
# Camera
# ═══════════════════════════════════════════════════════════════════════════
def create_camera():
    cam = Picamera2()
    # RGB888 — on most Pi 4B setups with picamera2, this label actually
    # outputs BGR-ordered arrays in memory, which OpenCV expects natively.
    # Confirmed correct by existing test captures (no red/blue swap).
    # If colors look wrong on a different Pi/OS version, try "BGR888" here.
    initial_controls = {
        "FrameDurationLimits": get_frame_duration_limits(),
        "NoiseReductionMode": get_camera_noise_reduction_control(),
        "Sharpness": CAMERA_SHARPNESS,
    }
    supported_controls = getattr(cam, "camera_controls", {})
    initial_controls = {
        name: value for name, value in initial_controls.items()
        if name in supported_controls
    }

    config = cam.create_preview_configuration(
        main={"size": RESOLUTION, "format": "RGB888"}
    )
    cam.configure(config)
    cam.start()
    for control_name, control_value in initial_controls.items():
        try:
            cam.set_controls({control_name: control_value})
        except Exception as e:
            print(f"[CAM] WARNING: failed to apply {control_name}={control_value}: {e}")
    time.sleep(2)  # let AWB + AE converge
    print(f"[CAM] Started at {RESOLUTION[0]}x{RESOLUTION[1]} @ {TARGET_FPS} fps target (RGB888)")
    print(f"[CAM] Noise reduction: camera={CAMERA_NOISE_REDUCTION_MODE}, stream={STREAM_DENOISE_FILTER}")
    return cam


# ═══════════════════════════════════════════════════════════════════════════
# TCP Streaming
# ═══════════════════════════════════════════════════════════════════════════
def connect_to_server():
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((SERVER_IP, SERVER_PORT))
        sock.settimeout(None)
        print(f"[TCP] Connected to {SERVER_IP}:{SERVER_PORT}")
        return sock
    except OSError as e:
        print(f"[TCP] Connection failed: {e}")
        if sock is not None:
            sock.close()
        return None


def send_frame_tcp(sock, frame_bgr):
    ret, jpeg = cv2.imencode(
        ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ret:
        return None  # signal encode failure (distinct from send failure)
    data = jpeg.tobytes()
    try:
        sock.sendall(struct.pack(">L", len(data)) + data)
        return True
    except OSError:
        return False


def tcp_stream_loop():
    frame_interval = 1.0 / TARGET_FPS

    while running:
        sock = connect_to_server()
        if sock is None:
            if not running:
                break
            time.sleep(RECONNECT_DELAY)
            continue

        last_sent = None  # track last sent frame reference

        try:
            while running:
                t0 = time.monotonic()

                with latest_frame_lock:
                    frame = latest_frame

                if frame is None or frame is last_sent:
                    # No frame yet, or same frame as last send — don't re-encode
                    time.sleep(0.01)
                    continue

                result = send_frame_tcp(sock, frame)
                if result is None:
                    # imencode failed — don't spin
                    time.sleep(0.01)
                    continue
                if result is False:
                    print("[TCP] Send failed — connection lost")
                    break

                last_sent = frame

                elapsed = time.monotonic() - t0
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        except OSError as e:
            print(f"[TCP] Dropped: {e}")
        finally:
            sock.close()

        if running:
            print(f"[TCP] Reconnecting in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)


# ═══════════════════════════════════════════════════════════════════════════
# imagezmq Streaming
# ═══════════════════════════════════════════════════════════════════════════
def imagezmq_stream_loop():
    pi_name = socket.gethostname()

    # In PUB/SUB mode, ImageSender calls zmq_socket.bind(address).
    # Must bind to a LOCAL address — the receiver connects to tcp://<pi-ip>:PORT.
    bind_addr = f"tcp://*:{IMAGEZMQ_PORT}"
    mode_str = "PUB/SUB" if not IMAGEZMQ_REQ_REP else "REQ/REP"

    while running:
        sender = None
        try:
            sender = imagezmq.ImageSender(
                connect_to=bind_addr,
                REQ_REP=IMAGEZMQ_REQ_REP
            )
            print(f"[ZMQ] Sender on port {IMAGEZMQ_PORT} ({mode_str})")

            frame_interval = 1.0 / TARGET_FPS
            last_sent = None

            while running:
                t0 = time.monotonic()

                with latest_frame_lock:
                    frame = latest_frame

                if frame is None or frame is last_sent:
                    time.sleep(0.01)
                    continue

                if IMAGEZMQ_SEND_JPG:
                    ret, jpeg = cv2.imencode(
                        ".jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    )
                    if ret:
                        sender.send_jpg(pi_name, jpeg)
                    else:
                        time.sleep(0.01)
                        continue
                else:
                    sender.send_image(pi_name, frame)

                last_sent = frame

                elapsed = time.monotonic() - t0
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        except Exception as e:
            print(f"[ZMQ] Error: {e}")
        finally:
            if sender is not None:
                try:
                    sender.close()
                except Exception:
                    pass

        if running:
            print(f"[ZMQ] Restarting in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)

    print("[ZMQ] Sender closed")


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Exposure Bracketed Capture (runs in its own thread)
# ═══════════════════════════════════════════════════════════════════════════
def compute_shutter_speeds(cam):
    """
    Return a list of (label, shutter_µs) tuples based on the bracket mode.

    'fixed'    → use BRACKET_SHUTTER_SPEEDS as-is
    'relative' → read current auto-exposure and scale by BRACKET_EV_MULTIPLIERS
    """
    metadata = cam.capture_metadata()
    auto_exposure = metadata.get("ExposureTime", 33000)
    auto_gain = metadata.get("AnalogueGain", 1.0)

    if BRACKET_MODE == "relative":
        speeds = []
        for mult in BRACKET_EV_MULTIPLIERS:
            us = int(auto_exposure * mult)
            us = max(SHUTTER_MIN_US, min(us, SHUTTER_MAX_US))
            speeds.append(us)
        if len(BRACKET_LABELS) != len(BRACKET_EV_MULTIPLIERS):
            print(f"[MEF] WARNING: BRACKET_LABELS ({len(BRACKET_LABELS)}) and "
                  f"BRACKET_EV_MULTIPLIERS ({len(BRACKET_EV_MULTIPLIERS)}) length mismatch")
    else:
        speeds = [
            max(SHUTTER_MIN_US, min(s, SHUTTER_MAX_US))
            for s in BRACKET_SHUTTER_SPEEDS
        ]
        if len(BRACKET_LABELS) != len(BRACKET_SHUTTER_SPEEDS):
            print(f"[MEF] WARNING: BRACKET_LABELS ({len(BRACKET_LABELS)}) and "
                  f"BRACKET_SHUTTER_SPEEDS ({len(BRACKET_SHUTTER_SPEEDS)}) length mismatch")

    pairs = list(zip(BRACKET_LABELS, speeds))
    return pairs, auto_exposure, auto_gain


def capture_bracket_set(cam):
    """
    Capture a set of bracketed images for MEF.

    Acquires both bracket_lock (no overlapping sets) and camera_lock
    (exclusive picamera2 access). The main capture loop blocks on
    camera_lock during this time — the stream pauses briefly but
    frames stay consistent.
    """
    global bracket_count

    if not bracket_lock.acquire(blocking=False):
        print("[MEF] Skipped — previous bracket still in progress")
        return

    # Hold camera_lock for the entire bracket to prevent the main
    # capture loop from interleaving picamera2 calls.
    if not camera_lock.acquire(timeout=10):
        print("[MEF] WARNING: camera_lock timeout after 10s — skipping bracket")
        bracket_lock.release()
        return

    try:
        pairs, auto_exp, auto_gain = compute_shutter_speeds(cam)
        bracket_count += 1
        set_id = f"set{bracket_count:04d}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"[MEF] #{bracket_count} starting — auto: {auto_exp} µs, gain: {auto_gain:.1f}")
        t_set_start = time.monotonic()

        metadata_entries = []
        saved_count = 0

        # Disable AE, lock gain at current auto level — restore is guaranteed
        # by the try/finally below, which now covers the disable call too.
        try:
            cam.set_controls({"AeEnable": False, "AnalogueGain": auto_gain})
            time.sleep(0.05)

            for i, (label, shutter_us) in enumerate(pairs):
                if not running:
                    break

                # Set exposure
                cam.set_controls({"ExposureTime": shutter_us})

                # Settle: discard frames so the sensor applies the new exposure.
                # Use capture_request()/release() — lighter than capture_array()
                # because it skips the numpy array allocation (~2.7 MB per frame).
                settle = max(1, BRACKET_SETTLE_FRAMES if shutter_us < 100_000 else 2)
                for _ in range(settle):
                    req = cam.capture_request()
                    req.release()

                # Brief pause for stability
                if BRACKET_POST_SETTLE > 0:
                    time.sleep(BRACKET_POST_SETTLE)

                # Capture — capture_request() returns frame + metadata atomically
                request = cam.capture_request()
                frame = request.make_array("main")
                actual_meta = request.get_metadata()
                request.release()

                # Save image
                filename = f"{set_id}_{timestamp}_{label}_exp{shutter_us}us.jpg"
                filepath = os.path.join(BRACKET_SAVE_DIR, filename)
                ok = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, BRACKET_JPEG_QUALITY])

                actual_exp = actual_meta.get("ExposureTime", shutter_us)
                actual_gain = actual_meta.get("AnalogueGain", auto_gain)

                if not ok:
                    print(f"[MEF]   [{i+1}/{len(pairs)}] {label}: FAILED to write {filename}")
                    continue

                saved_count += 1
                print(f"[MEF]   [{i+1}/{len(pairs)}] {label}: {filename} "
                      f"(requested={shutter_us}µs, actual={actual_exp}µs, gain={actual_gain:.1f})")

                metadata_entries.append({
                    "label": label,
                    "filename": filename,
                    "requested_exposure_us": shutter_us,
                    "actual_exposure_us": actual_exp,
                    "actual_gain": round(actual_gain, 2),
                })

        finally:
            # ALWAYS restore auto-exposure, even if an exception occurred
            try:
                cam.set_controls({
                    "ExposureTime": auto_exp,
                    "AnalogueGain": auto_gain,
                })
                time.sleep(0.05)
                cam.set_controls({"AeEnable": True})
            except Exception as ae_err:
                print(f"[MEF] WARNING: failed to restore AE: {ae_err}")

        elapsed = time.monotonic() - t_set_start
        print(f"[MEF] #{bracket_count} done — {saved_count}/{len(pairs)} images in {elapsed:.1f}s")

        # Save metadata sidecar (only if we actually captured something)
        if SAVE_METADATA_JSON and metadata_entries:
            meta = {
                "set_id": set_id,
                "timestamp": timestamp,
                "bracket_mode": BRACKET_MODE,
                "auto_exposure_us": auto_exp,
                "auto_gain": round(auto_gain, 2),
                "resolution": list(RESOLUTION),
                "capture_duration_s": round(elapsed, 2),
                "images": metadata_entries,
            }
            meta_path = os.path.join(BRACKET_SAVE_DIR, f"{set_id}_{timestamp}_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    finally:
        camera_lock.release()
        bracket_lock.release()


def bracket_loop():
    """
    Periodically trigger bracket captures.
    Runs in its own daemon thread — does NOT block the main capture loop.
    """
    # Wait for camera to be ready
    while running and picam2 is None:
        time.sleep(0.5)

    if not running:
        return

    # Initial delay to let AE converge after camera start
    time.sleep(3)

    os.makedirs(BRACKET_SAVE_DIR, exist_ok=True)

    while running:
        if picam2 is not None:
            try:
                capture_bracket_set(picam2)
            except Exception as e:
                print(f"[MEF] Bracket capture failed: {e}")
                print("[MEF] Will retry next interval")

        # Interruptible sleep
        for _ in range(int(BRACKET_SET_INTERVAL * 10)):
            if not running:
                return
            time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════════
# Main Capture Loop
# ═══════════════════════════════════════════════════════════════════════════
def capture_loop():
    """Continuously capture frames for the stream consumers."""
    global picam2, latest_frame

    picam2 = create_camera()
    frame_interval = 1.0 / TARGET_FPS

    while running:
        t0 = time.monotonic()

        # camera_lock blocks here during bracket capture (~2-4s).
        # This is intentional — the stream pauses but frames stay
        # consistent (no half-applied exposure settings).
        try:
            with camera_lock:
                if not running:
                    break
                frame_bgr = picam2.capture_array()

            frame_bgr = reduce_stream_noise(frame_bgr)

            with latest_frame_lock:
                latest_frame = frame_bgr

        except Exception as e:
            print(f"[CAM] Capture error: {e}")
            time.sleep(0.1)  # back off briefly, then retry
            continue

        elapsed = time.monotonic() - t0
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print("  Pi Camera Streamer v3.1 — MEF + imagezmq")
    print("=" * 60)
    print(f"  Resolution:    {RESOLUTION[0]}x{RESOLUTION[1]}")
    print(f"  Target FPS:    {TARGET_FPS}")
    print(f"  Pixel format:  RGB888")
    print(f"  Camera NR:     {CAMERA_NOISE_REDUCTION_MODE}")
    print(f"  Stream filter: {STREAM_DENOISE_FILTER}")
    print(f"  TCP stream:    {'ON' if ENABLE_TCP else 'OFF'}"
          + (f"  → {SERVER_IP}:{SERVER_PORT}" if ENABLE_TCP else ""))
    print(f"  imagezmq:      {'ON' if ENABLE_IMAGEZMQ else 'OFF'}"
          + (f"  → port {IMAGEZMQ_PORT} ({'PUB/SUB' if not IMAGEZMQ_REQ_REP else 'REQ/REP'})" if ENABLE_IMAGEZMQ else ""))
    print(f"  MEF bracket:   {'ON' if ENABLE_BRACKET else 'OFF'}"
          + (f"  → every {BRACKET_SET_INTERVAL}s, mode={BRACKET_MODE}" if ENABLE_BRACKET else ""))
    if ENABLE_BRACKET:
        if BRACKET_MODE == "fixed":
            speeds = ", ".join(f"{s}µs" for s in BRACKET_SHUTTER_SPEEDS)
            print(f"  Shutter speeds: [{speeds}]")
        else:
            mults = ", ".join(f"{m}x" for m in BRACKET_EV_MULTIPLIERS)
            print(f"  EV multipliers: [{mults}]")
        print(f"  Save dir:      {BRACKET_SAVE_DIR}")
    print("=" * 60)

    threads = []

    try:
        if ENABLE_IMAGEZMQ:
            t = threading.Thread(target=imagezmq_stream_loop, daemon=True, name="zmq")
            t.start()
            threads.append(t)

        if ENABLE_TCP:
            t = threading.Thread(target=tcp_stream_loop, daemon=True, name="tcp")
            t.start()
            threads.append(t)

        if ENABLE_BRACKET:
            t = threading.Thread(target=bracket_loop, daemon=True, name="bracket")
            t.start()
            threads.append(t)

        # Main thread runs the capture loop
        capture_loop()

    finally:
        if picam2 is not None:
            picam2.stop()
            print("[CAM] Stopped")

        # Give daemon threads a moment to clean up
        for t in threads:
            t.join(timeout=2)

        print("[INFO] Exited cleanly")


if __name__ == "__main__":
    main() 

#!/usr/bin/env python3
"""
imagezmq Receiver v3 — MEF Pipeline Ready
===========================================
Connects to a Pi running pi_camera_stream3.py and displays the stream.
Designed as the server-side entry point for the full pipeline:

    Pi Camera → imagezmq → [this receiver] → MEF Fusion → Defect Detection

Features over v1:
  - Real-time FPS, decode latency & bandwidth overlay
  - Optional frame saving to disk (press 's' to toggle)
  - Keyboard controls (fullscreen, pause, reset stats)
  - Hook point for MEF fusion + defect detection integration
  - CLI arguments for IP, port, headless mode
  - Graceful error handling on receive failures

Install (on your PC):
    pip install imagezmq opencv-python pyzmq numpy

Usage:
    python imagezmq_receiver3.py
    
    Keyboard controls:
      q     — quit
      s     — toggle saving frames to disk
      f     — toggle fullscreen
      r     — reset FPS counter
      space — pause/resume display (frames still received)

"""

import os
import time
import argparse
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import imagezmq


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════
PI_IP = "192.168.1.14"
PI_PORT = 5555

# Display
WINDOW_NAME = "MEF Stream Receiver"
SHOW_OSD = True                   # on-screen display (FPS, latency, etc.)
OSD_FONT = cv2.FONT_HERSHEY_SIMPLEX
OSD_SCALE = 0.55
OSD_COLOR = (0, 255, 0)          # green
OSD_THICKNESS = 1

# Frame saving
SAVE_DIR = os.path.expanduser("~/mef_received")
SAVE_QUALITY = 95

# Performance tracking
FPS_WINDOW = 60                   # number of frames for rolling FPS average


# ═══════════════════════════════════════════════════════════════════════════
# Argument Parser
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="imagezmq receiver for MEF pipeline")
    parser.add_argument("--ip", default=PI_IP, help=f"Pi IP address (default: {PI_IP})")
    parser.add_argument("--port", type=int, default=PI_PORT, help=f"Pi port (default: {PI_PORT})")
    parser.add_argument("--no-display", action="store_true", help="Headless mode (no cv2 window)")
    parser.add_argument("--save", action="store_true", help="Start with frame saving enabled")
    parser.add_argument("--save-dir", default=SAVE_DIR, help=f"Save directory (default: {SAVE_DIR})")
    parser.add_argument("--req-rep", action="store_true", help="Use REQ/REP instead of PUB/SUB")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Stats Tracker
# ═══════════════════════════════════════════════════════════════════════════
class StreamStats:
    """Track FPS, latency, frame count, and drops."""

    def __init__(self, window=FPS_WINDOW):
        self.window = window
        self.timestamps = deque(maxlen=window)
        self.frame_count = 0
        self.bytes_received = 0
        self.start_time = time.monotonic()
        self.decode_times = deque(maxlen=window)

    def tick(self, jpg_size=0, decode_ms=0.0):
        now = time.monotonic()
        self.timestamps.append(now)
        self.frame_count += 1
        self.bytes_received += jpg_size
        self.decode_times.append(decode_ms)

    @property
    def fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        span = self.timestamps[-1] - self.timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / span

    @property
    def avg_decode_ms(self):
        if not self.decode_times:
            return 0.0
        return sum(self.decode_times) / len(self.decode_times)

    @property
    def bandwidth_kbps(self):
        elapsed = time.monotonic() - self.start_time
        if elapsed <= 0:
            return 0.0
        return (self.bytes_received * 8) / elapsed / 1000

    def reset(self):
        self.timestamps.clear()
        self.frame_count = 0
        self.bytes_received = 0
        self.start_time = time.monotonic()
        self.decode_times.clear()


# ═══════════════════════════════════════════════════════════════════════════
# OSD (On-Screen Display)
# ═══════════════════════════════════════════════════════════════════════════
def draw_osd(frame, stats, pi_name, saving=False, paused=False):
    """Draw performance info overlay on the frame."""
    h, w = frame.shape[:2]
    lines = [
        f"Source: {pi_name}",
        f"FPS: {stats.fps:.1f}  |  Decode: {stats.avg_decode_ms:.1f}ms",
        f"Bandwidth: {stats.bandwidth_kbps:.0f} kbps  |  Frames: {stats.frame_count}",
        f"Resolution: {w}x{h}",
    ]

    if saving:
        lines.append("[REC] Saving frames to disk")
    if paused:
        lines.append("[PAUSED]")

    y = 25
    for line in lines:
        # Draw shadow for readability
        cv2.putText(frame, line, (11, y + 1), OSD_FONT, OSD_SCALE,
                    (0, 0, 0), OSD_THICKNESS + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), OSD_FONT, OSD_SCALE,
                    OSD_COLOR, OSD_THICKNESS, cv2.LINE_AA)
        y += 22

    return frame


# ═══════════════════════════════════════════════════════════════════════════
# Frame Processing Hook
# ═══════════════════════════════════════════════════════════════════════════
def process_frame(frame, pi_name):
    """
    Hook for MEF / defect detection pipeline integration.
    
    This is where you'll plug in:
      1. Frame buffering (collect 3 brackets into a set)
      2. MEF fusion (MEFLUT / MEF-Net via TensorRT)
      3. Defect detection (PatchCore / EfficientAD / YOLO)
      4. Result annotation / alerting

    For now, just returns the frame unchanged.
    
    Args:
        frame: BGR numpy array from the Pi
        pi_name: hostname string identifying which Pi sent this frame
    
    Returns:
        processed frame (BGR numpy array)
    """
    # ── Future pipeline placeholder ──
    # if is_bracket_frame(frame, metadata):
    #     bracket_buffer.append(frame)
    #     if len(bracket_buffer) == 3:
    #         fused = mef_model.fuse(bracket_buffer)
    #         defects = detector.predict(fused)
    #         frame = annotate_defects(fused, defects)
    #         bracket_buffer.clear()
    
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# Main Receiver Loop
# ═══════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    pi_ip = args.ip
    pi_port = args.port
    headless = args.no_display
    saving = args.save
    save_dir = args.save_dir
    req_rep = args.req_rep

    # Connect to Pi
    connect_str = f"tcp://{pi_ip}:{pi_port}"
    print(f"[INFO] Connecting to {connect_str} ({'REQ/REP' if req_rep else 'PUB/SUB'})...")

    image_hub = imagezmq.ImageHub(
        open_port=connect_str,
        REQ_REP=req_rep
    )

    print(f"[INFO] Subscribed — waiting for frames...")
    if not headless:
        print("[INFO] Keys: q=quit  s=save  f=fullscreen  r=reset  space=pause")

    stats = StreamStats()
    paused = False
    fullscreen = False
    save_count = 0
    last_log_time = time.monotonic()

    if saving:
        os.makedirs(save_dir, exist_ok=True)

    if not headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    try:
        while True:
            # Receive frame
            try:
                name, jpg_buffer = image_hub.recv_jpg()
            except Exception as e:
                print(f"[WARN] Receive error: {e}")
                time.sleep(0.5)
                continue

            jpg_size = jpg_buffer.nbytes

            # Decode JPEG → BGR
            t_decode = time.monotonic()
            frame = cv2.imdecode(
                np.frombuffer(jpg_buffer, dtype="uint8"), cv2.IMREAD_COLOR
            )
            decode_ms = (time.monotonic() - t_decode) * 1000

            if frame is None:
                print("[WARN] Failed to decode frame, skipping")
                continue

            stats.tick(jpg_size=jpg_size, decode_ms=decode_ms)

            # Run processing pipeline hook
            frame = process_frame(frame, name)

            # Save to disk if enabled
            if saving:
                save_count += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"frame_{ts}_{save_count:06d}.jpg"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_QUALITY])

            # Display
            if not headless and not paused:
                display = frame.copy()
                if SHOW_OSD:
                    display = draw_osd(display, stats, name,
                                       saving=saving, paused=paused)
                cv2.imshow(WINDOW_NAME, display)

            # Keyboard input
            if not headless:
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("[INFO] Quit requested")
                    break

                elif key == ord("s"):
                    saving = not saving
                    if saving:
                        os.makedirs(save_dir, exist_ok=True)
                    state = "ON" if saving else "OFF"
                    print(f"[INFO] Frame saving: {state}"
                          + (f" → {save_dir}" if saving else ""))

                elif key == ord("f"):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(
                            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN
                        )
                    else:
                        cv2.setWindowProperty(
                            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_NORMAL
                        )

                elif key == ord("r"):
                    stats.reset()
                    save_count = 0
                    print("[INFO] Stats reset")

                elif key == ord(" "):
                    paused = not paused
                    print(f"[INFO] Display {'paused' if paused else 'resumed'}")

            # Periodic console log (every 5 seconds)
            now = time.monotonic()
            if now - last_log_time >= 5.0 and stats.frame_count > 0:
                last_log_time = now
                print(f"[STATS] FPS: {stats.fps:.1f} | "
                      f"Decode: {stats.avg_decode_ms:.1f}ms | "
                      f"BW: {stats.bandwidth_kbps:.0f}kbps | "
                      f"Frames: {stats.frame_count}"
                      + (f" | Saved: {save_count}" if saving else ""))

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    finally:
        try:
            image_hub.close()
        except Exception:
            pass
        if not headless:
            cv2.destroyAllWindows()
        print(f"[INFO] Total frames received: {stats.frame_count}")
        if saving:
            print(f"[INFO] Frames saved: {save_count} → {save_dir}")
        print("[INFO] Receiver stopped.")


if __name__ == "__main__":
    main()

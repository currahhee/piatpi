#!/usr/bin/env python3
"""
imagezmq Receiver v4 - Clean LAN Receiver
=========================================
Direct Pi-to-PC streaming over imagezmq using PUB/SUB only.
Compatible with pi_camera_stream3_1.py without sender changes.

    Pi Camera -> imagezmq (PUB/SUB) -> [this receiver] -> display / save

Usage:
    python imagezmq_receiver4.py
    python imagezmq_receiver4.py --ip 192.168.1.14 --port 5555
    python imagezmq_receiver4.py --save --save-dir ~/my_frames
    python imagezmq_receiver4.py --headless --save

Keys:
    q     quit
    s     toggle frame saving
    f     toggle fullscreen
    r     reset stats
    space pause/resume display
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import imagezmq
import numpy as np


DEFAULT_PI_IP = "192.168.1.14"
DEFAULT_PI_PORT = 5555
DEFAULT_SAVE_DIR = Path.home() / "mef_received"
SAVE_QUALITY = 95
FPS_WINDOW = 60
WINDOW_NAME = "Pi4 Stream"
WINDOW_SIZE = (1280, 720)
OSD_FONT = cv2.FONT_HERSHEY_SIMPLEX
OSD_SCALE = 0.55
OSD_COLOR = (0, 255, 0)
OSD_SHADOW_COLOR = (0, 0, 0)
OSD_THICKNESS = 1
LOG_INTERVAL_S = 5.0
RECV_RETRY_DELAY_S = 0.5
SIGINT_CONFIRM_WINDOW_S = 2.0


class StreamStats:
    """Track rolling stream stats for the on-screen display and logs."""

    def __init__(self, window: int = FPS_WINDOW) -> None:
        self.window = window
        self.timestamps = deque(maxlen=window)
        self.decode_times_ms = deque(maxlen=window)
        self.frame_count = 0
        self.bytes_received = 0
        self.started_at = time.monotonic()

    def tick(self, payload_size: int, decode_ms: float) -> None:
        self.timestamps.append(time.monotonic())
        self.decode_times_ms.append(decode_ms)
        self.frame_count += 1
        self.bytes_received += payload_size

    @property
    def fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        span = self.timestamps[-1] - self.timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / span

    @property
    def avg_decode_ms(self) -> float:
        if not self.decode_times_ms:
            return 0.0
        return sum(self.decode_times_ms) / len(self.decode_times_ms)

    @property
    def bandwidth_kbps(self) -> float:
        elapsed = time.monotonic() - self.started_at
        if elapsed <= 0:
            return 0.0
        return (self.bytes_received * 8) / elapsed / 1000

    def reset(self) -> None:
        self.timestamps.clear()
        self.decode_times_ms.clear()
        self.frame_count = 0
        self.bytes_received = 0
        self.started_at = time.monotonic()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="imagezmq LAN receiver (PUB/SUB)")
    parser.add_argument("--ip", default=DEFAULT_PI_IP, help=f"Pi IP (default {DEFAULT_PI_IP})")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PI_PORT,
        help=f"Port (default {DEFAULT_PI_PORT})",
    )
    parser.add_argument("--headless", action="store_true", help="No display window")
    parser.add_argument("--save", action="store_true", help="Start with saving enabled")
    parser.add_argument(
        "--save-dir",
        default=str(DEFAULT_SAVE_DIR),
        help=f"Save dir (default {DEFAULT_SAVE_DIR})",
    )
    return parser.parse_args()


def coerce_jpg_payload(jpg_buffer: object) -> bytes:
    """Normalize imagezmq payloads across pyzmq versions."""
    if isinstance(jpg_buffer, bytes):
        return jpg_buffer
    if isinstance(jpg_buffer, (bytearray, memoryview)):
        return bytes(jpg_buffer)

    payload = getattr(jpg_buffer, "bytes", None)
    if payload is not None:
        return bytes(payload)

    return bytes(jpg_buffer)


def decode_jpg_frame(jpg_buffer: object) -> tuple[np.ndarray | None, int, float]:
    payload = coerce_jpg_payload(jpg_buffer)
    started = time.monotonic()
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    decode_ms = (time.monotonic() - started) * 1000
    return frame, len(payload), decode_ms


def process_frame(frame: np.ndarray, name: str) -> np.ndarray:
    """Placeholder for future MEF fusion / defect detection integration."""
    return frame


def ensure_save_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_frame(frame: np.ndarray, save_dir: Path, save_index: int) -> bool:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"frame_{timestamp}_{save_index:06d}.jpg"
    save_path = save_dir / filename
    return cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_QUALITY])


def draw_osd(
    frame: np.ndarray,
    stats: StreamStats,
    source_name: str,
    saving: bool,
    paused: bool,
) -> None:
    height, width = frame.shape[:2]
    lines = [
        f"Source: {source_name}",
        f"Resolution: {width}x{height}",
        f"FPS: {stats.fps:.1f}  |  Decode: {stats.avg_decode_ms:.1f}ms",
        f"Bandwidth: {stats.bandwidth_kbps:.0f} kbps  |  Frames: {stats.frame_count}",
    ]

    if saving:
        lines.append("[REC] Saving frames")
    if paused:
        lines.append("[PAUSED]")

    y = 25
    for line in lines:
        cv2.putText(
            frame,
            line,
            (11, y + 1),
            OSD_FONT,
            OSD_SCALE,
            OSD_SHADOW_COLOR,
            OSD_THICKNESS + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (10, y),
            OSD_FONT,
            OSD_SCALE,
            OSD_COLOR,
            OSD_THICKNESS,
            cv2.LINE_AA,
        )
        y += 22

def window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def log_stats(stats: StreamStats, save_count: int, saving: bool) -> None:
    message = (
        f"[STATS] FPS:{stats.fps:.1f} "
        f"Dec:{stats.avg_decode_ms:.1f}ms "
        f"BW:{stats.bandwidth_kbps:.0f}kbps "
        f"Frames:{stats.frame_count}"
    )
    if saving:
        message += f" Saved:{save_count}"
    print(message)


def main() -> None:
    args = parse_args()
    save_dir = Path(os.path.expanduser(args.save_dir))
    addr = f"tcp://{args.ip}:{args.port}"

    print(f"[*] Connecting to {addr} (PUB/SUB)...")
    hub = imagezmq.ImageHub(open_port=addr, REQ_REP=False)
    print("[*] Waiting for frames...")

    stats = StreamStats()
    saving = args.save
    paused = False
    fullscreen = False
    running = True
    stop_reason = ""
    saved_frames = 0
    last_log_time = time.monotonic()
    last_sigint_time = 0.0

    if saving:
        ensure_save_dir(save_dir)

    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *WINDOW_SIZE)
        print("[*] Keys: q=quit  s=save  f=fullscreen  r=reset  space=pause")

    def handle_signal(sig: int, _frame: object) -> None:
        nonlocal last_sigint_time, running, stop_reason

        if sig == signal.SIGINT and not args.headless:
            now = time.monotonic()
            if now - last_sigint_time > SIGINT_CONFIRM_WINDOW_S:
                last_sigint_time = now
                print(f"\n[!] Ctrl+C again within {SIGINT_CONFIRM_WINDOW_S:.0f}s to quit")
                return

        signal_name = signal.Signals(sig).name
        stop_reason = signal_name
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while running:
            try:
                name, jpg_buffer = hub.recv_jpg()
            except Exception as exc:
                if not running:
                    break
                print(f"[!] Receive error: {exc}")
                time.sleep(RECV_RETRY_DELAY_S)
                continue

            frame, payload_size, decode_ms = decode_jpg_frame(jpg_buffer)
            if frame is None:
                print("[!] Failed to decode frame, skipping")
                continue

            stats.tick(payload_size, decode_ms)
            frame = process_frame(frame, name)

            if saving:
                if save_frame(frame, save_dir, saved_frames + 1):
                    saved_frames += 1
                else:
                    print(f"[!] Failed to save frame to {save_dir}")

            if not args.headless:
                if window_closed(WINDOW_NAME):
                    stop_reason = "window_closed"
                    break

                if not paused:
                    display_frame = frame.copy()
                    draw_osd(display_frame, stats, name, saving=saving, paused=paused)
                    cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_reason = "quit_key"
                    break
                if key == ord("s"):
                    saving = not saving
                    if saving:
                        ensure_save_dir(save_dir)
                    state = f"ON -> {save_dir}" if saving else "OFF"
                    print(f"[*] Saving: {state}")
                if key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                    )
                if key == ord("r"):
                    stats.reset()
                    saved_frames = 0
                    print("[*] Stats reset")
                if key == ord(" "):
                    paused = not paused
                    print(f"[*] {'Paused' if paused else 'Resumed'}")

            now = time.monotonic()
            if now - last_log_time >= LOG_INTERVAL_S and stats.frame_count > 0:
                last_log_time = now
                log_stats(stats, saved_frames, saving)

    except KeyboardInterrupt:
        stop_reason = "KeyboardInterrupt"
        print("\n[*] Interrupted")
    finally:
        try:
            hub.close()
        except Exception:
            pass
        if not args.headless:
            cv2.destroyAllWindows()

        print(f"[*] Done. Received {stats.frame_count} frames", end="")
        if saved_frames:
            print(f", saved {saved_frames} to {save_dir}", end="")
        print()
        if stop_reason:
            print(f"[*] Stop reason: {stop_reason}")


if __name__ == "__main__":
    main()

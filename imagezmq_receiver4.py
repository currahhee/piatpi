"""
imagezmq Receiver v4 — Clean LAN Receiver
==========================================
Direct Pi4 <-> PC connection over LAN via imagezmq (PUB/SUB).
No TCP server, no legacy modes. Just receive and display.

    Pi Camera -> imagezmq (PUB/SUB) -> [this receiver] -> display / save

Install:
    pip install imagezmq opencv-python pyzmq numpy

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

import os
import time
import argparse
import signal
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import imagezmq

# ── Config ────────────────────────────────────────────────────────────────
PI_IP           = "192.168.1.14"
PI_PORT         = 5555
SAVE_DIR        = os.path.expanduser("~/mef_received")
SAVE_QUALITY    = 95
FPS_WINDOW      = 60
WINDOW_NAME     = "Pi4 Stream"
OSD_FONT        = cv2.FONT_HERSHEY_SIMPLEX
OSD_SCALE       = 0.55
OSD_COLOR       = (0, 255, 0)


# ── Stats ─────────────────────────────────────────────────────────────────
class Stats:
    def __init__(self):
        self.ts = deque(maxlen=FPS_WINDOW)
        self.decode_ms = deque(maxlen=FPS_WINDOW)
        self.frames = 0
        self.bytes = 0
        self.t0 = time.monotonic()

    def tick(self, size, dec_ms):
        self.ts.append(time.monotonic())
        self.decode_ms.append(dec_ms)
        self.frames += 1
        self.bytes += size

    @property
    def fps(self):
        if len(self.ts) < 2:
            return 0.0
        span = self.ts[-1] - self.ts[0]
        return (len(self.ts) - 1) / span if span > 0 else 0.0

    @property
    def avg_decode(self):
        return sum(self.decode_ms) / len(self.decode_ms) if self.decode_ms else 0.0

    @property
    def bw_kbps(self):
        dt = time.monotonic() - self.t0
        return (self.bytes * 8) / dt / 1000 if dt > 0 else 0.0

    def reset(self):
        self.ts.clear()
        self.decode_ms.clear()
        self.frames = 0
        self.bytes = 0
        self.t0 = time.monotonic()


# ── OSD ───────────────────────────────────────────────────────────────────
def draw_osd(frame, stats, src, saving):
    h, w = frame.shape[:2]
    lines = [
        f"{src}  |  {w}x{h}",
        f"FPS: {stats.fps:.1f}  |  Decode: {stats.avg_decode:.1f}ms  |  BW: {stats.bw_kbps:.0f} kbps",
        f"Frames: {stats.frames}",
    ]
    if saving:
        lines.append("[REC]")

    y = 25
    for line in lines:
        cv2.putText(frame, line, (11, y+1), OSD_FONT, OSD_SCALE, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y),   OSD_FONT, OSD_SCALE, OSD_COLOR, 1, cv2.LINE_AA)
        y += 22


# ── Pipeline hook ─────────────────────────────────────────────────────────
def process_frame(frame, name):
    """Plug MEF fusion / defect detection here. Returns frame unchanged for now."""
    return frame


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="imagezmq LAN receiver (PUB/SUB)")
    ap.add_argument("--ip",       default=PI_IP,   help=f"Pi IP (default {PI_IP})")
    ap.add_argument("--port",     type=int, default=PI_PORT, help=f"Port (default {PI_PORT})")
    ap.add_argument("--headless", action="store_true", help="No display window")
    ap.add_argument("--save",     action="store_true", help="Start with saving enabled")
    ap.add_argument("--save-dir", default=SAVE_DIR, help=f"Save dir (default {SAVE_DIR})")
    args = ap.parse_args()

    addr = f"tcp://{args.ip}:{args.port}"
    print(f"[*] Connecting to {addr} (PUB/SUB)...")

    hub = imagezmq.ImageHub(open_port=addr, REQ_REP=False)
    print("[*] Waiting for frames...")

    stats    = Stats()
    saving   = args.save
    save_ct  = 0
    paused   = False
    fs       = False
    running  = True
    log_t    = time.monotonic()

    # -- signal handling (double Ctrl+C to quit when GUI is open) --
    sigint_t = 0.0
    def on_signal(sig, _):
        nonlocal running, sigint_t
        if sig == signal.SIGINT and not args.headless:
            now = time.monotonic()
            if now - sigint_t > 2.0:
                sigint_t = now
                print("\n[!] Ctrl+C again within 2s to quit")
                return
        running = False
    signal.signal(signal.SIGINT,  on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if saving:
        os.makedirs(args.save_dir, exist_ok=True)

    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    try:
        while running:
            # ── receive ──
            try:
                name, jpg_buf = hub.recv_jpg()
            except Exception as e:
                if not running:
                    break
                print(f"[!] recv error: {e}")
                time.sleep(0.5)
                continue

            if isinstance(jpg_buf, (bytes, bytearray, memoryview)):
                payload = jpg_buf
            else:
                payload = getattr(jpg_buf, "bytes", None) or bytes(jpg_buf)
            jpg_size = len(payload)

            # ── decode ──
            t0 = time.monotonic()
            frame = cv2.imdecode(np.frombuffer(payload, dtype="uint8"), cv2.IMREAD_COLOR)
            dec_ms = (time.monotonic() - t0) * 1000

            if frame is None:
                continue

            stats.tick(jpg_size, dec_ms)
            frame = process_frame(frame, name)

            # ── save ──
            if saving:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                path = os.path.join(args.save_dir, f"frame_{ts}_{save_ct+1:06d}.jpg")
                if cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_QUALITY]):
                    save_ct += 1

            # ── display ──
            if not args.headless:
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break

                if not paused:
                    disp = frame.copy()
                    draw_osd(disp, stats, name, saving)
                    cv2.imshow(WINDOW_NAME, disp)

                key = cv2.waitKey(1) & 0xFF
                if   key == ord("q"): break
                elif key == ord("s"):
                    saving = not saving
                    if saving:
                        os.makedirs(args.save_dir, exist_ok=True)
                    print(f"[*] Saving: {'ON -> ' + args.save_dir if saving else 'OFF'}")
                elif key == ord("f"):
                    fs = not fs
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fs else cv2.WINDOW_NORMAL)
                elif key == ord("r"):
                    stats.reset(); save_ct = 0; print("[*] Stats reset")
                elif key == ord(" "):
                    paused = not paused
                    print(f"[*] {'Paused' if paused else 'Resumed'}")

            # ── periodic log ──
            now = time.monotonic()
            if now - log_t >= 5.0 and stats.frames > 0:
                log_t = now
                msg = f"[STATS] FPS:{stats.fps:.1f} Dec:{stats.avg_decode:.1f}ms BW:{stats.bw_kbps:.0f}kbps Frames:{stats.frames}"
                if saving:
                    msg += f" Saved:{save_ct}"
                print(msg)

    except KeyboardInterrupt:
        pass

    finally:
        try: hub.close()
        except Exception: pass
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"[*] Done. Received {stats.frames} frames" +
              (f", saved {save_ct}" if save_ct else ""))


if __name__ == "__main__":
    main()

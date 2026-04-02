# piatpi

Raspberry Pi camera streaming + MEF (Multi-Exposure Fusion) pipeline.

Streams frames from a Pi Camera Module to a gaming laptop over the local network using imagezmq, with multi-exposure bracketing for defect detection.

Current pipeline:

`Pi Camera -> imagezmq -> gaming laptop (RTX 5060) -> MEF model / downstream defect detection`

There is no separate TCP server in the active pipeline. The only network path is `imagezmq` (ZeroMQ over `tcp://...` under the hood).

## Files

Active entry points:

- `pi_camera_stream3_1.py` is the current Pi sender.
- `pi_camera_stream3_1_live.py` is the live-stream-only Pi launcher.
- `imagezmq_receiver4.py` is the current laptop receiver / MEF host.
- Older versioned scripts are kept as history, including early TCP experiments.

### Pi side (runs on Raspberry Pi)
| File | Description |
|------|-------------|
| `pi_camera_stream.py` | v1 - basic TCP + imagezmq streaming |
| `pi_camera_stream2.py` | v2 - improved streaming with numpy/imagezmq |
| `pi_camera_stream2up.py` | v2 updated - minor fixes over v2 |
| `pi_camera_stream3.py` | v3 - adds multi-exposure bracketing in a separate thread |
| `pi_camera_stream3_1.py` | v3.1 - latest, with camera lock, cleaner bracket capture, metadata JSON sidecars |

### Laptop / PC side (runs on your gaming laptop)
| File | Description |
|------|-------------|
| `imagezmq_receiver.py` | v1 - basic receiver/viewer |
| `imagezmq_receiver3.py` | v3 - FPS overlay, frame saving, keyboard controls, CLI args, MEF hook ready |
| `imagezmq_receiver4.py` | v4 - clean LAN-only PUB/SUB receiver, active laptop entry point |

### Other
| File | Description |
|------|-------------|
| `v2tov3.txt` | Notes on what changed between v2 and v3 |

## Setup (Laptop / PC side)

We use **uv** for dependency management. It replaces pip and venv in a single tool.

### 1. Install uv
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Restart your terminal after installing.

### 2. Clone and install dependencies
```bash
git clone https://github.com/currahhee/piatpi.git
cd piatpi
uv sync
```
`uv sync` creates the `.venv` and installs all dependencies from the lockfile automatically.

### 3. Run
```bash
# Activate the venv
.venv\Scripts\activate

# Or just use uv run
uv run python imagezmq_receiver4.py --ip <PI_IP>
```

### Pi side setup
```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-zmq
pip3 install imagezmq
python3 pi_camera_stream3_1.py
```

For live streaming without local bracket captures on the Pi:

```bash
python3 pi_camera_stream3_1_live.py
```

## Keyboard controls (receiver v4)
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Toggle saving frames to disk |
| `f` | Toggle fullscreen |
| `r` | Reset FPS counter |
| `space` | Pause/resume display |

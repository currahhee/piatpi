# piatpi

Raspberry Pi camera streaming + MEF (Multi-Exposure Fusion) pipeline.

Streams frames from a Pi Camera Module to a gaming laptop over the local network using imagezmq, with multi-exposure bracketing for defect detection.

## Pipeline overview

There are currently **two related pipelines** in this repo:

### 1. Live stream pipeline

`Pi Camera -> imagezmq -> gaming laptop receiver -> display / saving / future online inference`

This is the real-time path used for monitoring and receiver-side viewing.

### 2. MEF dataset pipeline

`Pi Camera -> bracket capture on Pi -> saved bracket dataset -> copied to laptop -> MEF fusion backend -> fused output`

This is the path used to take the bracketed images from the Pi 4B and feed them into your MEF model.

Current high-level goal:

`Pi Camera -> imagezmq -> gaming laptop (RTX 5060) -> MEF model / downstream defect detection`

There is no separate TCP server in the active pipeline. The only network path is `imagezmq` (ZeroMQ over `tcp://...` under the hood).

For bracketed dataset capture, the Pi also saves local image sets plus a metadata JSON sidecar. Those saved sets are what the offline MEF pipeline consumes on the laptop.

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
| `mef_pipeline.py` | Dataset loader + fusion backends for offline MEF processing |
| `run_mef_dataset.py` | Laptop CLI to load Pi bracket sets and run fusion |
| `mefnet_adapter_template.py` | Template adapter for plugging in a real MEF-Net model |

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

## Dataset to MEF-Net pipeline

The Pi already captures bracketed datasets locally in `~/mef_captures`:

- `set0001_20260416_101010_under_exp5000us.jpg`
- `set0001_20260416_101010_normal_exp33000us.jpg`
- `set0001_20260416_101010_over_exp200000us.jpg`
- `set0001_20260416_101010_meta.json`

The laptop-side dataset runner reads those bracket sets and feeds them to a fusion backend.

### End-to-end flow

1. Run `pi_camera_stream3_1.py` on the Pi.
2. Let the Pi save bracket sets into `~/mef_captures`.
3. Copy `~/mef_captures` from the Pi to the laptop.
4. Run `run_mef_dataset.py` on the laptop.
5. `mef_pipeline.py` discovers each set and loads the three exposure images.
6. The selected backend fuses the set:
   OpenCV fallback now, MEF-Net adapter when your model is plugged in.
7. The laptop writes the fused result into the output directory.

### 1. Copy bracket captures from the Pi to the laptop
```bash
scp -r pi@<PI_IP>:~/mef_captures ./pi_dataset
```

### 2. Inspect discovered bracket sets
```bash
uv run python run_mef_dataset.py --dataset-dir ./pi_dataset --list
```

### 3. Run the offline pipeline with the built-in OpenCV fallback
```bash
uv run python run_mef_dataset.py --dataset-dir ./pi_dataset --output-dir ./outputs/mef_fused
```

This is useful for validating that dataset discovery, loading, and output writing all work before you wire in the real model.

### 4. Plug in your MEF-Net model
Copy `mefnet_adapter_template.py`, replace the placeholder logic with your real MEF-Net import/checkpoint/inference code, then run:

```bash
uv run python run_mef_dataset.py \
  --dataset-dir ./pi_dataset \
  --output-dir ./outputs/mefnet \
  --backend module \
  --adapter ./my_mefnet_adapter.py \
  --model-path ./weights/mefnet.pth \
  --device cuda:0
```

The adapter receives:

- `images`: list of BGR `numpy` arrays from a single bracket set
- `bracket_set`: metadata object with `set_id`, `timestamp`, labels, and exposures

Use `prepare_model_inputs()` from `mef_pipeline.py` to convert those images into a normalized stack for your model.

## What is implemented now

- Pi-side bracket capture and metadata export
- Live Pi-to-laptop streaming with `imagezmq`
- Offline bracket-set discovery and loading on the laptop
- Offline fusion runner with a pluggable backend interface
- A MEF-Net adapter template for your real model integration

## What is not implemented yet

- Real-time MEF-Net inference directly from the live `imagezmq` stream
- Downstream defect detection after fusion
- Packaging MEF-Net dependencies and weights in this repo

## Keyboard controls (receiver v4)
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Toggle saving frames to disk |
| `f` | Toggle fullscreen |
| `r` | Reset FPS counter |
| `space` | Pause/resume display |

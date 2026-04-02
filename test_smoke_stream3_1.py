#!/usr/bin/env python3
"""
Smoke tests for pi_camera_stream3_1.py — levels 1-3 (no Pi hardware needed).

Run:  python test_smoke_stream3_1.py
"""

import ast
import importlib
import sys
import types
import textwrap
import numpy as np
import cv2

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)


# ── Stub out Pi-only modules before importing ────────────────────────────
# picamera2 and libcamera don't exist on Windows, so we create minimal
# stubs so the module can be imported for testing pure-logic functions.

picamera2_stub = types.ModuleType("picamera2")


class _FakePicamera2:
    camera_controls = {}
    def create_preview_configuration(self, **kw): return {}
    def configure(self, cfg): pass
    def start(self): pass
    def stop(self): pass
    def set_controls(self, ctrl): pass
    def capture_array(self): return np.zeros((1080, 1920, 3), dtype=np.uint8)
    def capture_metadata(self): return {"ExposureTime": 33000, "AnalogueGain": 1.0}
    def capture_request(self):
        class _Req:
            def release(self): pass
            def make_array(self, _): return np.zeros((1080, 1920, 3), dtype=np.uint8)
            def get_metadata(self): return {"ExposureTime": 33000, "AnalogueGain": 1.0}
        return _Req()


picamera2_stub.Picamera2 = _FakePicamera2
sys.modules["picamera2"] = picamera2_stub

libcamera_stub = types.ModuleType("libcamera")
controls_stub = types.ModuleType("libcamera.controls")


class _DraftNR:
    Off = 0
    Fast = 1
    HighQuality = 2


class _Draft:
    NoiseReductionModeEnum = _DraftNR


controls_stub.draft = _Draft()
libcamera_stub.controls = controls_stub
sys.modules["libcamera"] = libcamera_stub
sys.modules["libcamera.controls"] = controls_stub

# Now import the actual module
import pi_camera_stream3_1 as mod


# ═════════════════════════════════════════════════════════════════════════
# Level 1 — Import & Parse
# ═════════════════════════════════════════════════════════════════════════
print("\n=== Level 1: Import & Parse ===")

# 1a. Syntax check via AST
with open("pi_camera_stream3_1.py", encoding="utf-8") as f:
    source = f.read()
try:
    ast.parse(source)
    check("syntax_valid", True)
except SyntaxError as e:
    check("syntax_valid", False, str(e))

# 1b. Module imported successfully
check("module_imports", "pi_camera_stream3_1" in sys.modules)

# 1c. libcamera graceful fallback (our stub provides it, but test the pattern)
check("libcamera_import_guarded",
      hasattr(mod, "libcamera_controls"),
      "libcamera_controls attribute missing")

# 1d. Key functions exist
for fn in ["create_camera", "capture_loop", "imagezmq_stream_loop",
           "bracket_loop", "capture_bracket_set", "compute_shutter_speeds",
           "reduce_stream_noise", "main"]:
    check(f"func_exists_{fn}", callable(getattr(mod, fn, None)))


# ═════════════════════════════════════════════════════════════════════════
# Level 2 — Configuration Validation
# ═════════════════════════════════════════════════════════════════════════
print("\n=== Level 2: Configuration Validation ===")

# 2a. RESOLUTION
check("resolution_tuple",
      isinstance(mod.RESOLUTION, tuple) and len(mod.RESOLUTION) == 2)
check("resolution_positive",
      all(isinstance(v, int) and v > 0 for v in mod.RESOLUTION))

# 2b. TARGET_FPS
check("target_fps_positive",
      isinstance(mod.TARGET_FPS, (int, float)) and mod.TARGET_FPS > 0)

# 2c. JPEG_QUALITY
check("jpeg_quality_range",
      1 <= mod.JPEG_QUALITY <= 100,
      f"got {mod.JPEG_QUALITY}")

# 2d. Bracket shutter speeds within hardware limits
check("shutter_speeds_in_range",
      all(mod.SHUTTER_MIN_US <= s <= mod.SHUTTER_MAX_US
          for s in mod.BRACKET_SHUTTER_SPEEDS),
      f"limits: [{mod.SHUTTER_MIN_US}, {mod.SHUTTER_MAX_US}]")

# 2e. Label/speed length match
check("labels_match_speeds",
      len(mod.BRACKET_LABELS) == len(mod.BRACKET_SHUTTER_SPEEDS),
      f"{len(mod.BRACKET_LABELS)} labels vs {len(mod.BRACKET_SHUTTER_SPEEDS)} speeds")

# 2f. Label/multiplier length match
check("labels_match_ev_multipliers",
      len(mod.BRACKET_LABELS) == len(mod.BRACKET_EV_MULTIPLIERS),
      f"{len(mod.BRACKET_LABELS)} labels vs {len(mod.BRACKET_EV_MULTIPLIERS)} multipliers")

# 2g. BRACKET_MODE valid
check("bracket_mode_valid",
      mod.BRACKET_MODE in ("fixed", "relative"),
      f"got '{mod.BRACKET_MODE}'")

# 2h. STREAM_DENOISE_FILTER valid
check("denoise_filter_valid",
      mod.STREAM_DENOISE_FILTER.strip().lower() in ("off", "median", "gaussian", "bilateral"),
      f"got '{mod.STREAM_DENOISE_FILTER}'")

# 2i. BRACKET_JPEG_QUALITY
check("bracket_jpeg_quality_range",
      1 <= mod.BRACKET_JPEG_QUALITY <= 100,
      f"got {mod.BRACKET_JPEG_QUALITY}")

# 2j. BRACKET_SETTLE_FRAMES positive
check("settle_frames_positive",
      isinstance(mod.BRACKET_SETTLE_FRAMES, int) and mod.BRACKET_SETTLE_FRAMES >= 1)

# 2k. BRACKET_SET_INTERVAL positive
check("bracket_interval_positive",
      mod.BRACKET_SET_INTERVAL > 0)

# 2l. RECONNECT_DELAY positive
check("reconnect_delay_positive",
      mod.RECONNECT_DELAY > 0)

# 2m. EV multipliers positive
check("ev_multipliers_positive",
      all(m > 0 for m in mod.BRACKET_EV_MULTIPLIERS))

# 2n. Shutter min < max
check("shutter_min_lt_max",
      mod.SHUTTER_MIN_US < mod.SHUTTER_MAX_US)


# ═════════════════════════════════════════════════════════════════════════
# Level 3 — Helper Functions (pure logic)
# ═════════════════════════════════════════════════════════════════════════
print("\n=== Level 3: Helper Functions ===")

# 3a. odd_kernel_size
check("odd_kernel_basic",
      mod.odd_kernel_size(3, 5) == 3)
check("odd_kernel_even_rounds_up",
      mod.odd_kernel_size(4, 5) == 5)
check("odd_kernel_zero_uses_fallback",
      mod.odd_kernel_size(0, 7) == 7)
check("odd_kernel_none_fallback",
      mod.odd_kernel_size(None, 3) == 3)
check("odd_kernel_always_odd",
      mod.odd_kernel_size(6, 3) % 2 == 1)
check("odd_kernel_negative_clamps",
      mod.odd_kernel_size(-5, 3) >= 1)

# 3b. get_frame_duration_limits
fdl = mod.get_frame_duration_limits()
check("frame_duration_tuple",
      isinstance(fdl, tuple) and len(fdl) == 2)
expected_us = round(1_000_000 / mod.TARGET_FPS)
check("frame_duration_value",
      fdl == (expected_us, expected_us),
      f"expected ({expected_us}, {expected_us}), got {fdl}")

# 3c. get_camera_noise_reduction_control
nr = mod.get_camera_noise_reduction_control()
check("noise_reduction_returns_value",
      nr is not None,
      f"got {nr}")

# 3d. reduce_stream_noise — test each filter mode on a dummy frame
dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Save and restore the module-level config for each test
original_filter = mod.STREAM_DENOISE_FILTER

for filter_mode in ["off", "median", "gaussian", "bilateral"]:
    mod.STREAM_DENOISE_FILTER = filter_mode
    try:
        result = mod.reduce_stream_noise(dummy.copy())
        check(f"denoise_{filter_mode}_shape",
              result.shape == dummy.shape,
              f"expected {dummy.shape}, got {result.shape}")
        check(f"denoise_{filter_mode}_dtype",
              result.dtype == np.uint8)
    except Exception as e:
        check(f"denoise_{filter_mode}_runs", False, str(e))

mod.STREAM_DENOISE_FILTER = original_filter

# 3e. reduce_stream_noise — unknown mode doesn't crash
mod.STREAM_DENOISE_FILTER = "nonexistent_filter"
mod.stream_denoise_warning_shown = False
try:
    result = mod.reduce_stream_noise(dummy.copy())
    check("denoise_unknown_passthrough",
          np.array_equal(result, dummy),
          "should return frame unchanged")
except Exception as e:
    check("denoise_unknown_no_crash", False, str(e))
mod.STREAM_DENOISE_FILTER = original_filter

# 3f. compute_shutter_speeds — fixed mode
fake_cam = _FakePicamera2()

original_mode = mod.BRACKET_MODE
mod.BRACKET_MODE = "fixed"
pairs, auto_exp, auto_gain = mod.compute_shutter_speeds(fake_cam)
check("compute_fixed_count",
      len(pairs) == len(mod.BRACKET_SHUTTER_SPEEDS),
      f"expected {len(mod.BRACKET_SHUTTER_SPEEDS)}, got {len(pairs)}")
check("compute_fixed_labels",
      [p[0] for p in pairs] == mod.BRACKET_LABELS[:len(pairs)])
check("compute_fixed_clamped",
      all(mod.SHUTTER_MIN_US <= p[1] <= mod.SHUTTER_MAX_US for p in pairs))

# 3g. compute_shutter_speeds — relative mode
mod.BRACKET_MODE = "relative"
pairs_r, auto_exp_r, auto_gain_r = mod.compute_shutter_speeds(fake_cam)
check("compute_relative_count",
      len(pairs_r) == len(mod.BRACKET_EV_MULTIPLIERS),
      f"expected {len(mod.BRACKET_EV_MULTIPLIERS)}, got {len(pairs_r)}")
check("compute_relative_clamped",
      all(mod.SHUTTER_MIN_US <= p[1] <= mod.SHUTTER_MAX_US for p in pairs_r))
# The 1.0x multiplier should give back the auto_exposure value
unity_idx = mod.BRACKET_EV_MULTIPLIERS.index(1.0) if 1.0 in mod.BRACKET_EV_MULTIPLIERS else -1
if unity_idx >= 0:
    check("compute_relative_unity",
          pairs_r[unity_idx][1] == auto_exp_r,
          f"1.0x should equal auto ({auto_exp_r}), got {pairs_r[unity_idx][1]}")

mod.BRACKET_MODE = original_mode

# 3h. imencode sanity (cv2 available)
ret, buf = cv2.imencode(".jpg", dummy, [cv2.IMWRITE_JPEG_QUALITY, mod.JPEG_QUALITY])
check("cv2_imencode_works", ret and len(buf) > 0)


# ═════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 50}")
print(f"  Results: {passed}/{total} passed, {failed} failed")
print(f"{'=' * 50}")
sys.exit(1 if failed else 0)

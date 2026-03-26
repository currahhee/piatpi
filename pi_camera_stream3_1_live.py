#!/usr/bin/env python3
"""
Live-stream-only launcher for pi_camera_stream3_1.

Uses imagezmq streaming and disables bracket capture so the Pi does not
save any local frames or metadata sidecars.
"""

import pi_camera_stream3_1 as base


# Stream-only mode for the Pi.
base.ENABLE_TCP = False
base.ENABLE_IMAGEZMQ = True
base.ENABLE_BRACKET = False
base.SAVE_METADATA_JSON = False


if __name__ == "__main__":
    base.main()

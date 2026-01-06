#!/usr/bin/env bash
# Example headless runner for CI or servers using Xvfb
# Usage: ./scripts/headless_example.sh path/to/video.mp4

set -euo pipefail
VIDEO=${1:-"path/to/video.mp4"}
export PYVISTA_OFF_SCREEN=1
xvfb-run -a -s "-screen 0 1280x720x24" python hybrid_ultrasound_3d_safe.py --video "$VIDEO" --resize 128

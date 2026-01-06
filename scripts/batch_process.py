#!/usr/bin/env python3
"""Batch process a directory of videos using the project's scripts.

Usage examples:
  python scripts/batch_process.py --input-dir dataset/videos --script full --out-dir outputs
  python scripts/batch_process.py --input-dir dataset/videos --script safe --workers 4
"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_MAP = {
    "full": "hybrid_ultrasound_3d_full.py",
    "safe": "hybrid_ultrasound_3d_safe.py",
}


def process_video(script: str, video_path: Path, out_dir: Path, env: dict | None = None):
    cmd = ["python", script, "--video", str(video_path), "--out-dir", str(out_dir)]
    env_in = os.environ.copy()
    if env:
        env_in.update(env)
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, env=env_in)
    return res.returncode


def main():
    parser = argparse.ArgumentParser(description="Batch process videos")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--script", choices=SCRIPT_MAP.keys(), default="full")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--headless", action="store_true", help="Run with PYVISTA_OFF_SCREEN=1")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script = SCRIPT_MAP[args.script]
    videos = list(input_dir.glob(args.pattern))
    if not videos:
        print("No videos found for pattern", args.pattern)
        return

    env = {"PYVISTA_OFF_SCREEN": "1"} if args.headless else None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(process_video, script, v, out_dir, env) for v in videos
        ]
        for f in futures:
            rc = f.result()
            if rc != 0:
                print("Warning: process returned non-zero exit code:", rc)

    print("Batch processing completed")


if __name__ == "__main__":
    main()

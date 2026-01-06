#!/usr/bin/env python3
"""
ai_ultrasound_realtime.py

Real-time 3D ultrasound prototype from video input.
Compatible with PyVista 0.43+ (uses ImageData).
Supports smoothing, optional Real-ESRGAN super-resolution,
interactive volumetric rendering, and multithreading.

Example usage:
python ai_ultrasound_realtime.py \
    --video ../usliverseq-mp4/volunteer02.mp4 \
    --resize 256 \
    --max-slices 200 \
    --smooth-sigma 1.2 \
    --frame-delay 0.05 \
    --threads 4 \
    --use-sr \
    --sr-model-path RealESRGAN_x4plus.pth
"""

import argparse
import time
import threading
from collections import deque

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from typing import Optional, Tuple, Generator

# Reuse CLAHE instance to avoid re-creating it per-frame
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Optional libs will be imported lazily when needed
HAS_ESRGAN = False
BackgroundPlotter = None


# -------------------- Utilities --------------------
def preprocess_frame(frame: np.ndarray, resize: Optional[int]) -> np.ndarray:
    """Preprocess a single frame using the global CLAHE instance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if resize is not None:
        gray = cv2.resize(gray, (resize, resize), interpolation=cv2.INTER_AREA)
    gray = _CLAHE.apply(gray)
    return gray.astype(np.float32) / 255.0


def sr_frame(sr_model, frame: np.ndarray, device: str) -> np.ndarray:
    """Apply SR model; import torch lazily only when needed."""
    if sr_model is None:
        return frame
    import torch

    tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = sr_model(tensor)
    return np.clip(out.squeeze().cpu().numpy(), 0.0, 1.0)    


def threaded_frame_reader(
    video_path: str,
    max_frames: Optional[int],
    resize: Optional[int],
    sr_model,
    device: str,
    threads: int,
) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
    """Read frames from the video and yield preprocessed frames with synthetic landmarks.

    This is a streaming reader that avoids pre-submitting a large number of tasks
    (which can exhaust memory). If max_frames is None, read until EOF.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    to_read = int(max_frames) if max_frames is not None else total if total > 0 else None

    idx = 0
    while True:
        if to_read is not None and idx >= to_read:
            break
        ret, frame = cap.read()
        if not ret:
            break
        proc = preprocess_frame(frame, resize)
        proc = sr_frame(sr_model, proc, device) if sr_model else proc
        # simulated landmarks for demo
        rng = np.random.RandomState(idx + 42)
        lm_x = rng.randint(0, proc.shape[1], size=5)
        lm_y = rng.randint(0, proc.shape[0], size=5)
        landmarks = np.stack([lm_x, lm_y], axis=1)
        yield idx, proc, landmarks
        idx += 1
    cap.release()


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--max-slices", type=int, default=200)
    parser.add_argument("--smooth-sigma", type=float, default=1.2)
    parser.add_argument("--use-sr", action="store_true")
    parser.add_argument("--sr-model-path", type=str, default="RealESRGAN_x4plus.pth")
    parser.add_argument("--frame-delay", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--headless", action="store_true", help="Run without GUI (no pyvista/qt imports)")
    args = parser.parse_args()

    headless = args.headless

    # Lazy load torch only if SR is requested
    sr_model = None
    device = "cpu"
    if args.use_sr:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            print("Warning: torch not available; SR will be disabled.")
            device = "cpu"

        # Lazy import Real-ESRGAN
        try:
            from realesrgan import RealESRGAN

            HAS_ESRGAN = True
        except Exception:
            HAS_ESRGAN = False

        if args.use_sr and HAS_ESRGAN:
            try:
                sr_model = RealESRGAN(device, scale=4)
                sr_model.load_weights(args.sr_model_path)
                print("Real-ESRGAN loaded on", device)
            except Exception as e:
                print("Warning: Real-ESRGAN failed:", e)
                sr_model = None
        elif args.use_sr:
            print("Real-ESRGAN not installed, skipping SR.")

    resize = args.resize
    max_slices = args.max_slices
    smooth_sigma = args.smooth_sigma

    # rolling 3D volume
    volume = np.zeros((resize, resize, max_slices), dtype=np.float32)
    coverage = np.zeros_like(volume)
    # cap landmark history to avoid unbounded memory growth
    landmarks_3d = deque(maxlen=5000)

    # Conditionally import and initialize PyVista/Qt GUI only if not headless
    render_enabled = not headless
    if render_enabled:
        try:
            import pyvista as pv
            from pyvistaqt import BackgroundPlotter
        except Exception:
            print("pyvista/pyvistaqt UI not available; falling back to headless mode.")
            render_enabled = False

    if render_enabled:
        # PyVista ImageData
        grid = pv.ImageData(dimensions=(resize, resize, max_slices))
        grid.origin = (0, 0, 0)
        grid.spacing = (1, 1, 1)
        grid["values"] = volume.flatten(order="F")

        cover_grid = pv.ImageData(dimensions=(resize, resize, max_slices))
        cover_grid["coverage"] = coverage.flatten(order="F")

        # GUI
        pl = BackgroundPlotter(title="AI Ultrasound Live")
        pl.add_volume(grid, cmap="gray", opacity="linear", shade=True)
        pl.add_volume(cover_grid, cmap="coolwarm", opacity="linear")
        pl.add_axes()
        pl.show_grid()
    else:
        print("Running in headless mode; no GUI will be created.")
        pl = None
        grid = None
        cover_grid = None

    landmark_actor = None
    guidance_actor = None
    stop_flag = threading.Event()

    def stream_thread():
        nonlocal volume, coverage, landmarks_3d, landmark_actor, guidance_actor
        slice_idx = 0
        for idx, proc_frame, landmarks2d in threaded_frame_reader(
            args.video,
            max_frames=None,
            resize=resize,
            sr_model=sr_model,
            device=device,
            threads=args.threads,
        ):
            if stop_flag.is_set():
                break
            zpos = slice_idx % max_slices
            volume[:, :, zpos] = proc_frame
            coverage[:, :, zpos] += 1.0

            for xpix, ypix in landmarks2d:
                xi, yi, zi = (
                    int(np.clip(xpix, 0, resize - 1)),
                    int(np.clip(ypix, 0, resize - 1)),
                    zpos,
                )
                landmarks_3d.append((xi, yi, zi))
                coverage[xi, yi, zi] += 1.0

            if render_enabled and grid is not None:
                # smooth along Z only (cheaper and uses less temporaries)
                vol_sm = gaussian_filter1d(volume, sigma=smooth_sigma, axis=2)
                vmin, vmax = vol_sm.min(), vol_sm.max()
                norm = (vol_sm - vmin) / (vmax - vmin + 1e-8)
                grid["values"] = norm.flatten(order="F")
                cover_grid["coverage"] = coverage.flatten(order="F")

                # landmarks
                if len(landmarks_3d) > 0:
                    pts = np.array(list(landmarks_3d), dtype=np.float32)
                    if landmark_actor:
                        try:
                            pl.remove_actor(landmark_actor)
                        except Exception:
                            pass
                    landmark_actor = pl.add_points(
                        pts, color="red", point_size=8, render_points_as_spheres=True
                    )

                # guidance arrow
                min_idx = np.unravel_index(np.argmin(coverage), coverage.shape)
                center = np.array([resize // 2, resize // 2, zpos], dtype=float)
                tgt = np.array(min_idx, dtype=float)
                vec = tgt - center
                if guidance_actor:
                    try:
                        pl.remove_actor(guidance_actor)
                    except Exception:
                        pass
                arrow = pv.Arrow(start=center.tolist(), direction=vec.tolist(), scale=20.0)
                guidance_actor = pl.add_mesh(arrow, color="lime")

                pl.render()
            else:
                # headless: occasional lightweight logging to monitor progress/memory
                if slice_idx % 50 == 0:
                    try:
                        import psutil

                        rss = psutil.Process().memory_info().rss
                        print(f"[headless] frames={slice_idx} RSS={rss}")
                    except Exception:
                        pass

            slice_idx += 1
            time.sleep(args.frame_delay)

    t = threading.Thread(target=stream_thread, daemon=True)
    t.start()

    try:
        if render_enabled and pl is not None:
            pl.app.exec_()
        else:
            while t.is_alive():
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        t.join(timeout=2)
        if render_enabled and pl is not None:
            pl.close()


if __name__ == "__main__":
    main()

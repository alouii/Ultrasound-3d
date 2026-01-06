import argparse
import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

try:
    from realesrgan import RealESRGAN

    has_esrgan = True
except ImportError:
    has_esrgan = False
    print("⚠️ RealESRGAN not installed. Super-resolution will be skipped.")

import pyvista as pv
from pyvistaqt import BackgroundPlotter
import time


# ---------------------------
# Frame Preprocessing
# ---------------------------
def preprocess_frame(frame, resize=256):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (resize, resize))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray.astype(np.float32) / 255.0


def super_resolve_frame(model, frame, device="cuda"):
    frame_tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        sr = model(frame_tensor)
    sr = sr.squeeze().cpu().numpy()
    return np.clip(sr, 0, 1)


# ---------------------------
# Main Streaming Simulation
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--max-slices", type=int, default=200)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--use-sr", action="store_true")
    parser.add_argument("--sr-model-path", type=str, default="RealESRGAN_x4plus.pth")
    parser.add_argument(
        "--frame-delay", type=float, default=0.05
    )  # seconds between frames
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr_model = None
    if args.use_sr and has_esrgan:
        sr_model = RealESRGAN(device, scale=4)
        sr_model.load_weights(args.sr_model_path)
        print(f"✅ ESRGAN loaded on {device}")
    elif args.use_sr:
        print("⚠️ RealESRGAN not available. Skipping super-resolution.")

    # Initialize video
    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read video")
        return
    resize = args.resize
    max_slices = args.max_slices

    # Initialize 3D volume, coverage map, landmarks
    volume = np.zeros((resize, resize, max_slices), dtype=np.float32)
    coverage = np.zeros_like(volume, dtype=np.float32)
    landmark_points = []

    # PyVista plotter setup
    grid = pv.UniformGrid()
    grid.dimensions = np.array(volume.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    grid.cell_arrays["values"] = volume.flatten(order="F")

    pl = BackgroundPlotter()
    vol_actor = pl.add_volume(grid, cmap="gray", opacity="linear", shade=True)
    pl.add_axes()
    pl.show_grid()

    # Coverage overlay
    coverage_grid = pv.UniformGrid()
    coverage_grid.dimensions = np.array(volume.shape) + 1
    coverage_grid.spacing = (1, 1, 1)
    coverage_grid.origin = (0, 0, 0)
    coverage_grid.cell_arrays["coverage"] = coverage.flatten(order="F")
    coverage_actor = pl.add_volume(coverage_grid, cmap="coolwarm", opacity="linear")

    # Guidance arrow placeholder
    guidance_actor = None

    slice_idx = 0

    while ret:
        gray = preprocess_frame(frame, resize)
        if sr_model:
            gray = super_resolve_frame(sr_model, gray, device=device)

        # Update volume: simple rolling buffer
        volume[:, :, slice_idx % max_slices] = gray
        coverage[:, :, slice_idx % max_slices] += 1

        # Simulate landmarks per frame
        num_landmarks = 5
        landmarks_2d = np.column_stack(
            (
                np.random.randint(0, resize, size=num_landmarks),
                np.random.randint(0, resize, size=num_landmarks),
            )
        )
        z = slice_idx % max_slices
        for x, y in landmarks_2d:
            landmark_points.append([x, y, z])
            coverage[int(x), int(y), int(z)] += 1

        # Apply Gaussian smoothing
        vol_smoothed = gaussian_filter(volume, sigma=args.smooth_sigma)

        # Normalize
        vol_norm = (vol_smoothed - np.min(vol_smoothed)) / (
            np.max(vol_smoothed) - np.min(vol_smoothed) + 1e-8
        )
        grid.cell_arrays["values"] = vol_norm.flatten(order="F")
        vol_actor.mapper.update()

        # Update coverage overlay
        coverage_grid.cell_arrays["coverage"] = coverage.flatten(order="F")
        coverage_actor.mapper.update()

        # Update landmarks
        if landmark_points:
            landmark_array = np.array(landmark_points)
            if hasattr(pl, "landmark_actor"):
                pl.remove_actor(pl.landmark_actor)
            pl.landmark_actor = pl.add_points(
                landmark_array,
                color="red",
                point_size=10,
                render_points_as_spheres=True,
            )

        # Update guidance arrow
        idx_min = np.unravel_index(np.argmin(coverage), coverage.shape)
        current_pos = np.array([resize // 2, resize // 2, slice_idx % max_slices])
        vector = np.array(idx_min) - current_pos
        if guidance_actor:
            pl.remove_actor(guidance_actor)
        guidance_actor = pl.add_mesh(
            pv.Arrow(start=current_pos, direction=vector, scale=20), color="green"
        )

        # Render
        pl.render()
        slice_idx += 1

        # Wait for next frame
        time.sleep(args.frame_delay)
        ret, frame = cap.read()

    cap.release()
    print("✅ Video streaming simulation complete.")
    pl.show()


if __name__ == "__main__":
    main()

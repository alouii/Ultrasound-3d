import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pyvista as pv
from collections import deque
import time
from typing import Sequence

# -------------------------------
# Optional: CUDA acceleration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Simple Super-Resolution Model (placeholder)
# -------------------------------
class SimpleSR(torch.nn.Module):
    def __init__(self, upscale=2):
        super().__init__()
        self.upscale = upscale
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.interpolate(
            x, scale_factor=self.upscale, mode="bilinear", align_corners=False
        )
        return x


# -------------------------------
# Temporal 3D Ultrasound Reconstruction
# -------------------------------
def reconstruct_volume(frames: Sequence[np.ndarray], depth_scale: float = 0.02, time_decay: float = 0.9) -> np.ndarray:
    """
    Create a 3D volume from 2D frames using temporal accumulation.

    Args:
        frames: Sequence of 2D numpy arrays (H x W), uint8 (0-255) or float (0-1).
        depth_scale: (unused) depth step per frame retained for API compatibility.
        time_decay: multiplicative decay factor applied to older frames.

    Returns:
        A 3D numpy array shaped (H, W, len(frames)) with values normalized to [0, 1].
    """
    h, w = frames[0].shape
    volume = torch.zeros((h, w, len(frames)), device=device)
    depth = 0.0

    for i, f in enumerate(frames):
        f_t = torch.tensor(f, dtype=torch.float32, device=device) / 255.0
        f_t = f_t * (time_decay ** (len(frames) - i - 1))  # decay older frames
        volume[:, :, i] = f_t
        depth += depth_scale

    return volume.cpu().numpy()


# -------------------------------
# PyVista Rendering
# -------------------------------
def render_volume(volume, threshold=0.3):
    """
    Render 3D volume using PyVista (works with all versions).
    """
    volume = np.clip(volume, 0, 1).astype(np.float32)
    dims = np.array(volume.shape)
    print(f"[INFO] Rendering volume shape: {dims}")

    # --- Create structured or uniform grid safely ---
    try:
        grid = pv.UniformGrid(
            dimensions=dims + 1,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
    except AttributeError:
        x = np.arange(dims[0], dtype=np.float32)
        y = np.arange(dims[1], dtype=np.float32)
        z = np.arange(dims[2], dtype=np.float32)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        grid = pv.StructuredGrid(X, Y, Z)

    grid.points = grid.points.astype(np.float32)

    # Match data size to cell count
    if grid.n_cells == volume.size:
        grid.cell_data["values"] = volume.flatten(order="F")
    else:
        cell_volume = volume[:-1, :-1, :-1]
        grid.cell_data["values"] = cell_volume.flatten(order="F")

    # --- Volume Rendering ---
    p = pv.Plotter()
    p.add_volume(
        grid,
        cmap="plasma",
        opacity="sigmoid",
        opacity_unit_distance=2.0,
        shade=True,
    )
    p.add_axes()
    p.add_text("Ultrasound 3D Reconstruction", font_size=12)
    p.show()


# -------------------------------
# Main Logic
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Path to ultrasound video")
    parser.add_argument(
        "--frames", type=int, default=50, help="Number of frames to accumulate"
    )
    parser.add_argument(
        "--sr", action="store_true", help="Enable simple super-resolution"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="Rendering threshold"
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.video}")
        return

    sr_model = SimpleSR().to(device) if args.sr else None
    buffer = deque(maxlen=args.frames)

    print("[INFO] Processing video...")
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Optional super-resolution enhancement ---
        if sr_model:
            with torch.no_grad():
                inp = (
                    torch.tensor(gray, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    / 255.0
                )
                out = sr_model(inp)
                gray = (out.squeeze().cpu().numpy() * 255).astype(np.uint8)

        buffer.append(gray)
        frame_count += 1

        # Display progress
        cv2.imshow("Ultrasound Feed", gray)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Processed {frame_count} frames in {time.time() - start_time:.2f}s")

    # --- Reconstruct 3D volume ---
    if len(buffer) > 0:
        volume = reconstruct_volume(list(buffer))
        render_volume(volume, threshold=args.threshold)
    else:
        print("[ERROR] No frames captured.")


if __name__ == "__main__":
    main()
# python ultrasound_3d_prototype.py --video ultrasound_clip.mp4 --frames 60 --sr

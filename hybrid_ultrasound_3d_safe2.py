import cv2
import numpy as np
import argparse
from tqdm import tqdm
import pyvista as pv
from scipy.ndimage import zoom, gaussian_filter


def optimize_2d_frame(frame, resize=256):
    """Améliore le contraste et la netteté d'une coupe échographique."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resized = cv2.resize(normalized, (resize, resize))
    return resized


def load_video_as_volume(video_path, resize=256, max_frames=800):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    print(f"Chargement de {min(total, max_frames)} frames...")

    for i in tqdm(range(min(total, max_frames))):
        ret, frame = cap.read()
        if not ret:
            break
        optimized = optimize_2d_frame(frame, resize)
        frames.append(optimized)
    cap.release()
    volume = np.stack(frames, axis=-1)
    print(f"Volume brut : {volume.shape} (H x W x Z)")
    return volume


def interpolate_volume(volume, factor=2, smooth_sigma=1.2):
    """Interpolation et lissage 3D pour un rendu plus fluide."""
    vol_interp = zoom(volume, (1, 1, factor), order=1)
    vol_smooth = gaussian_filter(vol_interp, sigma=smooth_sigma)
    return vol_smooth


def render_volume(volume, opacity=0.3):
    """Visualisation 3D fidèle au contraste original."""
    print("Visualisation du volume 3D...")
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)
    grid.point_data["intensity"] = volume.flatten(order="F")

    pl = pv.Plotter()
    pl.add_volume(grid, cmap="gray", opacity=opacity)
    pl.add_axes()
    pl.show_grid()
    pl.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Ultrasound realistic volume reconstruction"
    )
    parser.add_argument(
        "--video", required=True, help="Chemin vers la vidéo échographique"
    )
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=800)
    parser.add_argument("--interpolate", type=int, default=2)
    parser.add_argument("--smooth-sigma", type=float, default=1.2)
    parser.add_argument("--opacity", type=float, default=0.3)
    args = parser.parse_args()

    volume = load_video_as_volume(
        args.video, resize=args.resize, max_frames=args.max_frames
    )
    volume = interpolate_volume(
        volume, factor=args.interpolate, smooth_sigma=args.smooth_sigma
    )
    render_volume(volume, opacity=args.opacity)

import cv2
import numpy as np
import argparse
import open3d as o3d
from tqdm import tqdm


def optimize_2d_frame(frame: np.ndarray, crop_ratio: float = 0.1, target_size: int = 128) -> np.ndarray:
    """Enhance and resize a 2D ultrasound frame.

    Returns a uint8 grayscale image of shape (target_size, target_size).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = normalized.shape
    y1, y2 = int(h * crop_ratio), int(h * (1 - crop_ratio))
    x1, x2 = int(w * crop_ratio), int(w * (1 - crop_ratio))
    roi = normalized[y1:y2, x1:x2]
    roi = cv2.resize(roi, (target_size, target_size))
    return roi


def load_video_in_chunks(video_path, resize=128, crop_ratio=0.1, max_frames=1000):
    """Load and preprocess frames in a memory-safe way."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    print(f"Loading up to {max_frames} frames from {video_path} ...")

    for i in tqdm(range(min(total, max_frames))):
        ret, frame = cap.read()
        if not ret:
            break
        optimized = optimize_2d_frame(frame, crop_ratio=crop_ratio, target_size=resize)
        frames.append(optimized)

    cap.release()
    volume = np.stack(frames, axis=-1)
    print(f"Volume shape: {volume.shape} (H x W x Z)")
    return volume


def visualize_volume(volume, threshold=0.5, voxel_size=2.0, downsample=2):
    """Convert 3D numpy volume into voxel grid for visualization."""
    print("Converting volume to voxel grid safely...")

    # Downsample volume to avoid overload
    volume = volume[::downsample, ::downsample, ::downsample]
    vol_norm = volume / 255.0
    mask = vol_norm > threshold

    points = np.argwhere(mask)
    if len(points) == 0:
        print("⚠️ No valid 3D structure found above threshold.")
        return

    points = points.astype(np.float32)
    points -= np.mean(points, axis=0)
    points *= voxel_size

    print(f"Generating point cloud with {len(points)} points...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Optional: color points according to slice index
    colors = np.zeros_like(points)
    colors[:, 2] = np.linspace(0, 1, len(points))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [pcd], window_name="3D Ultrasound Volume (Safe Mode)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Memory-safe 3D Ultrasound Volume Reconstruction"
    )
    parser.add_argument("--video", required=True, help="Path to ultrasound video file")
    parser.add_argument(
        "--resize", type=int, default=128, help="Resize each frame (smaller = safer)"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Voxel threshold")
    parser.add_argument(
        "--voxel-size", type=float, default=2.0, help="Voxel scaling factor"
    )
    parser.add_argument(
        "--crop-ratio", type=float, default=0.1, help="Fraction to crop edges"
    )
    parser.add_argument(
        "--max-frames", type=int, default=1000, help="Max frames to process"
    )
    parser.add_argument(
        "--downsample", type=int, default=2, help="3D volume downsampling factor"
    )

    args = parser.parse_args()

    volume = load_video_in_chunks(
        args.video,
        resize=args.resize,
        crop_ratio=args.crop_ratio,
        max_frames=args.max_frames,
    )
    visualize_volume(
        volume,
        threshold=args.threshold,
        voxel_size=args.voxel_size,
        downsample=args.downsample,
    )

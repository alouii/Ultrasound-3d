import cv2
import numpy as np
import argparse
import open3d as o3d
from tqdm import tqdm


# =========================
# 2D Image Optimization
# =========================
def optimize_2d_frame(frame, crop_ratio=0.1):
    """Enhance 2D ultrasound image before 3D volume acquisition."""
    # Convert to grayscale if needed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    # Denoise (remove speckle noise)
    denoised = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)

    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Normalize intensity to [0,1]
    normalized = cv2.normalize(enhanced, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Crop center region to keep consistent FOV
    h, w = normalized.shape
    y1, y2 = int(h * crop_ratio), int(h * (1 - crop_ratio))
    x1, x2 = int(w * crop_ratio), int(w * (1 - crop_ratio))
    roi = normalized[y1:y2, x1:x2]

    # Resize to consistent target
    roi = cv2.resize(roi, (256, 256))
    return roi


# =========================
# Load and preprocess video
# =========================
def load_video_as_volume(video_path, resize=256, crop_ratio=0.1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    frames = []
    print(f"Loading video: {video_path}")
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break

        # Optimize each 2D frame before stacking
        optimized = optimize_2d_frame(frame, crop_ratio=crop_ratio)
        frames.append(optimized)

    cap.release()
    volume = np.stack(frames, axis=-1)
    print(f"Volume shape: {volume.shape} (H x W x Z)")
    return volume


# =========================
# 3D Volume Visualization
# =========================
def visualize_volume(volume, threshold=0.5, voxel_size=1.0):
    """Convert 3D numpy volume into a voxel representation for visualization."""
    print("Converting volume to voxel mesh...")
    volume = (volume > threshold).astype(np.float32)

    # Convert to point cloud
    points = np.argwhere(volume > 0)
    if len(points) == 0:
        print("⚠️ No points found above threshold — empty volume.")
        return

    # Normalize and scale points
    points = points.astype(np.float32)
    points -= np.mean(points, axis=0)
    points /= np.max(np.linalg.norm(points, axis=1))
    points *= voxel_size * 100  # scale for visualization

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals for smoother visualization
    pcd.estimate_normals()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=voxel_size * 3.0
    )
    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([mesh], window_name="3D Ultrasound Volume")


# =========================
# Main Script
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Ultrasound 3D Volume Reconstruction (Optimized 2D Frames)"
    )
    parser.add_argument("--video", required=True, help="Path to ultrasound video file")
    parser.add_argument(
        "--resize", type=int, default=256, help="Resize each frame to (resize x resize)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4, help="Threshold for 3D voxel generation"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="Voxel size for visualization scale",
    )
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=0.1,
        help="Crop ratio (fraction to remove from edges)",
    )

    args = parser.parse_args()

    volume = load_video_as_volume(
        args.video, resize=args.resize, crop_ratio=args.crop_ratio
    )
    visualize_volume(volume, threshold=args.threshold, voxel_size=args.voxel_size)

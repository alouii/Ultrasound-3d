import cv2
import numpy as np
import open3d as o3d
import argparse
from tqdm import tqdm
import os

def load_video_frames(video_path, resize=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Loading {total} frames from {video_path}...")

    for _ in tqdm(range(total), desc="Reading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize:
            gray = cv2.resize(gray, (resize, resize))
        frames.append(gray.astype(np.float32) / 255.0)

    cap.release()
    return np.stack(frames, axis=-1)  # Shape: (H, W, N)

def preprocess_volume(volume, smoothing=True):
    # Optional smoothing across slices
    if smoothing:
        import scipy.ndimage as ndi
        volume = ndi.gaussian_filter(volume, sigma=(1, 1, 2))
    volume = np.clip(volume, 0, 1)
    return volume

def visualize_volume(volume, threshold=0.5, voxel_size=1.0, save_path="ultrasound_mesh.ply"):
    print("Thresholding and building 3D volume...")

    mask = volume > (np.max(volume) * threshold)
    coords = np.argwhere(mask)
    if coords.size == 0:
        print("⚠️ No voxels above threshold. Try lowering --threshold.")
        return

    # Normalize coordinates
    coords = coords.astype(np.float32)
    coords -= np.mean(coords, axis=0)
    coords /= np.max(np.abs(coords)) + 1e-5
    coords *= 100  # scale for visibility

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    intensities = volume[mask]
    colors = np.stack([intensities]*3, axis=-1)
    colors = (colors - colors.min()) / (colors.max() + 1e-8)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Estimate normals
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))

    # Surface reconstruction
    print("Reconstructing 3D surface using Poisson...")
    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    except Exception as e:
        print(f"⚠️ Poisson reconstruction failed: {e}")
        return

    # Clean up mesh
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    mesh.compute_vertex_normals()

    # Save result
    o3d.io.write_triangle_mesh(save_path, mesh)
    print(f"✅ Mesh saved to {save_path}")

    # Visualize
    print("Displaying 3D mesh...")
    o3d.visualization.draw_geometries([mesh])

def main():
    parser = argparse.ArgumentParser(description="3D Ultrasound Reconstruction from 2D Frames")
    parser.add_argument("--video", required=True, help="Path to ultrasound video file")
    parser.add_argument("--resize", type=int, default=None, help="Resize frames to this dimension")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for 3D voxel activation")
    parser.add_argument("--voxel-size", type=float, default=1.0, help="Voxel size for visualization")
    parser.add_argument("--preview", action="store_true", help="Preview input frames before reconstruction")
    args = parser.parse_args()

    volume = load_video_frames(args.video, resize=args.resize)
    print(f"Volume shape: {volume.shape} (H x W x Frames)")

    if args.preview:
        import matplotlib.pyplot as plt
        mid = volume.shape[-1] // 2
        plt.imshow(volume[..., mid], cmap="gray")
        plt.title("Middle Frame")
        plt.show()

    volume = preprocess_volume(volume)
    visualize_volume(volume, threshold=args.threshold, voxel_size=args.voxel_size)

if __name__ == "__main__":
    main()

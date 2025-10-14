import argparse
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def video_to_volume(video_path, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(frame_count, max_frames)
    frames = []

    for _ in tqdm(range(total), desc="Loading video"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        gray = gray.astype(np.float32) / 255.0
        frames.append(gray)
    cap.release()
    volume = np.stack(frames, axis=2)  # Shape: H x W x F
    # Smooth in 3D
    volume = gaussian_filter(volume, sigma=1)
    return volume

def volume_to_mesh(volume, voxel_size=0.01, threshold=0.2):
    # Convert volume to Open3D voxel grid
    dims = volume.shape
    voxels = []
    colors = []

    # Threshold to select “solid” voxels
    indices = np.argwhere(volume > threshold)
    for idx in indices:
        y, x, z = idx
        voxels.append([x*voxel_size, y*voxel_size, z*voxel_size])
        intensity = volume[y, x, z]
        colors.append([intensity, intensity, intensity])

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.array(voxels))
    pc.colors = o3d.utility.Vector3dVector(np.array(colors))

    return pc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-video', type=str, required=True)
    parser.add_argument('--output', type=str, default='volume.ply')
    parser.add_argument('--max-frames', type=int, default=150)
    parser.add_argument('--voxel-size', type=float, default=0.01)
    parser.add_argument('--threshold', type=float, default=0.2)
    args = parser.parse_args()

    volume = video_to_volume(args.input_video, args.max_frames)
    pc = volume_to_mesh(volume, voxel_size=args.voxel_size, threshold=args.threshold)

    o3d.io.write_point_cloud(args.output, pc)
    print(f"✅ 3D volume saved to {args.output}")
    o3d.visualization.draw_geometries([pc])

if __name__ == '__main__':
    main()

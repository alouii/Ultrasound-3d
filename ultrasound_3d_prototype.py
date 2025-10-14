import argparse
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def video_to_volume(video_path, max_frames=300, resize=256, smooth_sigma=1.0):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(frame_count, max_frames)
    frames = []

    prev_gray = None
    for _ in tqdm(range(total), desc="Loading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (resize, resize))
        gray = gray.astype(np.float32) / 255.0

        # Temporal difference for edges
        if prev_gray is not None:
            gray = np.abs(gray - prev_gray)
        prev_gray = gray.copy()

        frames.append(gray)
    
    cap.release()
    volume = np.stack(frames, axis=2)  # H x W x Frames
    # 3D Gaussian smoothing for precision
    volume = gaussian_filter(volume, sigma=smooth_sigma)
    return volume

def volume_to_pointcloud(volume, voxel_size=0.005, percentile=80):
    # Threshold based on percentile to keep fine structures
    threshold = np.percentile(volume, percentile)
    indices = np.argwhere(volume > threshold)
    colors = []
    for idx in indices:
        y, x, z = idx
        colors.append([volume[y,x,z]]*3)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(indices * voxel_size)
    pc.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-video', type=str, required=True)
    parser.add_argument('--output', type=str, default='high_precision.ply')
    parser.add_argument('--max-frames', type=int, default=150)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--voxel-size', type=float, default=0.005)
    parser.add_argument('--smooth-sigma', type=float, default=1.0)
    parser.add_argument('--percentile', type=float, default=80)
    args = parser.parse_args()

    volume = video_to_volume(args.input_video, max_frames=args.max_frames, 
                             resize=args.resize, smooth_sigma=args.smooth_sigma)
    pc = volume_to_pointcloud(volume, voxel_size=args.voxel_size, percentile=args.percentile)

    o3d.io.write_point_cloud(args.output, pc)
    print(f" High-precision 3D reconstruction saved to {args.output}")
    o3d.visualization.draw_geometries([pc])

if __name__ == '__main__':
    main()

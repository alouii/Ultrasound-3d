import argparse
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

def video_to_depth(video_path, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(frame_count, max_frames)
    depths = []

    for _ in tqdm(range(total), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))
        # Invert intensity so bright pixels are closer
        depth = 1.0 - (gray.astype(np.float32) / 255.0)
        depths.append(depth)
    cap.release()
    return np.array(depths)

def reconstruct_point_cloud(depth_stack, scale=1.0):
    n_frames, h, w = depth_stack.shape
    points = []
    colors = []

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    for i in range(n_frames):
        depth = depth_stack[i]
        # Use intensity as Z
        x = xx * scale
        y = yy * scale
        z = depth * scale  # <-- depth now projects per pixel
        pts = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
        col = np.repeat(depth.flatten()[:, None], 3, axis=1)  # grayscale color
        points.append(pts)
        colors.append(col)

    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-video', type=str, required=True)
    parser.add_argument('--output', type=str, default='reconstruction.ply')
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--scale', type=float, default=1.0)
    args = parser.parse_args()

    depth_stack = video_to_depth(args.input_video, args.max_frames)
    pc = reconstruct_point_cloud(depth_stack, scale=args.scale)

    o3d.io.write_point_cloud(args.output, pc)
    print(f"✅ 3D reconstruction saved to {args.output}")

    o3d.visualization.draw_geometries([pc])

if __name__ == '__main__':
    main()

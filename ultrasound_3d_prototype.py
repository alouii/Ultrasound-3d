import argparse
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage import exposure

def preprocess_frame(frame, resize=256):
    """Grayscale, resize, CLAHE for contrast, normalize."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (resize, resize))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = gray.astype(np.float32) / 255.0
    return gray

def load_video_as_volume(video_path, max_frames=200, resize=256, smooth_sigma=1.0):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(total_frames, max_frames)
    frames = []
    prev_gray = None

    for _ in tqdm(range(total), desc="Loading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = preprocess_frame(frame, resize=resize)

        # Temporal edge detection: highlight changes across frames
        if prev_gray is not None:
            gray = np.abs(gray - prev_gray)
        prev_gray = gray.copy()
        
        frames.append(gray)

    cap.release()
    volume = np.stack(frames, axis=2)  # H x W x Frames

    # 3D Gaussian smoothing for noise reduction
    volume = gaussian_filter(volume, sigma=smooth_sigma)
    return volume

def volume_to_mesh(volume, voxel_size=0.005, percentile=80):
    """Convert 3D volume to mesh using Marching Cubes approximation."""
    threshold = np.percentile(volume, percentile)
    # Get voxel indices above threshold
    indices = np.argwhere(volume > threshold)
    if len(indices) == 0:
        raise ValueError("No voxels above threshold; lower percentile or adjust preprocessing.")
    
    # Color mapping from intensity
    colors = [volume[y, x, z] for y, x, z in indices]
    colors = np.stack([colors, colors, colors], axis=1)

    # Create point cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(indices * voxel_size)
    pc.colors = o3d.utility.Vector3dVector(colors)

    # Optional: voxel downsampling to reduce memory
    pc = pc.voxel_down_sample(voxel_size=voxel_size)

    # Estimate normals for mesh generation
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))

    # Create mesh using Ball Pivoting or Poisson
    try:
        distances = pc.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pc, o3d.utility.DoubleVector([avg_dist*1.5, avg_dist*2])
        )
        bpa_mesh.compute_vertex_normals()
        return pc, bpa_mesh
    except Exception as e:
        print(f"Mesh generation failed: {e}. Returning only point cloud.")
        return pc, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-video', type=str, required=True)
    parser.add_argument('--output-ply', type=str, default='ultrasound_volume.ply')
    parser.add_argument('--output-mesh', type=str, default='ultrasound_mesh.ply')
    parser.add_argument('--max-frames', type=int, default=200)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--voxel-size', type=float, default=0.005)
    parser.add_argument('--smooth-sigma', type=float, default=1.0)
    parser.add_argument('--percentile', type=float, default=80)
    args = parser.parse_args()

    print("🎥 Loading and preprocessing video...")
    volume = load_video_as_volume(args.input_video, max_frames=args.max_frames,
                                  resize=args.resize, smooth_sigma=args.smooth_sigma)
    print(f"Volume shape: {volume.shape}")

    print(" Converting volume to point cloud and mesh...")
    pc, mesh = volume_to_mesh(volume, voxel_size=args.voxel_size, percentile=args.percentile)

    o3d.io.write_point_cloud(args.output_ply, pc)
    print(f"Point cloud saved to {args.output_ply}")
    if mesh is not None:
        o3d.io.write_triangle_mesh(args.output_mesh, mesh)
        print(f" Mesh saved to {args.output_mesh}")

    print("Launching visualization...")
    if mesh is not None:
        o3d.visualization.draw_geometries([mesh, pc])
    else:
        o3d.visualization.draw_geometries([pc])

if __name__ == '__main__':
    main()

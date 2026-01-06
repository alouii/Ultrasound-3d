#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import open3d as o3d

def video_to_volume(video_path, resize=256, preview=False):
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (resize, resize))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32) / 255.0)  # normalize 0-1

        if preview and idx % 10 == 0:
            cv2.imshow("Frame preview", gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        idx += 1

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    volume = np.stack(frames, axis=-1)  # shape: H x W x Z
    print(f"Volume shape: {volume.shape} (H x W x Z)")

    return volume

def volume_to_mesh(volume, threshold=0.5, voxel_size=1.0):
    # Convert to voxel grid
    voxels = (volume > threshold).astype(np.uint8)
    # Use Open3D VoxelGrid
    dims = voxels.shape
    voxel_list = []
    for z in range(dims[2]):
        for y in range(dims[0]):
            for x in range(dims[1]):
                if voxels[y, x, z]:
                    voxel_list.append(o3d.geometry.VoxelGrid.create_dense(
                        origin=[x*voxel_size, y*voxel_size, z*voxel_size],
                        voxel_size=voxel_size,
                        width=1, height=1, depth=1
                    ))
    if not voxel_list:
        print("⚠️ No voxels above threshold. Try lowering threshold.")
        return None
    # Merge all voxel grids
    voxel_grid = voxel_list[0]
    for v in voxel_list[1:]:
        voxel_grid += v

    return voxel_grid

def visualize_volume(volume, threshold=0.5, voxel_size=1.0):
    print("Converting volume to voxel mesh...")
    voxels = (volume > threshold).astype(np.uint8)
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxels = [o3d.geometry.VoxelGrid.Voxel([x, y, z], [1,1,1])
                         for z in range(voxels.shape[2])
                         for y in range(voxels.shape[0])
                         for x in range(voxels.shape[1])
                         if voxels[y, x, z]]
    print(f"Number of voxels: {len(voxel_grid.voxels)}")
    o3d.visualization.draw_geometries([voxel_grid])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to ultrasound video")
    parser.add_argument("--resize", type=int, default=256, help="Resize width/height")
    parser.add_argument("--threshold", type=float, default=0.5, help="Voxel threshold")
    parser.add_argument("--voxel-size", type=float, default=1.0, help="Voxel size")
    parser.add_argument("--preview", action="store_true", help="Show frames preview")
    args = parser.parse_args()

    volume = video_to_volume(args.video, resize=args.resize, preview=args.preview)
    visualize_volume(volume, threshold=args.threshold, voxel_size=args.voxel_size)

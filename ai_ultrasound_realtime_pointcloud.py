#!/usr/bin/env python3
"""Real-time point-cloud ultrasound reconstruction demo.

- Streams frames from a video (uses existing threaded_frame_reader behavior)
- Builds a bounded point cloud by sampling high-intensity pixels per frame
- Displays point cloud live using PyVista (if available) or logs headless
- Runs periodic background Poisson reconstructions (on a downsampled cloud)

Usage (headless):
  python ai_ultrasound_realtime_pointcloud.py --video ./usliverseq-mp4/volunteer02.mp4 --headless --recon-period 300

"""

import argparse
import threading
import time
from collections import deque
import os

import cv2
import numpy as np
from scipy.ndimage import median_filter

# Optional heavy libs (import lazily)
try:
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter
except Exception:
    pv = None
    BackgroundPlotter = None

try:
    import open3d as o3d
except Exception:
    o3d = None

from ai_ultrasound_realtime_imageData import threaded_frame_reader, preprocess_frame


def sample_points_from_frame(frame, zpos, max_samples=2000, percentile=99.0, factor=0.6):
    """Return sampled (N,4) array: x,y,z,intensity in normalized coords."""
    # frame: HxW float32 [0,1]
    # compute threshold
    th = np.percentile(frame, percentile) * factor
    mask = frame > th
    if not mask.any():
        return np.zeros((0, 4), dtype=np.float32)

    # median to reduce speckle
    mask = median_filter(mask.astype(np.uint8), size=3).astype(bool)
    coords = np.column_stack(np.nonzero(mask))  # rows(y), cols(x)
    if coords.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # sample if too many
    if coords.shape[0] > max_samples:
        choice = np.random.choice(coords.shape[0], max_samples, replace=False)
        coords = coords[choice]

    intensities = frame[coords[:, 0], coords[:, 1]]
    # convert to x, y (cols, rows)
    points = np.column_stack([coords[:, 1], coords[:, 0], np.full(len(coords), zpos)])
    return np.column_stack([points, intensities])


class LivePointCloud:
    def __init__(self, max_points=200000):
        self.max_points = max_points
        self.points = deque(maxlen=max_points)  # stores (x,y,z,int)
        self.lock = threading.Lock()

    def add(self, pts):
        if pts is None or len(pts) == 0:
            return
        with self.lock:
            for r in pts:
                self.points.append(tuple(r))

    def to_numpy(self):
        with self.lock:
            if len(self.points) == 0:
                return np.zeros((0, 4), dtype=np.float32)
            return np.array(self.points, dtype=np.float32)

    def clear(self):
        with self.lock:
            self.points.clear()


def run_poisson_async(pc: LivePointCloud, out_path: str, voxel_size=1.0, depth=8):
    """Run a Poisson reconstruction on a downsampled copy of the point cloud in a background thread."""

    def _worker(points_copy):
        if o3d is None:
            print("Open3D not available; skipping Poisson.")
            return
        if points_copy.shape[0] < 50:
            print("Not enough points for Poisson:", points_copy.shape[0])
            return
        print(f"[recon] Running Poisson on {points_copy.shape[0]} points...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_copy[:, :3])
        # colors or intensities are optional
        # downsample
        pcd = pcd.voxel_down_sample(voxel_size=max(1.0, voxel_size))
        if len(pcd.points) < 50:
            print("[recon] Too few points after downsample; abort.")
            return
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
        try:
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
            bbox = pcd.get_axis_aligned_bounding_box()
            mesh = mesh.crop(bbox)
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(out_path, mesh)
            print(f"[recon] Poisson mesh saved to {out_path}")
        except Exception as e:
            print('[recon] Poisson failed:', e)

    pts = pc.to_numpy()
    if pts.shape[0] == 0:
        print('[recon] no points available')
        return
    # sample to reasonable size for Poisson
    sample_n = min(200000, pts.shape[0])
    if pts.shape[0] > sample_n:
        idx = np.random.choice(pts.shape[0], sample_n, replace=False)
        pts = pts[idx]

    t = threading.Thread(target=_worker, args=(pts.copy(),), daemon=True)
    t.start()
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--samples-per-frame", type=int, default=1500)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--factor", type=float, default=0.6)
    parser.add_argument("--recon-period", type=int, default=600, help="Run Poisson every N frames")
    parser.add_argument("--recon-depth", type=int, default=8)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pc = LivePointCloud(max_points=args.max_points)

    # PyVista plotter if available and not headless
    plotter = None
    pts_actor = None
    if not args.headless and BackgroundPlotter is not None and pv is not None:
        plotter = BackgroundPlotter(title="Live Point Cloud")
        plotter.add_axes()

    stop_flag = threading.Event()

    def stream_thread():
        slice_idx = 0
        recon_count = 0
        for idx, frame, landmarks in threaded_frame_reader(
            args.video,
            max_frames=None,
            resize=args.resize,
            sr_model=None,
            device='cpu',
            threads=2,
        ):
            if stop_flag.is_set():
                break
            zpos = slice_idx
            # sample points from frame
            samp = sample_points_from_frame(frame, zpos, max_samples=args.samples_per_frame,
                                           percentile=args.percentile, factor=args.factor)
            if samp.shape[0] > 0:
                pc.add(samp)

            # Update visualizer
            if plotter is not None:
                pts_np = pc.to_numpy()
                if pts_np.shape[0] > 0:
                    pts_xyz = pts_np[:, :3]
                    colors = np.clip(pts_np[:, 3:4], 0, 1)
                    colors = np.concatenate([colors, colors, colors], axis=1)
                    if pts_actor:
                        try:
                            plotter.remove_actor(pts_actor)
                        except Exception:
                            pass
                    pts_actor = plotter.add_points(pts_xyz, scalars=None, render_points_as_spheres=False)

            slice_idx += 1
            recon_count += 1

            if args.recon_period > 0 and (recon_count % args.recon_period == 0):
                out_mesh = os.path.join(args.out_dir, f"live_recon_{slice_idx}.ply")
                print(f"Triggering reconstruction at frame {slice_idx} -> {out_mesh}")
                run_poisson_async(pc, out_mesh, voxel_size=args.voxel_size, depth=args.recon_depth)

            time.sleep(0.01)

    t = threading.Thread(target=stream_thread, daemon=True)
    t.start()

    try:
        if plotter is not None:
            plotter.app.exec_()
        else:
            # headless run until stream finishes
            t.join()
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        t.join(timeout=2)


if __name__ == "__main__":
    main()

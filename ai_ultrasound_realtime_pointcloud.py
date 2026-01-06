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
            if mesh is None or len(mesh.triangles) == 0:
                print("[recon] Poisson produced empty mesh; skipping save.")
                return
            bbox = pcd.get_axis_aligned_bounding_box()
            ext = np.array(bbox.get_extent())
            if not np.isfinite(ext).all() or (ext.min() <= 1e-6):
                print("[recon] Bounding box invalid or degenerate; skipping crop.")
            else:
                try:
                    mesh = mesh.crop(bbox)
                except Exception as e:
                    print('[recon] Crop failed, skipping crop step:', e)
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


def points_to_o3d(pts):
    """Convert Nx>=3 array to an Open3D PointCloud."""
    if o3d is None or pts.shape[0] == 0:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    return pcd


def apply_transform_to_points(pts, T):
    """Apply 4x4 transform T to Nx4 pts (x,y,z,intensity) and return Nx4 array."""
    if pts.shape[0] == 0:
        return pts
    R = T[:3, :3]
    t = T[:3, 3]
    xyz = pts[:, :3].dot(R.T) + t
    if pts.shape[1] > 3:
        return np.hstack([xyz, pts[:, 3:4]])
    else:
        return xyz


def icp_register(src_pts, target_pcd, voxel_size=2.0, threshold=10.0, max_iter=50):
    """Register src_pts (Nx>=3 array) to target_pcd (Open3D PointCloud) using ICP and return 4x4 T."""
    if o3d is None or target_pcd is None or src_pts.shape[0] == 0:
        return np.eye(4)
    src_pcd = points_to_o3d(src_pts)
    if src_pcd is None:
        return np.eye(4)
    try:
        src_down = src_pcd.voxel_down_sample(voxel_size)
        target_down = target_pcd.voxel_down_sample(voxel_size)
        if len(src_down.points) < 10 or len(target_down.points) < 10:
            return np.eye(4)
        src_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        icp_res = o3d.pipelines.registration.registration_icp(
            src_down, target_down, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )
        return icp_res.transformation
    except Exception as e:
        print('[icp] failed:', e)
        return np.eye(4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (for quick runs)")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--samples-per-frame", type=int, default=1500)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--factor", type=float, default=0.6)
    parser.add_argument("--recon-period", type=int, default=600, help="Run Poisson every N frames")
    parser.add_argument("--recon-depth", type=int, default=8)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--register", choices=["none","frame_icp","global_icp"], default="frame_icp",
                        help="Type of per-frame registration to perform")
    parser.add_argument("--icp-voxel", type=float, default=2.0, help="Voxel size for ICP downsampling")
    parser.add_argument("--icp-threshold", type=float, default=10.0, help="ICP max correspondence distance")
    parser.add_argument("--icp-max-iter", type=int, default=50, help="ICP max iterations")
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
        # for registration
        prev_pcd = None
        global_pcd = None
        for idx, frame, landmarks in threaded_frame_reader(
            args.video,
            max_frames=args.max_frames,
            resize=args.resize,
            sr_model=None,
            device='cpu',
            threads=2,
        ):
            if stop_flag.is_set():
                break

            # Registration-enabled path (frame_icp/global_icp) uses ICP to align incoming per-frame samples
            if args.register == 'none':
                zpos = slice_idx
                samp = sample_points_from_frame(frame, zpos, max_samples=args.samples_per_frame,
                                               percentile=args.percentile, factor=args.factor)
                if samp.shape[0] > 0:
                    pc.add(samp)
            else:
                # sample without Z (we'll compute transform to place them in 3D)
                samp = sample_points_from_frame(frame, 0, max_samples=args.samples_per_frame,
                                               percentile=args.percentile, factor=args.factor)
                if samp.shape[0] == 0:
                    slice_idx += 1
                    recon_count += 1
                    time.sleep(0.01)
                    continue

                if o3d is None:
                    # fallback: apply trivial Z offset
                    samp[:, 2] = slice_idx
                    pc.add(samp)
                else:
                    if args.register == 'frame_icp':
                        if prev_pcd is None:
                            transformed = samp
                            prev_pcd = points_to_o3d(transformed)
                        else:
                            T = icp_register(samp, prev_pcd, voxel_size=args.icp_voxel,
                                             threshold=args.icp_threshold, max_iter=args.icp_max_iter)
                            transformed = apply_transform_to_points(samp, T)
                            prev_pcd = points_to_o3d(transformed)
                    elif args.register == 'global_icp':
                        if global_pcd is None:
                            transformed = samp
                            global_pcd = points_to_o3d(transformed)
                        else:
                            T = icp_register(samp, global_pcd, voxel_size=args.icp_voxel,
                                             threshold=args.icp_threshold, max_iter=args.icp_max_iter)
                            transformed = apply_transform_to_points(samp, T)
                            # merge into global_pcd
                            try:
                                g_pts = np.asarray(global_pcd.points)
                                merged = np.vstack([g_pts, transformed[:, :3]])
                                global_pcd.points = o3d.utility.Vector3dVector(merged)
                                # periodically downsample to control growth
                                if slice_idx % 50 == 0:
                                    global_pcd = global_pcd.voxel_down_sample(max(0.5, args.icp_voxel))
                            except Exception:
                                pass

                    # ensure we have intensity column
                    if transformed.shape[1] == 3:
                        if samp.shape[1] > 3:
                            intens = samp[:, 3:4]
                        else:
                            intens = np.ones((transformed.shape[0], 1), dtype=transformed.dtype)
                        transformed = np.hstack([transformed, intens])

                    pc.add(transformed)

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
                suffix = 'registered' if args.register != 'none' else 'raw'
                out_mesh = os.path.join(args.out_dir, f"live_recon_{suffix}_{slice_idx}.ply")
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

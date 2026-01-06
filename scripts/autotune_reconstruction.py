#!/usr/bin/env python3
"""Auto-tune reconstruction parameters by running short headless sweeps.

Heuristic scoring (higher is better):
 - more vertices (coverage)
 - lower triangle area std (more uniform mesh)

Saves results to outputs/autotune/ and writes best mesh + preview.
"""

import argparse
import os
import sys
import itertools
import time
import numpy as np
import open3d as o3d

# ensure project root on path so we can import helper functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_ultrasound_realtime_pointcloud import (
    sample_points_from_frame, points_to_o3d, icp_register, apply_transform_to_points,
)
from ai_ultrasound_realtime_pointcloud import threaded_frame_reader


def build_registered_pointcloud(video, max_frames, samples_per_frame, percentile, factor,
                                register='frame_icp', icp_voxel=2.0, icp_threshold=12.0, icp_max_iter=50,
                                resize=128):
    pts_list = []
    prev_pcd = None
    global_pcd = None
    frame_count = 0
    for idx, frame, landmarks in threaded_frame_reader(video, max_frames=max_frames, resize=resize, sr_model=None,
                                                       device='cpu', threads=2):
        frame_count += 1
        samp = sample_points_from_frame(frame, 0, max_samples=samples_per_frame,
                                       percentile=percentile, factor=factor)
        if samp.shape[0] == 0:
            continue
        if register == 'none' or o3d is None:
            samp[:, 2] = frame_count
            pts_list.append(samp)
        else:
            if register == 'frame_icp':
                if prev_pcd is None:
                    transformed = samp
                    prev_pcd = points_to_o3d(transformed)
                else:
                    T = icp_register(samp, prev_pcd, voxel_size=icp_voxel, threshold=icp_threshold, max_iter=icp_max_iter)
                    transformed = apply_transform_to_points(samp, T)
                    prev_pcd = points_to_o3d(transformed)
            elif register == 'global_icp':
                if global_pcd is None:
                    transformed = samp
                    global_pcd = points_to_o3d(transformed)
                else:
                    T = icp_register(samp, global_pcd, voxel_size=icp_voxel, threshold=icp_threshold, max_iter=icp_max_iter)
                    transformed = apply_transform_to_points(samp, T)
                    # merge
                    try:
                        g_pts = np.asarray(global_pcd.points)
                        merged = np.vstack([g_pts, transformed[:, :3]])
                        global_pcd.points = o3d.utility.Vector3dVector(merged)
                        if frame_count % 50 == 0:
                            global_pcd = global_pcd.voxel_down_sample(max(0.5, icp_voxel))
                    except Exception:
                        pass
            # ensure intensity present
            if transformed.shape[1] == 3:
                if samp.shape[1] > 3:
                    intens = samp[:, 3:4]
                else:
                    intens = np.ones((transformed.shape[0], 1), dtype=transformed.dtype)
                transformed = np.hstack([transformed, intens])
            pts_list.append(transformed)
    if len(pts_list) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.vstack(pts_list)


def poisson_sync(points, out_path, voxel_size=1.0, depth=8):
    if points.shape[0] == 0 or o3d is None:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd = pcd.voxel_down_sample(voxel_size=max(0.5, voxel_size))
    if len(pcd.points) < 50:
        print('[poisson_sync] too few points after downsample')
        return None
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        if mesh is None or len(mesh.triangles) == 0:
            return None
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(out_path, mesh)
        return mesh
    except Exception as e:
        print('[poisson_sync] failed:', e)
        return None


def mesh_metrics(mesh):
    if mesh is None:
        return {'vertices': 0, 'triangles': 0, 'area_mean': np.inf, 'area_std': np.inf}
    v = len(mesh.vertices)
    t = len(mesh.triangles)
    # triangle areas
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if t == 0:
        return {'vertices': v, 'triangles': 0, 'area_mean': np.inf, 'area_std': np.inf}
    A = verts[tris[:, 0]]
    B = verts[tris[:, 1]]
    C = verts[tris[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    return {'vertices': v, 'triangles': t, 'area_mean': float(areas.mean()), 'area_std': float(areas.std())}


def render_mesh(mesh_path, img_path):
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        from open3d.visualization import rendering
        w, h = 1024, 768
        r = rendering.OffscreenRenderer(w, h)
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultLit'
        r.scene.add_geometry('mesh', mesh, mat)
        bounds = mesh.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        extent = bounds.get_extent()
        eye = center + [0, -max(extent) * 2.0, 0]
        r.scene.camera.look_at(center, eye, [0, 0, 1])
        img = r.render_to_image()
        o3d.io.write_image(img_path, img)
        return True
    except Exception as e:
        print('[render_mesh] failed:', e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--out-dir', type=str, default='outputs/autotune')
    parser.add_argument('--recon-depth', type=int, default=8)
    parser.add_argument('--recon-voxel', type=float, default=2.0)
    parser.add_argument('--register', choices=['None','frame_icp','global_icp'], default='frame_icp')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # small grid (adjust as needed)
    samples_grid = [600, 1200]
    percentiles = [98.0, 99.0]
    factors = [0.5, 0.6]
    icp_voxels = [1.0, 2.0]
    icp_thresholds = [8.0, 12.0]

    combos = list(itertools.product(samples_grid, percentiles, factors, icp_voxels, icp_thresholds))
    results = []
    start = time.time()
    for i, (samp, pct, fac, vox, thr) in enumerate(combos):
        print(f'[autotune] running combo {i+1}/{len(combos)} samp={samp} pct={pct} fac={fac} vox={vox} thr={thr}')
        pts = build_registered_pointcloud(args.video, args.max_frames, samp, pct, fac,
                                         register=(args.register.lower()), icp_voxel=vox, icp_threshold=thr,
                                         icp_max_iter=40, resize=args.resize)
        print('[autotune] collected points:', pts.shape)
        mesh_path = os.path.join(args.out_dir, f'combo_{i}_s{samp}_p{int(pct)}_f{int(fac*100)}_v{int(vox)}_t{int(thr)}.ply')
        mesh = poisson_sync(pts, mesh_path, voxel_size=args.recon_voxel, depth=args.recon_depth)
        metrics = mesh_metrics(mesh)
        print('[autotune] metrics:', metrics)
        results.append({'combo': (samp, pct, fac, vox, thr), 'mesh': mesh_path, 'metrics': metrics})

    # score combos
    verts = np.array([r['metrics']['vertices'] for r in results], dtype=float)
    stds = np.array([r['metrics']['area_std'] for r in results], dtype=float)
    # valid mask
    valid = (verts > 0) & np.isfinite(stds)
    if not valid.any():
        print('[autotune] no valid meshes produced')
        return
    vmin, vmax = verts[valid].min(), verts[valid].max()
    smin, smax = stds[valid].min(), stds[valid].max()
    scores = np.full(len(results), -np.inf)
    for idx, r in enumerate(results):
        if not valid[idx]:
            continue
        v = r['metrics']['vertices']
        s = r['metrics']['area_std']
        v_n = (v - vmin) / (vmax - vmin + 1e-9)
        s_n = (s - smin) / (smax - smin + 1e-9)
        score = 0.6 * v_n + 0.4 * (1.0 - s_n)
        scores[idx] = score

    best_idx = int(np.nanargmax(scores))
    best = results[best_idx]
    print('[autotune] best combo:', best['combo'], 'metrics:', best['metrics'], 'score:', scores[best_idx])

    best_mesh = best['mesh']
    best_img = os.path.join(args.out_dir, 'best_preview.png')
    if os.path.exists(best_mesh):
        ok = render_mesh(best_mesh, best_img)
        if ok:
            print('[autotune] best preview saved to', best_img)
        else:
            print('[autotune] failed to render best mesh')
    else:
        print('[autotune] best mesh file missing:', best_mesh)

    print('[autotune] done in', time.time() - start)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Simple TSDF fusion from freehand ultrasound frames (slice-accumulation + EDT).

This script:
 - streams frames (resized) from the input video
 - computes per-frame soft masks (adaptive percentile threshold + morphology)
 - builds a 3D binary volume mask (z,y,x)
 - computes signed distance via distance_transform_edt: sd = dist_out - dist_in
 - truncates and normalizes to [-1,1] as TSDF
 - runs marching_cubes(level=0) to extract mesh
 - saves mesh and preview PNG

Note: this is a prototype approach; for better results consider per-frame registration and per-pixel confidence weighting.
"""

import argparse
import os
import sys
import time
import numpy as np
from scipy.ndimage import median_filter, binary_closing, distance_transform_edt, label
from skimage import measure

# ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_ultrasound_realtime_imageData import threaded_frame_reader

try:
    import open3d as o3d
except Exception:
    o3d = None


def adaptive_mask(frame, percentile=98.0, factor=0.6, min_size=50, closing_iter=1):
    th = np.percentile(frame, percentile) * factor
    mask = frame > th
    mask = median_filter(mask.astype(np.uint8), size=3).astype(bool)
    if closing_iter > 0:
        for _ in range(closing_iter):
            mask = binary_closing(mask)
    if min_size > 0:
        # remove small objects
        lbl, n = label(mask)
        counts = np.bincount(lbl.ravel())
        keep = counts >= min_size
        keep[0] = False
        mask = keep[lbl]
    return mask


def volume_to_tsdf(mask3d, trunc_vox=8):
    # mask3d: (D,H,W) boolean where True indicates surface/occupied
    print('Computing distance transforms...')
    t0 = time.time()
    outside = distance_transform_edt(~mask3d)
    inside = distance_transform_edt(mask3d)
    sd = outside - inside
    print('Done EDT (%.2fs)' % (time.time() - t0))
    # truncate
    tsdf = np.clip(sd, -trunc_vox, trunc_vox) / float(trunc_vox)
    return tsdf, sd


def tsdf_to_mesh(tsdf, voxel_size=1.0, origin=(0,0,0), level=0.0):
    # tsdf is D,H,W -> we will run marching_cubes on (-tsdf) so that inside is negative? skimage expects volume
    print('Running marching_cubes on volume shape', tsdf.shape)
    verts, faces, normals, vals = measure.marching_cubes(tsdf, level=level, spacing=(1.0, 1.0, 1.0))
    # verts are in (z,y,x) order if volume is (D,H,W) with spacing; convert to x,y,z
    verts_xyz = verts[:, [2,1,0]] * voxel_size
    faces_idx = faces.astype(np.int32)
    return verts_xyz, faces_idx


def save_mesh_ply(verts, faces, out_path):
    if o3d is None:
        # fallback: save basic PLY
        try:
            with open(out_path, 'w') as f:
                f.write('ply\nformat ascii 1.0\n')
                f.write('element vertex %d\n' % len(verts))
                f.write('property float x\nproperty float y\nproperty float z\n')
                f.write('element face %d\n' % len(faces))
                f.write('property list uchar int vertex_indices\nend_header\n')
                for v in verts:
                    f.write('%f %f %f\n' % (v[0], v[1], v[2]))
                for face in faces:
                    f.write('3 %d %d %d\n' % (face[0], face[1], face[2]))
            return True
        except Exception as e:
            print('PLY save failed:', e)
            return False
    else:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(out_path, mesh)
        return True


def render_preview(mesh_path, img_path):
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        from open3d.visualization import rendering
        w,h = 1024, 768
        r = rendering.OffscreenRenderer(w,h)
        mat = rendering.MaterialRecord(); mat.shader='defaultLit'
        r.scene.add_geometry('mesh', mesh, mat)
        bounds = mesh.get_axis_aligned_bounding_box()
        center = bounds.get_center(); extent = bounds.get_extent()
        eye = center + [0, -max(extent)*2.0, 0]
        r.scene.camera.look_at(center, eye, [0,0,1])
        img = r.render_to_image()
        o3d.io.write_image(img_path, img)
        return True
    except Exception as e:
        print('Preview render failed:', e)
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--resize', type=int, default=128)
    p.add_argument('--max-frames', type=int, default=512)
    p.add_argument('--percentile', type=float, default=98.0)
    p.add_argument('--factor', type=float, default=0.5)
    p.add_argument('--min-size', type=int, default=100)
    p.add_argument('--closing-iter', type=int, default=1)
    p.add_argument('--trunc', type=int, default=8)
    p.add_argument('--voxel-size', type=float, default=1.0)
    p.add_argument('--out-dir', type=str, default='outputs/tsdf')
    p.add_argument('--headless', action='store_true')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    frames = []
    masks = []
    print('Streaming frames (max %s) ...' % (args.max_frames,))
    for idx, frame, landmarks in threaded_frame_reader(args.video, max_frames=args.max_frames, resize=args.resize, sr_model=None, device='cpu', threads=2):
        print('\rframe', idx, end='')
        mask = adaptive_mask(frame, percentile=args.percentile, factor=args.factor, min_size=args.min_size, closing_iter=args.closing_iter)
        frames.append(frame)
        masks.append(mask)
    print('\nCollected %d frames' % len(frames))
    if len(frames) == 0:
        print('No frames read')
        return

    D = len(frames)
    H, W = frames[0].shape
    print('Building mask volume (D,H,W)=', (D,H,W))
    mask3d = np.zeros((D, H, W), dtype=bool)
    for z in range(D):
        mask3d[z] = masks[z]

    tsdf, sd = volume_to_tsdf(mask3d, trunc_vox=args.trunc)

    # marching cubes
    verts, faces = tsdf_to_mesh(tsdf, voxel_size=args.voxel_size, origin=(0,0,0), level=0.0)
    print('Mesh vertices:', verts.shape, 'faces:', faces.shape)

    mesh_name = os.path.join(args.out_dir, 'tsdf_fused.ply')
    ok = save_mesh_ply(verts, faces, mesh_name)
    if ok:
        print('Saved mesh to', mesh_name)
    else:
        print('Failed to save mesh')

    img_path = os.path.join(args.out_dir, 'tsdf_fused.png')
    if o3d is not None:
        if render_preview(mesh_name, img_path):
            print('Saved preview to', img_path)
        else:
            print('Preview failed')
    print('Done')

if __name__ == '__main__':
    main()

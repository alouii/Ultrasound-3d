#!/usr/bin/env python3
"""
ultrasound_real_recon.py

Prototype: reconstruct a 3D point cloud from a folder of 2D ultrasound frames.

Features:
- Load frames from folder (PNG/JPG). Optionally DICOM if 'pydicom' installed.
- Depth model optional: if provided with --load-model (PyTorch state_dict), it will be used.
  If not provided, a robust intensity->pseudo-depth fallback is used.
- Dense optical-flow (Farneback) for inter-frame motion estimation -> approximate camera motion.
- Simple voxel-based accumulation for quick prototyping + Open3D visualization with
  auto-centering and sane defaults.

Usage:
    python ultrasound_real_recon.py --input-folder path/to/frames --max-frames 500
    python ultrasound_real_recon.py --input-folder path/to/frames --load-model depth.pt

Notes:
- This is a prototype for research/experimentation. For clinical use, replace depth network,
  pose estimation and fusion with production-grade components.
"""

import argparse
import os
import sys
import time
import math
import glob
import numpy as np
import cv2

# Try imports and provide actionable errors
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    print("PyTorch not found or failed to import. Install with `pip install torch torchvision`.")
    raise

try:
    import open3d as o3d
except Exception as e:
    print("Open3D not found. Install with `pip install open3d`.")
    raise

# Optional DICOM support
try:
    import pydicom
    HAVE_PYDICOM = True
except Exception:
    HAVE_PYDICOM = False

# ---------------------------
# Small UNet depth model (tiny)
# ---------------------------
class SmallUNet(nn.Module):
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        self.enc1 = nn.Conv2d(in_ch, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base*2, 3, padding=1)
        self.enc3 = nn.Conv2d(base*2, base*4, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec2 = nn.Conv2d(base*4, base*2, 3, padding=1)
        self.dec1 = nn.Conv2d(base*2, base, 3, padding=1)
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(self.pool(x1)))
        x3 = F.relu(self.enc3(self.pool(x2)))
        u2 = self.up2(x3)
        d2 = F.relu(self.dec2(torch.cat([u2, x2], dim=1)))
        u1 = self.up1(d2)
        d1 = F.relu(self.dec1(torch.cat([u1, x1], dim=1)))
        out = self.outc(d1)
        return F.softplus(out)


# ---------------------------
# Utilities
# ---------------------------
def list_image_files(folder, pattern="*"):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(sorted(glob.glob(os.path.join(folder, e))))
    # also optionally include pattern
    if pattern != "*":
        files = [p for p in files if pattern in os.path.basename(p)]
    return files

def read_frame(path, target_size=None):
    # handle DICOM if extension suggests or pydicom available
    ext = os.path.splitext(path)[1].lower()
    if ext in (".dcm",) and HAVE_PYDICOM:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        frame = (arr * 255.0).astype(np.uint8)
    else:
        frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f"Could not read image {path}")
    if target_size is not None:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    return frame

def pseudo_depth_from_intensity(gray):
    # heuristic: brighter intensity -> closer (smaller depth)
    # we map [0,255] -> [max_depth,min_depth]
    # use contrast stretching and a nonlinear curve to mimic ultrasound shading
    img = gray.astype(np.float32) / 255.0
    img = (img - img.mean()) * 0.8 + img.mean()  # mild contrast adjust
    img = np.clip(img, 0.0, 1.0)
    # invert intensity so bright->small depth
    depth = 0.25 + (1.0 - img) * 2.5  # between 0.25 and ~2.75 m
    return depth.astype(np.float32)

# ---------------------------
# Pose estimation: dense flow -> px translation -> small world translation heuristic
# ---------------------------
def estimate_pose_dense(prev_img, cur_img):
    # both grayscale uint8
    flow = cv2.calcOpticalFlowFarneback(prev_img, cur_img, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    u = flow[...,0]
    v = flow[...,1]
    # robust median translation
    med_u = float(np.median(u))
    med_v = float(np.median(v))
    # estimate small rotation by cross-correlation of gradients (very heuristic)
    # We'll keep rotation zero for safety and apply translation only.
    M = np.array([[1.0, 0.0, med_u],
                  [0.0, 1.0, med_v],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return M

def affine2pose_delta(M2d, fx, fy):
    # Convert 2D pixel translation -> small 3D translation in meters (heuristic)
    dx_px = float(M2d[0,2])
    dy_px = float(M2d[1,2])
    tx = dx_px / fx * 0.5  # scale factor (tune if needed)
    ty = dy_px / fy * 0.5
    # no rotation in 3D other than small z-rotation from 2x2 part
    R2 = M2d[:2,:2]
    theta = math.atan2(R2[1,0], R2[0,0])
    Rz = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                   [math.sin(theta),  math.cos(theta), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    delta = np.eye(4, dtype=np.float32)
    delta[:3,:3] = Rz
    delta[:3,3] = np.array([tx, ty, 0.0], dtype=np.float32)
    return delta

# ---------------------------
# Simple voxel accumulator (not a full TSDF) - quick and simple
# ---------------------------
class SimpleVoxelAccumulator:
    def __init__(self, voxel_size=0.01, grid_dim=256, origin=(-1.2,-1.2,0.0)):
        self.voxel_size = voxel_size
        self.grid_dim = grid_dim
        self.origin = np.array(origin, dtype=np.float32)
        self.sum_depth = np.zeros((grid_dim, grid_dim, grid_dim), dtype=np.float32)
        self.count = np.zeros((grid_dim, grid_dim, grid_dim), dtype=np.uint16)

    def world_to_voxel(self, xyz):
        rel = (xyz - self.origin[None,:]) / self.voxel_size
        return np.floor(rel).astype(int)

    def integrate_depth(self, depth_map, cam_pose, intrinsics):
        # depth_map: HxW in meters, cam_pose: 4x4 world-from-camera
        H,W = depth_map.shape
        fx, fy, cx, cy = intrinsics
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        zs = depth_map
        valid = zs > 0
        xs_cam = (xs[valid] - cx) * zs[valid] / fx
        ys_cam = (ys[valid] - cy) * zs[valid] / fy
        zs_cam = zs[valid]
        pts_cam = np.stack([xs_cam, ys_cam, zs_cam], axis=1)
        R = cam_pose[:3,:3]
        t = cam_pose[:3,3]
        pts_world = (R @ pts_cam.T).T + t[None,:]
        vox = self.world_to_voxel(pts_world)
        mask = np.all((vox >= 0) & (vox < self.grid_dim), axis=1)
        if not np.any(mask):
            return
        vox = vox[mask]
        depths = zs_cam[mask]
        vx = vox[:,0]; vy = vox[:,1]; vz = vox[:,2]
        self.sum_depth[vx,vy,vz] += depths
        self.count[vx,vy,vz] += 1

    def to_pointcloud(self, min_count=1):
        idxs = np.where(self.count >= min_count)
        if len(idxs[0]) == 0:
            return o3d.geometry.PointCloud()
        vx = np.array(idxs[0], dtype=np.float32)
        vy = np.array(idxs[1], dtype=np.float32)
        vz = np.array(idxs[2], dtype=np.float32)
        xyz = np.stack([vx, vy, vz], axis=1) * self.voxel_size + self.origin[None,:] + self.voxel_size*0.5
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz)
        return pc

# ---------------------------
# Visualization helpers
# ---------------------------
def create_visualizer(window_name="Ultra3D", width=1024, height=768, bg_color=(0,0,0)):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height)
    opt = vis.get_render_option()
    opt.background_color = np.asarray(bg_color, dtype=np.float32)
    opt.point_size = 2.0
    return vis

def center_view_on_pc(vis, pcd, zoom=0.7):
    # compute bbox center and set view control accordingly
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    # guard
    if np.linalg.norm(extent) < 1e-6:
        extent = np.array([1.0,1.0,1.0])
    ctr = vis.get_view_control()
    # choose front vector so camera points to -Z
    front = np.array([0.0, 0.0, -1.0])
    up = np.array([0.0, -1.0, 0.0])
    ctr.set_front(front.tolist())
    ctr.set_up(up.tolist())
    ctr.set_lookat(center.tolist())
    ctr.set_zoom(zoom)

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="3D reconstruction from 2D ultrasound frames (folder)")
    parser.add_argument("--input-folder", required=True, help="Folder containing image frames (PNG/JPG) sorted by name")
    parser.add_argument("--load-model", default=None, help="Path to PyTorch depth model (state_dict)")
    parser.add_argument("--width", type=int, default=256, help="resize width for processing")
    parser.add_argument("--height", type=int, default=256, help="resize height for processing")
    parser.add_argument("--fx", type=float, default=300.0, help="focal length in px (approx)")
    parser.add_argument("--max-frames", type=int, default=1000, help="max frames to process")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="voxel size in meters")
    parser.add_argument("--grid-dim", type=int, default=128, help="voxel grid dimension (each axis)")
    args = parser.parse_args()

    files = list_image_files(args.input_folder)
    if len(files) == 0:
        print("No image files found in", args.input_folder)
        sys.exit(1)
    print(f"Found {len(files)} image files - will process up to {args.max_frames}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # instantiate model if requested
    model = None
    if args.load_model:
        print("Loading depth model from:", args.load_model)
        model = SmallUNet(in_ch=1, base=16).to(device)
        try:
            state = torch.load(args.load_model, map_location=device)
            # handle if whole model was saved vs state_dict
            if isinstance(state, dict) and any(k.startswith("enc1") for k in state.keys()):
                model.load_state_dict(state)
            else:
                # try load entire object
                model.load_state_dict(state)
            model.eval()
            print("Model loaded and set to eval")
        except Exception as e:
            print("Failed to load model:", e)
            print("Continuing without model (will use pseudo-depth)")

    # intrinsics
    fx = args.fx
    fy = args.fx
    cx = args.width / 2.0
    cy = args.height / 2.0
    intrinsics = (fx, fy, cx, cy)

    # fuser
    origin = (-args.grid_dim*args.voxel_size/2.0, -args.grid_dim*args.voxel_size/2.0, 0.0)
    fuser = SimpleVoxelAccumulator(voxel_size=args.voxel_size, grid_dim=args.grid_dim, origin=origin)

    # open3d visualizer
    vis = create_visualizer(width=1000, height=700)
    pcd_geom = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_geom)

    prev_gray = None
    prev_pose = np.eye(4, dtype=np.float32)
    frame_idx = 0
    t0 = time.time()

    for i, path in enumerate(files):
        if frame_idx >= args.max_frames:
            break
        try:
            gray = read_frame(path, target_size=(args.width, args.height))
        except Exception as e:
            print("Failed to read", path, ":", e)
            continue

        if gray is None:
            continue

        # compute depth: either model or pseudo
        depth_map = None
        if model is not None:
            with torch.no_grad():
                inp = torch.from_numpy(gray.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(device)
                out = model(inp)[0,0].cpu().numpy()
                # normalize and map to meters
                outn = out - out.min()
                if outn.max() > 0:
                    outn = outn / outn.max()
                depth_map = 0.25 + outn * 2.75
        else:
            depth_map = pseudo_depth_from_intensity(gray)

        # get pose
        if prev_gray is None:
            pose2d = np.eye(3, dtype=np.float32)
        else:
            try:
                pose2d = estimate_pose_dense(prev_gray, gray)
            except Exception as e:
                print("Flow failed, falling back to identity:", e)
                pose2d = np.eye(3, dtype=np.float32)

        delta = affine2pose_delta(pose2d, fx, fy)
        cam_pose = prev_pose.copy() if prev_gray is not None else np.eye(4, dtype=np.float32)
        cam_pose = cam_pose @ delta  # apply delta in camera frame

        # integrate
        fuser.integrate_depth(depth_map, cam_pose, intrinsics)

        # update prev
        prev_gray = gray.copy()
        prev_pose = cam_pose.copy()
        frame_idx += 1

        # update visualization every N frames
        if frame_idx % 5 == 0 or frame_idx == 1:
            pc = fuser.to_pointcloud(min_count=1)
            if len(pc.points) > 0:
                pc.estimate_normals()
                pcd_geom.points = pc.points
                pcd_geom.normals = pc.normals
                vis.update_geometry(pcd_geom)
                # auto center view
                center_view_on_pc(vis, pcd_geom, zoom=0.7)
            else:
                # show a tiny test cube so window doesn't go blank
                cube = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
                try:
                    vis.clear_geometries()
                except Exception:
                    pass
                vis.add_geometry(cube)
                vis.poll_events()
                vis.update_renderer()
                # then re-add pointcloud object for subsequent updates
                vis.clear_geometries()
                vis.add_geometry(pcd_geom)

            vis.poll_events()
            vis.update_renderer()

        # lightweight progress print
        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")

    # finalize: show final cloud with draw_geometries (blocks)
    final_pc = fuser.to_pointcloud(min_count=1)
    if len(final_pc.points) == 0:
        print("No points were reconstructed. Possible causes:")
        print("- depth map values are invalid (all zeros or NaN)")
        print("- frames are all identical -> no parallax")
        print("- intrinsics/focal length are set wrongly")
        # show a debug cube so user sees something
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        o3d.visualization.draw_geometries([frame], window_name="Debug frame")
    else:
        print("Reconstruction complete. Points:", len(final_pc.points))
        center_view_on_pc(vis, final_pc, zoom=0.7)
        # wait a moment and then close the live window and open a final blocking viewer
        time.sleep(0.3)
        vis.destroy_window()
        o3d.visualization.draw_geometries([final_pc], window_name="Final reconstruction")

if __name__ == "__main__":
    main()

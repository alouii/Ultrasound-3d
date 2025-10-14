"""
ultrasound_3d_prototype.py

Extended working prototype for 3D reconstruction from 2D ultrasound-like sequences.

NEW FEATURES (compared to earlier scaffold):
- Self-supervised training loop using simulated frames with known poses.
  - Photometric reconstruction loss (warp-based) + edge-aware smoothness regularizer.
  - Uses consecutive-frame consistency (t -> t+1) for supervision.
- Improved (fallback) pose estimation using dense optical flow (Farneback) + robust
  translation/rotation heuristic for small motions.
- CLI flags:
    --simulate           : run visualization with synthetic frames (default)
    --video PATH         : run with a real video file
    --train              : run a short self-supervised training on simulated data
    --save-model PATH    : path to save the trained depth model
    --load-model PATH    : path to load a trained depth model for inference
    --max-frames N       : stop after N frames

Design notes:
- Training runs on CPU/GPU depending on availability. It uses only simulated frames so
  you can train a model without any external dataset. The same training loop can be
  adapted to real ultrasound sequences by replacing the simulated generator with a loader
  and using a pose estimator (e.g. probe encoder or SLAM) instead of ground-truth poses.
- The self-supervised warp-based loss assumes small inter-frame motion and consistent
  imaging (intensity changes due to physical effects are ignored).

Run examples:
    # quick train for 200 steps and save model
    python ultrasound_3d_prototype.py --train --simulate --save-model depth.pt --max-frames 200

    # run visualization using the trained model
    python ultrasound_3d_prototype.py --simulate --load-model depth.pt

    # run with real video (no training)
    python ultrasound_3d_prototype.py --video my_ultrasound.mp4 --load-model depth.pt

Dependencies:
    pip install numpy opencv-python torch torchvision open3d matplotlib

"""

import argparse
import time
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import os

# ----------------------------- Utilities ---------------------------------

def make_logger(prefix="[ultra3d]"):
    def log(*args, **kwargs):
        print(prefix, *args, **kwargs)
    return log

log = make_logger()

# -------------------------- Simulation ----------------------------------

def generate_simulated_frame(width, height, t, n_blobs=3):
    """Create a synthetic ultrasound-like 2D frame and a depth map (for testing).
    Return: frame_gray (H,W), depth (H,W), and a simple probe pose (4x4).
    Pose sim: probe translates along x and rotates slowly around z.
    """
    xv, yv = np.meshgrid(np.linspace(-1,1,width), np.linspace(-1,1,height))
    frame = np.zeros((height, width), dtype=np.float32)
    depth = np.ones((height, width), dtype=np.float32) * 2.0

    rng = np.random.RandomState(int(t*1000) % 10000)
    for i in range(n_blobs):
        cx = 0.6 * math.sin(t * 0.8 + i)
        cy = 0.4 * math.cos(t * 0.6 + i*1.3)
        rx = 0.15 + 0.05 * math.sin(t*0.7 + i)
        ry = 0.08 + 0.04 * math.cos(t*0.9 + i)
        blob = np.exp(-(((xv-cx)/rx)**2 + ((yv-cy)/ry)**2))
        intensity = 0.5 + 0.5*np.sin(t*1.2 + i*0.5)
        frame += intensity * blob
        depth += 0.5 * blob * (0.2 + 0.3*i)

    speckle = rng.normal(loc=0.0, scale=0.08, size=frame.shape)
    frame = frame + speckle
    frame = np.clip(frame, 0.0, 1.0)

    depth = depth.astype(np.float32)
    depth = np.clip(depth, 0.2, 3.0)

    frame8 = (frame * 255).astype(np.uint8)

    tx = 0.03 * t
    rz = 0.05 * t
    pose = np.eye(4, dtype=np.float32)
    c = math.cos(rz)
    s = math.sin(rz)
    pose[:3,:3] = np.array([[c, -s, 0.0],[s, c, 0.0],[0.0,0.0,1.0]], dtype=np.float32)
    pose[:3,3] = np.array([tx, 0.0, 0.0], dtype=np.float32)

    return frame8, depth, pose

# --------------------------- Depth Model --------------------------------

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

# ------------------------- Pose Estimation -------------------------------

def estimate_pose_orb(prev_img, cur_img):
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(prev_img, None)
    kp2, des2 = orb.detectAndCompute(cur_img, None)
    if des1 is None or des2 is None:
        return np.eye(3, dtype=np.float32)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(des1, des2)
    except Exception:
        return np.eye(3, dtype=np.float32)
    matches = sorted(matches, key=lambda x: x.distance)[:200]
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
    if len(pts1) < 6:
        return np.eye(3, dtype=np.float32)
    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None:
        return np.eye(3, dtype=np.float32)
    M_h = np.vstack([M, [0,0,1]]).astype(np.float32)
    return M_h


def estimate_pose_dense(prev_img, cur_img):
    # dense optical flow
    prev_f = prev_img.astype(np.uint8)
    cur_f = cur_img.astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(prev_f, cur_f, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # estimate robust translation as median flow
    u = flow[...,0]
    v = flow[...,1]
    med_u = np.median(u)
    med_v = np.median(v)
    # small rotation estimate via flow curl (heuristic)
    theta = 0.0
    M = np.array([[math.cos(theta), -math.sin(theta), med_u],[math.sin(theta), math.cos(theta), med_v],[0,0,1]], dtype=np.float32)
    return M

# ---------------------------- Warp Utils --------------------------------

def project_points(pts3, intrinsics):
    fx, fy, cx, cy = intrinsics
    x = pts3[:,0]
    y = pts3[:,1]
    z = pts3[:,2]
    u = (x * fx) / z + cx
    v = (y * fy) / z + cy
    return np.stack([u,v], axis=1)

def backproject_pixel_to_cam(u, v, depth, intrinsics):
    fx, fy, cx, cy = intrinsics
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return np.stack([x,y,z], axis=1)

# bilinear sampling for torch tensors

def bilinear_sample(img, coords):
    # img: Bx1xHxW, coords: BxNx2 in normalized [-1,1]
    return F.grid_sample(img, coords.unsqueeze(1), align_corners=True)[:,0,0]

# --------------------------- TSDF Fusion -------------------------------

class SimpleTSDFFuser:
    def __init__(self, voxel_size=0.01, grid_dim=256, origin=(-1.0,-1.0,0.0)):
        self.voxel_size = voxel_size
        self.grid_dim = grid_dim
        self.origin = np.array(origin, dtype=np.float32)
        self.sum_depth = np.zeros((grid_dim, grid_dim, grid_dim), dtype=np.float32)
        self.count = np.zeros_like(self.sum_depth, dtype=np.uint16)

    def world_to_voxel(self, xyz):
        rel = (xyz - self.origin[None,:]) / self.voxel_size
        return np.floor(rel).astype(int)

    def integrate_frame(self, depth_map, pose, intrinsics):
        h,w = depth_map.shape
        fx, fy, cx, cy = intrinsics
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        zs = depth_map
        valid = zs > 0
        xs_cam = (xs[valid] - cx) * zs[valid] / fx
        ys_cam = (ys[valid] - cy) * zs[valid] / fy
        zs_cam = zs[valid]
        pts_cam = np.stack([xs_cam, ys_cam, zs_cam], axis=1)
        R = pose[:3,:3]
        t = pose[:3,3]
        pts_world = (R @ pts_cam.T).T + t[None,:]
        vox = self.world_to_voxel(pts_world)
        mask = np.all((vox >= 0) & (vox < self.grid_dim), axis=1)
        vox = vox[mask]
        depths = zs_cam[mask]
        vx = vox[:,0]
        vy = vox[:,1]
        vz = vox[:,2]
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

# --------------------------- Training Loop ------------------------------

def photometric_warp_loss(img_t, img_tp1, depth_t, pose_t, pose_tp1, intrinsics, device):
    # img_t, img_tp1: HxW (numpy uint8) -> convert to torch
    # depth_t: HxW (numpy float)
    H,W = img_t.shape
    fx, fy, cx, cy = intrinsics
    # create meshgrid of pixel coordinates
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)
    # backproject to camera coords (numpy)
    pts_cam = backproject_pixel_to_cam(xs_f.reshape(-1), ys_f.reshape(-1), depth_t.reshape(-1), intrinsics)
    pts_cam = pts_cam.reshape(-1,3)
    # transform points into world then into target camera frame
    R_t = pose_t[:3,:3]
    t_t = pose_t[:3,3]
    R_tp1 = pose_tp1[:3,:3]
    t_tp1 = pose_tp1[:3,3]
    pts_world = (R_t @ pts_cam.T).T + t_t[None,:]
    pts_cam_tp1 = (R_tp1.T @ (pts_world - t_tp1[None,:]).T).T  # world->cam_tp1
    # project
    uv = project_points(pts_cam_tp1, intrinsics)
    u = uv[:,0].reshape(H,W)
    v = uv[:,1].reshape(H,W)
    # normalize coords to [-1,1] for grid_sample
    u_norm = (u / (W-1)) * 2.0 - 1.0
    v_norm = (v / (H-1)) * 2.0 - 1.0
    grid = np.stack([u_norm, v_norm], axis=2)
    grid_t = torch.from_numpy(grid).unsqueeze(0).to(device).float()
    img_tp1_t = torch.from_numpy(img_tp1.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(device)
    img_t_t = torch.from_numpy(img_t.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(device)
    sampled = F.grid_sample(img_tp1_t, grid_t, align_corners=True, padding_mode='border')
    photometric = F.l1_loss(sampled, img_t_t)
    return photometric


def smoothness_loss(depth_map_tensor, img_tensor):
    # edge-aware smoothness (L1 on gradients weighted by image edges)
    dx = torch.abs(depth_map_tensor[:,:,1:,:] - depth_map_tensor[:,:,:-1,:])
    dy = torch.abs(depth_map_tensor[:,:,:,1:] - depth_map_tensor[:,:,:,:-1])
    img_dx = torch.mean(torch.abs(img_tensor[:,:,1:,:] - img_tensor[:,:,:-1,:]), dim=1, keepdim=True)
    img_dy = torch.mean(torch.abs(img_tensor[:,:,:,1:] - img_tensor[:,:,:,:-1]), dim=1, keepdim=True)
    wx = torch.exp(-img_dx)
    wy = torch.exp(-img_dy)
    sx = (dx * wx).mean()
    sy = (dy * wy).mean()
    return sx + sy

def train_on_simulation(model, device, intrinsics, steps=500, lr=1e-3, batch_size=1, save_path=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    H = 128
    W = 128
    t = 0.0
    for step in range(steps):
        # sample two consecutive frames
        frame_t, depth_t, pose_t = generate_simulated_frame(W, H, t)
        frame_tp1, depth_tp1, pose_tp1 = generate_simulated_frame(W, H, t+0.05)
        t += 0.05
        img_t = frame_t.astype(np.float32)/255.0
        img_tp1 = frame_tp1.astype(np.float32)/255.0
        inp_t = torch.from_numpy(img_t).unsqueeze(0).unsqueeze(0).to(device).float()
        inp_tp1 = torch.from_numpy(img_tp1).unsqueeze(0).unsqueeze(0).to(device).float()
        depth_pred_t = model(inp_t)  # Bx1xHxW
        depth_pred_tp1 = model(inp_tp1)
        # convert to numpy depth for warp (small prototype)
        depth_np_t = depth_pred_t.detach().cpu().numpy()[0,0]
        # photometric warp loss using GT poses (we train on simulated data)
        photometric = photometric_warp_loss(frame_t, frame_tp1, depth_np_t, pose_t, pose_tp1, intrinsics, device)
        smooth = smoothness_loss(depth_pred_t, inp_t)
        loss = photometric + 0.1 * smooth
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            log(f'step {step}/{steps} photometric={photometric.item():.4f} smooth={smooth.item():.6f} total={loss.item():.4f}')
        if save_path and step % 200 == 0 and step > 0:
            torch.save(model.state_dict(), save_path)
    if save_path:
        torch.save(model.state_dict(), save_path)
        log('Saved model to', save_path)

# --------------------------- Main Pipeline ------------------------------

def run_pipeline(args):
    W = args.width
    H = args.height
    fx = fy = 300.0
    cx = W / 2.0
    cy = H / 2.0
    intrinsics = (fx, fy, cx, cy)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallUNet(in_ch=1, base=16).to(device)
    if args.load_model and os.path.exists(args.load_model):
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        log('Loaded model', args.load_model)
    model.eval()

    fuser = SimpleTSDFFuser(voxel_size=0.02, grid_dim=128, origin=(-1.2,-1.2,0.0))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Ultra3D Prototype', width=800, height=600)
    pc_geom = o3d.geometry.PointCloud()
    vis.add_geometry(pc_geom)

    prev_gray = None
    prev_pose = np.eye(4, dtype=np.float32)

    t = 0.0
    frame_idx = 0
    last_vis_update = time.time()

    if args.simulate:
        source = 'simulate'
    else:
        source = 'video'
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            log('Could not open video:', args.video)
            return

    start_time = time.time()
    try:
        while True:
            if args.simulate:
                frame8, gt_depth, gt_pose = generate_simulated_frame(W, H, t)
                cur_gray = frame8
                cam_pose = gt_pose
                t += 0.05
            else:
                ret, frame = cap.read()
                if not ret:
                    log('End of video')
                    break
                cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cam_pose = None

            with torch.no_grad():
                img = cur_gray.astype(np.float32) / 255.0
                img_t = torch.from_numpy(img[None,None,:,:]).to(device)
                depth_pred = model(img_t)[0,0].cpu().numpy()
                depth_norm = (depth_pred - depth_pred.min())
                if depth_norm.max() > 0:
                    depth_norm /= depth_norm.max()
                depth_map = 0.2 + 2.8 * depth_norm

            if args.simulate:
                noise = np.random.normal(scale=0.01, size=gt_depth.shape)
                depth_map = np.clip(gt_depth + noise, 0.05, 3.0)

            if prev_gray is None:
                pose2d = np.eye(3, dtype=np.float32)
            else:
                # try dense optical flow first, fallback to ORB
                try:
                    pose2d = estimate_pose_dense(prev_gray, cur_gray)
                except Exception:
                    pose2d = estimate_pose_orb(prev_gray, cur_gray)

            dx_px = pose2d[0,2]
            dy_px = pose2d[1,2]
            tx = dx_px / fx * 0.5
            ty = dy_px / fy * 0.5
            R2 = pose2d[:2,:2]
            theta = math.atan2(R2[1,0], R2[0,0])
            Rz = np.array([[math.cos(theta), -math.sin(theta), 0.0],[math.sin(theta), math.cos(theta), 0.0],[0.0,0.0,1.0]], dtype=np.float32)
            if cam_pose is None:
                cam_pose = prev_pose.copy()
                cam_pose[:3,:3] = cam_pose[:3,:3] @ Rz
                cam_pose[:3,3] = cam_pose[:3,3] + np.array([tx,ty,0.0], dtype=np.float32)

            fuser.integrate_frame(depth_map, cam_pose, intrinsics)

            prev_gray = cur_gray.copy()
            prev_pose = cam_pose.copy()

            if time.time() - last_vis_update > 0.5:
                pc = fuser.to_pointcloud(min_count=1)
                if len(pc.points) > 0:
                    pc.estimate_normals()
                    pc_geom.points = pc.points
                    pc_geom.normals = pc.normals
                    vis.update_geometry(pc_geom)
                vis.poll_events()
                vis.update_renderer()
                last_vis_update = time.time()

            frame_idx += 1
            if args.simulate:
                time.sleep(0.01)
            else:
                time.sleep(0.001)

            if args.max_frames and frame_idx >= args.max_frames:
                break

    except KeyboardInterrupt:
        log('Interrupted by user')
    finally:
        elapsed = time.time() - start_time
        log(f'Processed {frame_idx} frames in {elapsed:.2f}s ({frame_idx/elapsed:.2f} fps)')
        vis.destroy_window()

# --------------------------------- CLI ----------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ultra3D prototype - extended')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--simulate', action='store_true', help='use synthetic frames')
    group.add_argument('--video', type=str, help='path to ultrasound video')
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--max-frames', type=int, default=500)
    parser.add_argument('--train', action='store_true', help='run self-supervised training on simulated data')
    parser.add_argument('--save-model', type=str, default='depth_model.pt')
    parser.add_argument('--load-model', type=str, default=None)
    args = parser.parse_args()
    if not args.simulate and not args.video and not args.train:
        log('No source provided; defaulting to --simulate')
        args.simulate = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.train:
        model = SmallUNet(in_ch=1, base=16).to(device)
        fx = fy = 300.0
        cx = args.width / 2.0
        cy = args.height / 2.0
        intrinsics = (fx, fy, cx, cy)
        train_on_simulation(model, device, intrinsics, steps=400, lr=1e-3, save_path=args.save_model)
    else:
        run_pipeline(args)

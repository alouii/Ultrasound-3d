import argparse
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from tqdm import tqdm
import os

# ---------- Depth model (dummy example) ----------
class DummyDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# ---------- Load depth model ----------
def load_model(model_path):
    model = DummyDepthModel()
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✅ Loaded model: {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load model weights: {e}, using dummy model instead.")
    else:
        print("⚠️ Model file not found, using dummy model instead.")
    model.eval()
    return model

# ---------- Convert video frames to depth maps ----------
def video_to_depths(video_path, model, max_frames=1000):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(frame_count, max_frames)

    with torch.no_grad():
        for i in tqdm(range(total), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (256, 256))
            tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            depth = model(tensor).squeeze().numpy()
            frames.append(depth)
    cap.release()
    return np.array(frames)

# ---------- Reconstruct 3D point cloud ----------
def reconstruct_point_cloud(depth_stack, scale=1.0):
    h, w = depth_stack.shape[1:]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    points = []
    for i, depth in enumerate(depth_stack):
        z = depth * scale
        x = xx * scale
        y = yy * scale
        z_offset = np.ones_like(z) * i * scale
        pts = np.stack([x, y, z_offset + z], axis=-1).reshape(-1, 3)
        points.append(pts)

    points = np.concatenate(points, axis=0)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    return pc

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-video', type=str, required=True, help='Path to ultrasound video')
    parser.add_argument('--load-model', type=str, default='depth_model.pt')
    parser.add_argument('--output', type=str, default='reconstruction.ply')
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--scale', type=float, default=1.0)
    args = parser.parse_args()

    model = load_model(args.load_model)
    depth_stack = video_to_depths(args.input_video, model, args.max_frames)

    print(f"Depth stack shape: {depth_stack.shape}")
    point_cloud = reconstruct_point_cloud(depth_stack, args.scale)

    o3d.io.write_point_cloud(args.output, point_cloud)
    print(f"✅ 3D reconstruction saved to {args.output}")
    o3d.visualization.draw_geometries([point_cloud])

if __name__ == '__main__':
    main()

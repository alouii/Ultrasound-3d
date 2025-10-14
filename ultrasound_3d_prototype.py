import argparse
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter
from skimage import exposure
import torch

# Try to import RealESRGAN
try:
    from realesrgan import RealESRGAN
    has_esrgan = True
except ImportError:
    has_esrgan = False
    print("⚠️ RealESRGAN not installed. Super-resolution will be skipped.")

# ---------------------------
# Utility Functions
# ---------------------------

def preprocess_frame(frame, resize=256):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (resize, resize))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = gray.astype(np.float32)/255.0
    return gray

def super_resolve_frame(model, frame, device='cuda'):
    frame_tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        sr = model(frame_tensor)
    sr = sr.squeeze().cpu().numpy()
    sr = np.clip(sr, 0, 1)
    return sr

def load_video_frames(video_path, max_frames=100, resize=256, sr_model=None, device='cuda', threads=4):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(total_frames, max_frames)

    def process_frame(_):
        ret, frame = cap.read()
        if not ret:
            return None
        gray = preprocess_frame(frame, resize)
        if sr_model:
            gray = super_resolve_frame(sr_model, gray, device=device)
        return gray

    with ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(tqdm(executor.map(process_frame, range(total)), total=total, desc="Loading frames"))
    cap.release()
    return [f for f in results if f is not None]

def interpolate_missing_slices(volume, target_slices=100):
    orig_slices = volume.shape[2]
    x = np.arange(orig_slices)
    x_new = np.linspace(0, orig_slices-1, target_slices)
    volume_interp = np.zeros((volume.shape[0], volume.shape[1], target_slices), dtype=volume.dtype)
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            volume_interp[i,j,:] = np.interp(x_new, x, volume[i,j,:])
    return volume_interp

def volume_to_pointcloud(volume, voxel_size=0.005, percentile=80):
    threshold = np.percentile(volume, percentile)
    indices = np.argwhere(volume > threshold)
    colors = np.repeat(volume[indices[:,0], indices[:,1], indices[:,2]][:,None], 3, axis=1)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(indices * voxel_size)
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--voxel-size', type=float, default=0.005)
    parser.add_argument('--smooth-sigma', type=float, default=1.0)
    parser.add_argument('--percentile', type=float, default=80)
    parser.add_argument('--target-slices', type=int, default=150)
    parser.add_argument('--use-sr', action='store_true', help="Enable super-resolution")
    parser.add_argument('--sr-model-path', type=str, default='RealESRGAN_x4plus.pth')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sr_model = None

    # Load ESRGAN if requested and available
    if args.use_sr:
        if has_esrgan:
            sr_model = RealESRGAN(device, scale=4)
            sr_model.load_weights(args.sr_model_path)
            print(f"✅ ESRGAN loaded on {device}")
        else:
            print("⚠️ RealESRGAN not found. Skipping super-resolution.")

    # Step 1: Load frames
    print("🎥 Loading video frames...")
    frames = load_video_frames(args.video, max_frames=args.max_frames,
                               resize=args.resize, sr_model=sr_model, device=device)

    # Step 2: Stack frames into volume
    volume = np.stack(frames, axis=2)

    # Step 3: Interpolate missing slices
    print("🔄 Interpolating slices...")
    volume = interpolate_missing_slices(volume, target_slices=args.target_slices)

    # Step 4: 3D Gaussian smoothing
    print("🧱 Applying 3D Gaussian smoothing...")
    volume = gaussian_filter(volume, sigma=args.smooth_sigma)

    # Step 5: Convert to point cloud
    print("📦 Converting volume to point cloud...")
    pc = volume_to_pointcloud(volume, voxel_size=args.voxel_size, percentile=args.percentile)

    # Step 6: Interactive visualization
    print("🔍 Launching interactive 3D visualization...")
    o3d.visualization.draw_geometries([pc])

if __name__ == '__main__':
    main()

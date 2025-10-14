import cv2
import numpy as np
import open3d as o3d

# Settings for testing
video_path = "path/to/video.mp4"
resize = 128
max_frames = 20
voxel_size = 0.01

# Load frames
cap = cv2.VideoCapture(video_path)
frames = []
for _ in range(max_frames):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (resize, resize))
    gray = gray.astype(np.float32)/255.0
    frames.append(gray)
cap.release()

volume = np.stack(frames, axis=2)
volume = volume / np.max(volume)  # normalize

# Convert to simple point cloud for quick test
threshold = 0.2
indices = np.argwhere(volume > threshold)
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(indices * voxel_size)
pc.colors = o3d.utility.Vector3dVector(np.repeat(volume[indices[:,0],indices[:,1],indices[:,2]][:,None],3,axis=1))

o3d.visualization.draw_geometries([pc])

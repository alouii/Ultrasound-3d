import cv2
import numpy as np
import open3d as o3d
import argparse
from tqdm import tqdm

# added for adaptive thresholding and TSDF/MC reconstruction
from scipy.ndimage import distance_transform_edt, median_filter, binary_closing
from skimage import measure, morphology


def load_video_frames(video_path, resize=None, normalize=True):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Loading {total} frames from {video_path}...")

    for _ in tqdm(range(total), desc="Reading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize:
            gray = cv2.resize(gray, (resize, resize))
        gray = gray.astype(np.float32)
        if normalize:
            gray = gray / 255.0
        frames.append(gray)

    cap.release()
    return np.stack(frames, axis=-1)  # Shape: (H, W, N)


def preprocess_volume(volume, smoothing=True, preserve_intensity=False):
    # Optional smoothing across slices
    if smoothing:
        import scipy.ndimage as ndi

        volume = ndi.gaussian_filter(volume, sigma=(1, 1, 2))
    # If we are preserving original intensity range (e.g., uint8 0..255), do not clip to 0..1
    if not preserve_intensity:
        volume = np.clip(volume, 0, 1)
    return volume


def adaptive_mask_from_volume(volume, percentile=99.0, factor=0.6, 
                              min_size=5000, closing_iter=2):
    """Compute a cleaned binary mask from volume using an adaptive threshold.

    Steps:
    - threshold = percentile of intensity * factor
    - median filter and binary closing to reduce speckle
    - remove small connected components, keep largest
    """
    th = np.percentile(volume, percentile) * factor
    mask = volume > th
    # median to reduce speckle
    mask = median_filter(mask.astype(np.uint8), size=3).astype(bool)
    # binary closing to fill small holes
    struct = np.ones((3, 3, 3), dtype=bool)
    for _ in range(closing_iter):
        mask = binary_closing(mask, structure=struct)

    # remove small objects
    mask = morphology.remove_small_objects(mask, min_size=min_size)

    # ensure largest connected component
    labels = morphology.label(mask)
    if labels.max() == 0:
        return mask
    largest = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest


def sample_volume_trilinear(volume, coords):
    """Trilinear sample `volume` at fractional voxel coordinates.

    Args:
        volume: numpy array shape (D, H, W)
        coords: (N,3) array of fractional coordinates in (z, y, x) voxel index space
    Returns:
        values: (N,) interpolated intensities
    """
    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0:
        return np.array([], dtype=np.float32)
    z = coords[:, 0]
    y = coords[:, 1]
    x = coords[:, 2]

    z0 = np.floor(z).astype(int)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    z1 = z0 + 1
    y1 = y0 + 1
    x1 = x0 + 1

    z0 = np.clip(z0, 0, volume.shape[0] - 1)
    y0 = np.clip(y0, 0, volume.shape[1] - 1)
    x0 = np.clip(x0, 0, volume.shape[2] - 1)
    z1 = np.clip(z1, 0, volume.shape[0] - 1)
    y1 = np.clip(y1, 0, volume.shape[1] - 1)
    x1 = np.clip(x1, 0, volume.shape[2] - 1)

    xd = (x - x0).astype(np.float32)
    yd = (y - y0).astype(np.float32)
    zd = (z - z0).astype(np.float32)

    c000 = volume[z0, y0, x0]
    c001 = volume[z0, y0, x1]
    c010 = volume[z0, y1, x0]
    c011 = volume[z0, y1, x1]
    c100 = volume[z1, y0, x0]
    c101 = volume[z1, y0, x1]
    c110 = volume[z1, y1, x0]
    c111 = volume[z1, y1, x1]

    c00 = c000 * (1 - xd) + c001 * xd
    c01 = c010 * (1 - xd) + c011 * xd
    c10 = c100 * (1 - xd) + c101 * xd
    c11 = c110 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd
    return c.astype(np.float32)


def mask_to_mesh_tsdf(mask, save_path, voxel_scale=1.0, preserve_voxel_coords=False, volume=None, preserve_intensity=False, sample_method='nearest'):
    """Convert binary mask to mesh using signed-distance (TSDF) and marching cubes.

    If preserve_voxel_coords=True and `volume` is provided, vertex colors will be sampled
    from the `volume` at nearest-neighbor or trilinear locations so the mesh retains fidelity
    to the original frames.

    `sample_method` can be 'nearest' or 'trilinear'.

    Returns an Open3D TriangleMesh and saves it to save_path.
    """
    # signed distance: outside distance minus inside distance
    print("Computing signed distance transform...")
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    signed = outside - inside  # zero at boundary

    # marching cubes at level 0
    print("Running marching cubes on signed distance (level=0)...")
    verts, faces, normals, values = measure.marching_cubes(signed, level=0.0)

    if verts.size == 0 or faces.size == 0:
        raise RuntimeError("Marching cubes returned no geometry")

    verts = verts.astype(np.float32)

    if preserve_voxel_coords:
        # keep vertices in voxel coordinates (so vertex -> voxel mapping is preserved)
        verts_scaled = verts * voxel_scale
    else:
        # normalize and scale for visibility (legacy behavior)
        verts_scaled = verts.copy()
        verts_scaled -= verts_scaled.mean(axis=0, keepdims=True)
        verts_scaled /= (np.max(np.abs(verts_scaled)) + 1e-8)
        verts_scaled *= 100 * voxel_scale

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_scaled)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()

    # If we have the original volume and requested to preserve voxel coordinates,
    # sample per-vertex intensity and store as vertex colors. This helps keep
    # the reconstructed mesh visually faithful to the input video frames.
    if preserve_voxel_coords and (volume is not None):
        print(f"Sampling vertex intensities from volume to preserve fidelity (method={sample_method})...")
        # marching_cubes returns coordinates in voxel index space (z, y, x)
        if sample_method == 'nearest':
            idx = np.round(verts).astype(int)
            # clamp
            idx[:, 0] = np.clip(idx[:, 0], 0, volume.shape[0] - 1)
            idx[:, 1] = np.clip(idx[:, 1], 0, volume.shape[1] - 1)
            idx[:, 2] = np.clip(idx[:, 2], 0, volume.shape[2] - 1)
            intensities = volume[idx[:, 0], idx[:, 1], idx[:, 2]]
        elif sample_method == 'trilinear':
            # use trilinear interpolation at fractional vertex positions
            intensities = sample_volume_trilinear(volume, verts)
        else:
            raise ValueError(f"Unknown sample_method: {sample_method}")
        intensities = intensities.astype(np.float32)
        # if original intensities were in 0..255, scale to 0..1 for Open3D colors
        if preserve_intensity and intensities.max() > 1.0:
            colors = (intensities / 255.0).clip(0.0, 1.0)
        else:
            # assume already in 0..1
            colors = np.clip(intensities, 0.0, 1.0)
        colors = np.stack([colors, colors, colors], axis=-1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # save
    o3d.io.write_triangle_mesh(save_path, mesh)
    print(f"✅ TSDF mesh saved to {save_path}")
    return mesh


def visualize_volume(
    volume, threshold=0.5, voxel_size=1.0, save_path="ultrasound_mesh.ply", show=True, method='poisson'
):
    print("Thresholding and building 3D volume...")

    mask = volume > (np.max(volume) * threshold)
    coords = np.argwhere(mask)
    if coords.size == 0:
        print("⚠️ No voxels above threshold. Try lowering --threshold.")
        return

    # Normalize coordinates
    coords = coords.astype(np.float32)
    coords -= np.mean(coords, axis=0)
    coords /= np.max(np.abs(coords)) + 1e-5
    coords *= 100  # scale for visibility

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    intensities = volume[mask]
    colors = np.stack([intensities] * 3, axis=-1)
    colors = (colors - colors.min()) / (colors.max() + 1e-8)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Estimate normals
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
    )

    if method == 'poisson':
        # Surface reconstruction using Poisson
        print("Reconstructing 3D surface using Poisson...")
        try:
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=8
            )
        except Exception as e:
            print(f"⚠️ Poisson reconstruction failed: {e}")
            return

        # Clean up mesh
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
        mesh.compute_vertex_normals()

        # Save result
        o3d.io.write_triangle_mesh(save_path, mesh)
        print(f"✅ Mesh saved to {save_path}")

    elif method == 'tsdf':
        # Convert the intensity-based cloud to a binary mask and use TSDF+MC
        try:
            mask = adaptive_mask_from_volume(volume)
            mesh = mask_to_mesh_tsdf(mask, save_path)
        except Exception as e:
            print(f"⚠️ TSDF reconstruction failed: {e}")
            return
    else:
        print(f"Unknown reconstruction method: {method}")
        return

    # Visualize if requested
    if show:
        try:
            print("Displaying 3D mesh...")
            o3d.visualization.draw_geometries([mesh])
        except Exception as e:
            print(f"⚠️ Visualization failed (headless?): {e}")
            print("Saved mesh can be opened with Open3D/meshlab/Paraview.")


def main():
    parser = argparse.ArgumentParser(
        description="3D Ultrasound Reconstruction from 2D Frames"
    )
    parser.add_argument("--video", required=True, help="Path to ultrasound video file")
    parser.add_argument(
        "--resize", type=int, default=None, help="Resize frames to this dimension"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for 3D voxel activation"
    )
    parser.add_argument(
        "--voxel-size", type=float, default=1.0, help="Voxel size for visualization"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview input frames before reconstruction",
    )
    parser.add_argument(
        "--out-dir", type=str, default="outputs", help="Directory to save output mesh (PLY)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open an interactive viewer after mesh reconstruction (useful for headless/CI)",
    )
    # Fidelity-preserving options
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable Gaussian smoothing across slices (preserves per-frame detail)",
    )
    parser.add_argument(
        "--preserve-intensity",
        action="store_true",
        help="Keep original intensity range (do not normalize frames to 0..1)",
    )
    parser.add_argument(
        "--preserve-voxel-coords",
        action="store_true",
        help="Keep mesh vertices in voxel coordinates so vertex colors can map back to original frames",
    )
    parser.add_argument(
        "--sample-method",
        choices=["nearest", "trilinear"],
        default="nearest",
        help="Method to sample per-vertex intensities when preserving voxel coordinates",
    )
    parser.add_argument(
        "--keep-small",
        action="store_true",
        help="Keep small connected components when creating mask (do not remove them)",
    )
    parser.add_argument(
        "--method",
        choices=["poisson", "tsdf"],
        default="tsdf",
        help="Reconstruction method: poisson (Poisson on point cloud) or tsdf (binary mask -> TSDF -> marching cubes)",
    )
    parser.add_argument(
        "--mask-percentile",
        type=float,
        default=99.0,
        help="Percentile for adaptive thresholding when using tsdf",
    )
    parser.add_argument(
        "--mask-factor",
        type=float,
        default=0.6,
        help="Factor multiplier applied to the percentile threshold when making the binary mask (tsdf method)",
    )
    parser.add_argument(
        "--mask-min-size",
        type=int,
        default=2000,
        help="Minimum connected component size to keep (voxels)",
    )
    args = parser.parse_args()

    volume = load_video_frames(args.video, resize=args.resize, normalize=(not args.preserve_intensity))
    print(f"Volume shape: {volume.shape} (H x W x Frames)")

    if args.preview:
        import matplotlib.pyplot as plt

        mid = volume.shape[-1] // 2
        plt.imshow(volume[..., mid], cmap="gray")
        plt.title("Middle Frame")
        plt.show()

    volume = preprocess_volume(volume, smoothing=(not args.no_smoothing), preserve_intensity=args.preserve_intensity)

    from utils.io import output_path_for_video

    save_path = output_path_for_video(args.video, args.out_dir)

    # if requested, keep small connected components by setting min_size to 0
    if args.keep_small:
        args.mask_min_size = 0

    if args.method == 'tsdf':
        # pass mask params
        global adaptive_mask_from_volume
        mask = adaptive_mask_from_volume(
            volume,
            percentile=args.mask_percentile,
            factor=args.mask_factor,
            min_size=args.mask_min_size,
        )
        mesh = mask_to_mesh_tsdf(
            mask,
            save_path,
            voxel_scale=args.voxel_size,
            preserve_voxel_coords=args.preserve_voxel_coords,
            volume=volume,
            preserve_intensity=args.preserve_intensity,
        )
        if not args.no_display:
            try:
                o3d.visualization.draw_geometries([mesh])
            except Exception as e:
                print(f"Visualization failed (headless?): {e}")
    else:
        visualize_volume(
            volume,
            threshold=args.threshold,
            voxel_size=args.voxel_size,
            save_path=save_path,
            show=not args.no_display,
            method=args.method,
        )
    print(f"✅ Mesh saved to {save_path}")


if __name__ == "__main__":
    main()

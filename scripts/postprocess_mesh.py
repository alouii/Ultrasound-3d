#!/usr/bin/env python3
"""Post-process a mesh: smoothing, decimation, small-component removal, and preview rendering."""

import argparse
import os
import numpy as np
import open3d as o3d


def mesh_stats(mesh):
    return {'vertices': len(mesh.vertices), 'triangles': len(mesh.triangles)}


def largest_component(mesh):
    # Compute connected components of triangles via vertex adjacency (BFS)
    import collections
    tris = np.asarray(mesh.triangles)
    n = len(tris)
    v2t = {}
    for ti, tri in enumerate(tris):
        for v in tri:
            v2t.setdefault(int(v), []).append(ti)
    visited = np.zeros(n, dtype=bool)
    components = []
    for i in range(n):
        if visited[i]:
            continue
        q = [i]
        comp = []
        visited[i] = True
        while q:
            cur = q.pop()
            comp.append(cur)
            for v in tris[cur]:
                for nei in v2t.get(int(v), []):
                    if not visited[nei]:
                        visited[nei] = True
                        q.append(nei)
        components.append(comp)
    if not components:
        return mesh
    # find largest
    largest = max(components, key=len)
    remove_idx = [i for i in range(n) if i not in set(largest)]
    if remove_idx:
        mesh.remove_triangles_by_index(remove_idx)
        mesh.remove_unreferenced_vertices()
    return mesh


def postprocess(mesh, smoothing_iters=3, decimation_factor=0.2, min_triangles=1000):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    before = mesh_stats(mesh)

    # smoothing
    try:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smoothing_iters)
    except Exception as e:
        print('[post] Taubin smoothing failed:', e)
    # keep largest connected component
    mesh = largest_component(mesh)

    # decimation
    tri_count = len(mesh.triangles)
    target = max(min_triangles, int(max(min_triangles, tri_count * decimation_factor)))
    if target < tri_count:
        try:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
            mesh.remove_unreferenced_vertices()
        except Exception as e:
            print('[post] Decimation failed:', e)

    after = mesh_stats(mesh)
    return mesh, before, after


def render_mesh(mesh, img_path):
    try:
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
        print('[render] failed:', e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input mesh file')
    parser.add_argument('--out-dir', default='outputs', help='Output directory')
    parser.add_argument('--smoothing-iters', type=int, default=3)
    parser.add_argument('--decimation-factor', type=float, default=0.2)
    parser.add_argument('--min-triangles', type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(args.input)
    if mesh is None or len(mesh.triangles) == 0:
        print('Input mesh invalid or empty')
        return

    basename = os.path.splitext(os.path.basename(args.input))[0]
    out_mesh = os.path.join(args.out_dir, f"{basename}_cleaned.ply")
    out_img = os.path.join(args.out_dir, f"{basename}_cleaned.png")

    print('Before postprocessing:', mesh_stats(mesh))
    mesh, before, after = postprocess(mesh, smoothing_iters=args.smoothing_iters,
                                     decimation_factor=args.decimation_factor,
                                     min_triangles=args.min_triangles)
    print('Before:', before, 'After:', after)

    o3d.io.write_triangle_mesh(out_mesh, mesh)
    ok = render_mesh(mesh, out_img)
    if ok:
        print('Saved cleaned mesh and preview:', out_mesh, out_img)
    else:
        print('Saved cleaned mesh:', out_mesh)


if __name__ == '__main__':
    main()

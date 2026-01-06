#!/usr/bin/env python3
"""Interactive mesh viewer with fallbacks.

Usage:
  python scripts/view_mesh.py outputs/autotune/combo_19_s1200_p98_f50_v2_t12_cleaned.ply --viewer auto

Options for --viewer: auto, pyvista, open3d, html
- auto: prefer PyVista, then Open3D, then HTML
- html: exports a GLB and writes a simple Three.js HTML file you can open in any browser
"""

import argparse
import os
import sys
import webbrowser

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception:
    pv = None


def view_pyvista(mesh_path):
    mesh = pv.read(mesh_path)
    p = pv.Plotter()
    p.add_mesh(mesh, color='lightcoral', show_edges=False)
    p.add_axes()
    p.show()


def view_open3d(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh is None or len(mesh.triangles) == 0:
        print('Open3D: mesh invalid or empty')
        return
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], window_name='Mesh Viewer')


def export_html_glb(mesh_path, out_dir=None):
    try:
        import trimesh
    except Exception as e:
        print('Trimesh not available, cannot export HTML viewer:', e)
        return None

    mesh = trimesh.load(mesh_path, force='mesh')
    if mesh is None:
        print('trimesh failed to load mesh')
        return None

    out_dir = out_dir or os.path.dirname(mesh_path)
    base = os.path.splitext(os.path.basename(mesh_path))[0]
    glb_path = os.path.join(out_dir, base + '.glb')
    html_path = os.path.join(out_dir, base + '_viewer.html')

    # export GLB
    try:
        mesh.export(glb_path)
    except Exception as e:
        print('Failed to export GLB:', e)
        return None

    # write a simple three.js HTML that loads the GLB
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{base} viewer</title>
  <style>body {{ margin:0; }} canvas {{ width:100%; height:100%; }}</style>
</head>
<body>
<script src="https://cdn.jsdelivr.net/npm/three@0.154.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.154.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.154.0/examples/js/loaders/GLTFLoader.js"></script>
<script>
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({alpha:false});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
const controls = new THREE.OrbitControls(camera, renderer.domElement);

const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(1,1,1);
scene.add(light);
scene.add(new THREE.AmbientLight(0x888888));

const loader = new THREE.GLTFLoader();
loader.load('{os.path.basename(glb_path)}', function(gltf) {{
  const obj = gltf.scene || gltf.scene.children[0];
  scene.add(obj);
  // center camera on object
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3()).length();
  const center = box.getCenter(new THREE.Vector3());
  camera.position.copy(center.clone().add(new THREE.Vector3(size, size, size)));
  controls.target.copy(center);
  controls.update();
}}, undefined, function(e) {{ console.error(e); }});

function animate() {{
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>"""

    with open(html_path, 'w') as f:
        f.write(html)

    print('Exported GLB and HTML viewer to', out_dir)
    return html_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mesh', help='Mesh file to view')
    p.add_argument('--viewer', choices=['auto', 'pyvista', 'open3d', 'html'], default='auto')
    p.add_argument('--open', action='store_true', help='Open HTML viewer in browser when using --viewer html')
    args = p.parse_args()

    mesh = args.mesh
    if not os.path.exists(mesh):
        print('Mesh file not found:', mesh)
        sys.exit(1)

    viewer = args.viewer
    if viewer == 'auto':
        if pv is not None:
            viewer = 'pyvista'
        elif o3d is not None:
            viewer = 'open3d'
        else:
            viewer = 'html'

    if viewer == 'pyvista':
        if pv is None:
            print('PyVista not available, install pyvista and pyvistaqt or choose open3d/html')
            sys.exit(1)
        view_pyvista(mesh)
    elif viewer == 'open3d':
        if o3d is None:
            print('Open3D not available, install open3d or choose pyvista/html')
            sys.exit(1)
        view_open3d(mesh)
    elif viewer == 'html':
        html = export_html_glb(mesh, out_dir=os.path.dirname(mesh))
        if html is None:
            print('Failed to produce HTML viewer; ensure trimesh is installed')
            sys.exit(1)
        print('Open the file in a browser:', html)
        if args.open:
            # serve the directory with a simple server and open browser
            import http.server, socketserver, threading, webbrowser
            cwd = os.path.dirname(mesh)
            os.chdir(cwd)
            port = 8000
            handler = http.server.SimpleHTTPRequestHandler
            httpd = socketserver.TCPServer(('0.0.0.0', port), handler)
            print('Serving', cwd, 'on http://0.0.0.0:%d/' % port)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            webbrowser.open(f'http://127.0.0.1:{port}/{os.path.basename(html)}')
    else:
        print('Unknown viewer:', viewer)

if __name__ == '__main__':
    main()

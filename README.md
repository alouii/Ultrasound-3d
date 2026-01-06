# Ultrasound 3D 
How to run (copy/paste):

Quick training (CPU or GPU if available), saves model to depth_model.pt:
python ultrasound_3d_prototype.py --train --simulate --save-model depth_model.pt --max-frames 400

Visualize reconstruction using the trained model:
python ultrasound_3d_prototype.py --simulate --load-model depth_model.pt --max-frames 1000

Use a real video (grayscale or color; the code converts to grayscale):
python ultrasound_3d_prototype.py --video path/to/your_video.mp4 --load-model depth_model.pt --max-frames 1000

Notes & limitations (important):

The training loop currently uses simulated ground-truth poses so the photometric warp is valid — adapting it to real unlabeled video will require either:

a reliable pose estimator between frames (probe encoder, SLAM), or

adding a learnable pose network (monodepth-style) and training jointly (I can add that next).

The fuser is a simplified TSDF-like accumulator (averaging depth into voxels). It's intentionally simple to be easy to understand. For production you'll want a proper TSDF on GPU (Open3D, Voxblox, or custom CUDA).

Optical-flow → 3D pose conversion is heuristic here (assumes small motion, near-planar probe motion). It's suitable for prototyping but not clinical use.

## Development

- Install dev tools: `pip install ruff black mypy pytest pytest-cov`
- Run linters: `ruff check .` and `black --check .`
- Type-check: `mypy --ignore-missing-imports .`
- Run tests: `pytest -q`

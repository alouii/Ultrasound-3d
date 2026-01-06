# Ultrasound 3D

A small prototype project to build 3D ultrasound volumes from 2D video frames. It provides multiple scripts for quick experimentation, real-time visualization, and safe memory-limited workflows.

---

## Quick start

1. Install runtime requirements:

   ```bash
   pip install -r requirements.txt
   ```

2. Run a safe low-resolution reconstruction (example):

   ```bash
   python hybrid_ultrasound_3d_safe.py --video path/to/video.mp4 --resize 128
   ```

3. Run the interactive real-time UI (requires `pyvistaqt`):

   ```bash
   python ai_ultrasound_realtime_imageData.py --video path/to/video.mp4 --resize 256 --max-slices 200
   ```

4. Reconstruct a full-resolution mesh (may be slow / memory-heavy):

   ```bash
   python hybrid_ultrasound_3d_full.py --video path/to/video.mp4 --resize 256 --threshold 0.5
   ```

---

## Scripts overview

- `us.py` — simple temporal accumulation + PyVista renderer.
- `ai_ultrasound_realtime_imageData.py` — real-time streaming demo with optional Real-ESRGAN and PyVista GUI.
- `ultrasound_3d_prototype.py` — simulation / prototype utilities and SR option.
- `hybrid_ultrasound_3d_full.py` — point-cloud / Poisson reconstruction pipeline (heavy-duty).
- `hybrid_ultrasound_3d_safe.py` — memory-safe low-res pipeline for quick tests.
- `hybrid_ultrasound_3d_safe2.py` — realistic smoothing & interpolation pipeline (includes French-language comments).

---

## Development & tests 

- Install developer tools (linters, type-checker, test runner):

  ```bash
  pip install -r requirements-dev.txt
  ```

- Common commands (also in `Makefile`):

  - `make install` — install runtime deps
  - `make dev-install` — install runtime + dev deps
  - `make lint` — run ruff/black/mypy
  - `make format` — autoformat and apply fixes
  - `make test` — run pytest

---

## Headless / CI

- For headless/off-screen rendering set the environment variable:

  ```bash
  export PYVISTA_OFF_SCREEN=1
  make headless VIDEO=path/to/video.mp4
  ```

- Example script using Xvfb (useful in CI): `scripts/headless_example.sh`.

### Batch processing multiple videos

There is a small helper to process a directory of videos and save per-video outputs:

```bash
# process all .mp4 files in a directory and save PLY outputs to outputs/
python scripts/batch_process.py --input-dir dataset/videos --pattern "*.mp4" --script full --out-dir outputs --workers 4 --headless
```

This will run `hybrid_ultrasound_3d_full.py` on each video and save a mesh PLY per input to the `outputs/` directory. Use `--script safe` to run the lighter-weight safe pipeline instead.

- CI is configured (`.github/workflows/ci.yml`) to run linters and tests on push/PR.

---

## Notes & limitations 

- This is a research / prototype codebase — **not** suitable for clinical use.
- Pose estimation and photometric consistency are simulated; adapting to real unlabeled data requires a pose estimator or a learnable pose network.
- Mesh/voxel reconstruction implementations are intentionally simple; for production use optimized TSDF/Poisson pipelines or GPU-based approaches are recommended.

---


Contributions welcome — open an issue or PR with improvements or questions.

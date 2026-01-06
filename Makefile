PY=python
VIDEO=path/to/video.mp4

.PHONY: install dev-install lint format test run-us run-rt run-safe headless xvfb-run

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

dev-install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -r requirements-dev.txt

lint:
	ruff check .
	black --check .
	mypy --ignore-missing-imports .

format:
	black .
	ruff check . --fix

test:
	pytest -q

run-us:
	$(PY) us.py --video $(VIDEO) --frames 60

run-rt:
	$(PY) ai_ultrasound_realtime_imageData.py --video $(VIDEO) --resize 256 --max-slices 200

run-safe:
	$(PY) hybrid_ultrasound_3d_safe.py --video $(VIDEO) --resize 128

headless:
	# Run rendering scripts in headless/off-screen mode
	PYVISTA_OFF_SCREEN=1 $(PY) hybrid_ultrasound_3d_safe.py --video $(VIDEO) --resize 128

xvfb-run:
	# Example: make xvfb-run SCRIPT="python hybrid_ultrasound_3d_safe.py --video $(VIDEO)"
	xvfb-run -a -s "-screen 0 1280x720x24" $(SCRIPT)

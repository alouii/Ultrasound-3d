import numpy as np
from hybrid_ultrasound_3d_safe import optimize_2d_frame


def test_optimize_2d_frame_output():
    img = np.random.randint(0, 256, size=(200, 300, 3), dtype=np.uint8)
    out = optimize_2d_frame(img, crop_ratio=0.1, target_size=128)
    assert out.shape == (128, 128)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255

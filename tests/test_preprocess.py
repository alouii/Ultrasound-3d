import numpy as np
from ai_ultrasound_realtime_imageData import preprocess_frame


def test_preprocess_frame_shape_and_range():
    img = np.random.randint(0, 256, size=(120, 160, 3), dtype=np.uint8)
    out = preprocess_frame(img, resize=64)
    assert out.shape == (64, 64)
    assert out.dtype == np.float32 or out.dtype == np.float64
    assert np.nanmin(out) >= 0.0 and np.nanmax(out) <= 1.0

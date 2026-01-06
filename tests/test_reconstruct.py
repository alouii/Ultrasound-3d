import numpy as np
from us import reconstruct_volume


def test_reconstruct_volume_shape_and_values():
    h, w = 32, 32
    frames = [np.full((h, w), i * 50, dtype=np.uint8) for i in range(3)]
    vol = reconstruct_volume(frames, depth_scale=0.1, time_decay=0.8)
    assert vol.shape == (h, w, len(frames))
    assert np.min(vol) >= 0.0 and np.max(vol) <= 1.0

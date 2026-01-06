import numpy as np
from hybrid_ultrasound_3d_safe2 import interpolate_volume


def test_interpolate_volume_shape():
    vol = np.random.rand(16, 16, 8).astype(np.float32)
    vol_out = interpolate_volume(vol, factor=2, smooth_sigma=0.5)
    assert vol_out.shape[2] == vol.shape[2] * 2

import numpy as np
from hybrid_ultrasound_3d_full import sample_volume_trilinear


def make_linear_volume(shape=(4, 4, 4)):
    # volume value = 100*z + 10*y + x -> linear separable
    z = np.arange(shape[0])[:, None, None]
    y = np.arange(shape[1])[None, :, None]
    x = np.arange(shape[2])[None, None, :]
    return (100 * z + 10 * y + x).astype(np.float32)


def test_trilinear_center_point():
    vol = make_linear_volume((4, 4, 4))
    coord = np.array([[1.25, 2.5, 0.75]], dtype=np.float32)  # z,y,x

    # Manually compute expected trilinear interpolation
    z, y, x = coord[0]
    z0, y0, x0 = int(np.floor(z)), int(np.floor(y)), int(np.floor(x))
    z1, y1, x1 = z0 + 1, y0 + 1, x0 + 1
    zd, yd, xd = z - z0, y - y0, x - x0

    def val(zz, yy, xx):
        return 100 * zz + 10 * yy + xx

    c000 = val(z0, y0, x0)
    c001 = val(z0, y0, x1)
    c010 = val(z0, y1, x0)
    c011 = val(z0, y1, x1)
    c100 = val(z1, y0, x0)
    c101 = val(z1, y0, x1)
    c110 = val(z1, y1, x0)
    c111 = val(z1, y1, x1)

    c00 = c000 * (1 - xd) + c001 * xd
    c01 = c010 * (1 - xd) + c011 * xd
    c10 = c100 * (1 - xd) + c101 * xd
    c11 = c110 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd

    expected = c0 * (1 - zd) + c1 * zd

    sampled = sample_volume_trilinear(vol, coord)
    assert sampled.shape == (1,)
    assert np.allclose(sampled[0], expected, atol=1e-5)


def test_trilinear_edge_and_out_of_bounds_clamp():
    vol = make_linear_volume((3, 3, 3))
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.5, 2.5, 2.5],  # near outside, should clamp to max index
    ], dtype=np.float32)
    sampled = sample_volume_trilinear(vol, coords)

    # first is exact corner
    assert sampled[0] == 0.0

    # second should effectively interpolate with clamped corner values (max indices at 2)
    # for our linear formula, the max value is 100*2 + 10*2 + 2 = 222
    assert np.isclose(sampled[1], 222.0, atol=1e-5)

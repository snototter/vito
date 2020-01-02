import numpy as np
from ..cam_projections import dot


def test_dot():
    x = np.random.rand(3, 1)
    y = np.random.rand(3)
    d = x[0]*y[0] + x[1]*y[1] + x[2]*y[2]
    assert dot(x, y) == d
    assert dot(y, x) == d

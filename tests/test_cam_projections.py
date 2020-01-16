import numpy as np
import pytest
from vito.cam_projections import dot

# def dot(a,b):
#     """Dot product of Dx1 vectors.
#     :params a,b: np.array    
#     """
#     las = len(a.shape)
#     lbs = len(b.shape)
#     if las != lbs:
#         if las == 1:
#             a = a.reshape((a.shape[0], 1))
#         if lbs == 1:
#             b = b.reshape((b.shape[0], 1))
#         las = len(a.shape)
#         lbs = len(b.shape)

#     if las > 1 and lbs > 1:
#         assert a.shape == b.shape
#         assert a.shape[1] == 1, 'Currently we only support vector dot products'

#         sum = 0.0
#         for i in range(a.shape[0]):
#             sum += a[i] * b[i]
#         return sum
#     elif las == 1 or lbs == 1:
#         return np.dot(a, b)

def test_dot():
    x = np.random.rand(3, 1)
    y = np.random.rand(3)
    d = x[0]*y[0] + x[1]*y[1] + x[2]*y[2]
    assert dot(x, y) == d
    assert dot(y, x) == d

    with pytest.raises(ValueError):
        dot(np.zeros((10)), np.zeros((20)))
    with pytest.raises(ValueError):
        dot(np.zeros((10,)), np.zeros((20,)))

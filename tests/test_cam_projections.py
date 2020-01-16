import numpy as np
import pytest
from vito.cam_projections import dot, apply_transformation, \
    apply_projection, P_from_K_R_t, project_world_to_image_K_Rt, \
    project_world_to_image_K_R_t, project_world_to_image_K_R_C, \
    project_world_to_image_with_distortion_K_Rt, \
    get_groundplane_to_image_homography


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
    with pytest.raises(ValueError):
        dot(np.array([]), np.zeros((20,)))
    with pytest.raises(ValueError):
        dot(np.zeros((13,)), np.array([]))


def test_apply_transformation():
    T2d = np.random.rand(2, 2)
    T3d = np.random.rand(3, 2)

    x = apply_transformation(T2d, np.ones((2, 1)))
    assert x.shape[0] == 2 and x.shape[1] == 1
    assert np.all(np.sum(T2d, axis=1).reshape(x.shape) == x)

    x = apply_transformation(T3d, 2*np.ones((2, 1), dtype=np.uint8))
    assert x.shape[0] == 3 and x.shape[1] == 1
    assert x.dtype == np.float64
    assert np.all(np.sum(T3d, axis=1).reshape(x.shape) == pytest.approx(x/2))

    # 3x2 * 1x1
    x = apply_transformation(T3d, 2*np.ones((1, 1), dtype=np.uint8))
    assert x.shape[0] == 3 and x.shape[1] == 1
    assert x.dtype == np.float64
    expected = 2*T3d[:,0] + T3d[:,1]
    assert np.all(expected.reshape(x.shape) == pytest.approx(x))

    with pytest.raises(ValueError):
        # 3x2 * 3x1
        apply_transformation(T3d, np.ones((3, 1), dtype=np.uint8))
    with pytest.raises(ValueError):
        # 3x2 * 2x2
        apply_transformation(np.random.rand(3, 4), np.random.rand(1, 2))


def test_apply_projection():
    T2d = np.random.rand(2, 2)
    x = apply_projection(T2d, 2*np.ones((2, 3)))
    assert x.shape[0] == 1 and x.shape[1] == 3
    s = np.sum(T2d, axis=1)
    expected = s[0] / s[1]
    assert np.all(expected == pytest.approx(x))

    # Project the world origin to some point.
    # This works, because the homogeneous coordinate will be added.
    T = np.random.rand(3, 4)
    x = apply_projection(T, np.zeros((3, 1)))
    assert x.shape[0] == 2 and x.shape[1] == 1
    assert x[0] == pytest.approx(T[0, 3]/T[2, 3])
    assert x[1] == pytest.approx(T[1, 3]/T[2, 3])
    # 3D-2D projection can also be done via:
    y = project_world_to_image_K_Rt(np.eye(3, 3), T, np.zeros((3, 1)))
    assert np.all(y == x)
    z = project_world_to_image_K_R_t(np.eye(3, 3), T[:, :3], T[:, -1], np.zeros((3, 1)))
    assert np.all(z == y)
    no_distortion = project_world_to_image_with_distortion_K_Rt(
        np.eye(3, 3), T, np.zeros((5, 1)), np.zeros((3, 1)))
    assert np.all(no_distortion == z)

    H = get_groundplane_to_image_homography(T)
    assert H.shape[0] == 3 and H.shape[1] == 3
    assert np.all(H[:, 0] == T[:, 0])
    assert np.all(H[:, 1] == T[:, 1])
    assert np.all(H[:, 2] == T[:, 3])

    # Divide by zero (NumPy should take care of this)
    np.seterr(divide='ignore', invalid='ignore')
    x = apply_projection(np.random.rand(3, 4), np.zeros((4, 1)))
    assert np.all(np.isnan(x))


def test_P():
    I = np.eye(3, 3)
    R = np.random.rand(3, 3)
    t = np.random.rand(3, 1).reshape((3, ))
    P = P_from_K_R_t(I, R, t)
    assert P.shape[0] == 3 and P.shape[1] == 4
    assert np.all(P == np.column_stack((R, t)))
    K = np.random.rand(3, 3)
    P = P_from_K_R_t(K, R, t)
    assert P.shape[0] == 3 and P.shape[1] == 4
    Rt = np.column_stack((R, t))
    for r in range(P.shape[0]):
        for c in range(P.shape[1]):
            assert P[r, c] == pytest.approx(dot(K[r, :], Rt[:, c]))

# project_world_to_image_K_Rt(K, Rt, world_pts)
# project_world_to_image_K_R_t
# TODO project_world_to_image_K_R_C

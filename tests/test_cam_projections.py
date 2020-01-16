import numpy as np
import pytest
from vito.cam_projections import dot, apply_transformation, \
    apply_projection, P_from_K_R_t, project_world_to_image_K_Rt, \
    project_world_to_image_K_R_t, project_world_to_image_K_R_C, \
    project_world_to_image_with_distortion_K_Rt, \
    project_world_to_image_with_distortion_K_R_t, \
    project_world_to_image_with_distortion_K_R_C, \
    get_groundplane_to_image_homography, shift_points_along_viewing_rays, \
    apply_dehomogenization, C_from_Rt, C_from_R_t, \
    get_image_to_groundplane_homography, \
    normalize_image_coordinates, normalize_image_coordinates_with_distortion


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
    assert dot(np.float32(3), np.float64(7)) == 21
    assert dot(np.float32(3), np.float64([7])) == 21
    assert dot(np.float32([3]), np.float64([7])) == 21


def test_apply_transformation():
    T2d = np.random.rand(2, 2)
    T3d = np.random.rand(3, 2)

    with pytest.raises(ValueError):
        x = apply_transformation(T2d, np.int32(2))
    x = apply_transformation(T2d, np.int32([17]))
    assert x.shape[0] == 2 and x.shape[1] == 1
    expected = 17 * T2d[:, 0] + T2d[:, 1]
    assert np.all(x == expected.reshape(x.shape))

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
    expected = 2*T3d[:, 0] + T3d[:, 1]
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
    # Now, try difference project_world_to_image functions:
    pts = np.random.rand(3, 17)
    K = np.diag(np.random.rand(3)) * 92
    K[2, 2] = 1
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    t = np.random.rand(3, 1)
    Rt = np.column_stack((R, t))
    y = project_world_to_image_K_Rt(K, Rt, pts)
    z = project_world_to_image_K_R_t(K, R, t, pts)
    assert np.all(z == y)
    # Or even abusing the projection with(out) distortion
    no_distortion = project_world_to_image_with_distortion_K_Rt(
        K, Rt, np.zeros((5, 1)), pts)
    assert np.all(no_distortion == pytest.approx(z))
    # Again, above is the same as:
    no_distortion = project_world_to_image_with_distortion_K_R_t(
        K, R, t, np.zeros((5, 1)), pts)
    assert np.all(no_distortion == pytest.approx(z))
    # ... and also:
    no_distortion = project_world_to_image_with_distortion_K_R_C(
        K, R, C_from_R_t(R, t), np.zeros((5, 1)), pts)
    assert np.all(no_distortion == pytest.approx(z))
    # ... it doesn't matter whether we use C_from_R_t or C_from_Rt:
    no_distortion = project_world_to_image_with_distortion_K_R_C(
        K, R, C_from_Rt(Rt), np.zeros((5, 1)), pts)
    assert np.all(no_distortion == pytest.approx(z))
    # Finally, they're all the same as:
    tmp = project_world_to_image_K_R_C(K, R, C_from_Rt(Rt), pts)
    assert np.all(tmp == z)

    P = P_from_K_R_t(K, R, t)
    H = get_groundplane_to_image_homography(P)
    assert H.shape[0] == 3 and H.shape[1] == 3
    assert np.all(H[:, 0] == P[:, 0])
    assert np.all(H[:, 1] == P[:, 1])
    assert np.all(H[:, 2] == P[:, 3])

    Hinv = np.linalg.inv(H)
    H2 = get_image_to_groundplane_homography(P)
    assert np.all(Hinv == H2)

    # Divide by zero (NumPy should take care of this)
    np.seterr(divide='ignore', invalid='ignore')
    x = apply_projection(np.random.rand(3, 4), np.zeros((4, 1)))
    assert np.all(np.isnan(x))

    # Normalization of image coordinates
    K[0, 1] = 0.05
    K[0, 2] = 77
    K[1, 2] = 99
    img_pts = np.random.rand(2, 100)
    n_pts = normalize_image_coordinates(K, img_pts)
    Kinv = np.linalg.inv(K)
    assert np.all(apply_transformation(Kinv, img_pts) == n_pts)

    n_pts_wo = normalize_image_coordinates_with_distortion(
        K, np.zeros((5, )), img_pts)
    assert np.all(n_pts == pytest.approx(n_pts_wo))


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


def test_shift_points_along_viewing_rays():
    invalid_pts = [
        np.float32(3),
        np.float64([17]),
        np.ones((2, )),
        np.ones((4, 1)),
        np.ones((1, 3))
    ]
    for ip in invalid_pts:
        with pytest.raises(ValueError):
            shift_points_along_viewing_rays(ip, 10)

    pts = np.random.rand(3, 42) * 23
    pts_dh = apply_dehomogenization(pts)
    for d in (23.7, np.random.rand(1, pts.shape[1])):
        shifted = shift_points_along_viewing_rays(pts, d)
        expected_x = np.multiply(np.asarray(d), pts_dh[0, :])
        expected_y = np.multiply(np.asarray(d), pts_dh[1, :])
        assert np.all(expected_x == shifted[0, :])
        assert np.all(expected_y == shifted[1, :])
        assert np.all(shifted[2, :] == d)

    with pytest.raises(ValueError):
        shift_points_along_viewing_rays(pts, np.random.rand(1, pts.shape[1] - 1))
    with pytest.raises(ValueError):
        shift_points_along_viewing_rays(pts, np.random.rand(1, pts.shape[1] + 1))
    with pytest.raises(ValueError):
        shift_points_along_viewing_rays(pts, np.random.rand(2, 3))
#TODO test projections with distortions (works in practice, but unit tests are only for D=[0])
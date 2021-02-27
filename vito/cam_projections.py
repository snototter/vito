#!/usr/bin/env python
# coding=utf-8
"""Geometry utilities related to projections, camera calibration, etc."""

import numpy as np
import numbers

from . import pyutils as pu

# Check, if we can use np.matmul()
__has_np_matmul = pu.compare_version_strings(np.version.version, '1.10.0') >= 0

"""Util to use the correct matrix multiplication across different NumPy versions."""
matmul = np.matmul if __has_np_matmul else np.dot


def dot(a, b):
    """Dot product of Dx1 vectors.
    :params a,b: np.array
    """
    las = len(a.shape)
    lbs = len(b.shape)
    if las != lbs:
        if las == 1:
            a = a.reshape((a.shape[0], 1))
        if lbs == 1:
            b = b.reshape((b.shape[0], 1))
        las = len(a.shape)
        lbs = len(b.shape)

    if las > 1 and lbs > 1:
        assert a.shape == b.shape
        assert a.shape[1] == 1, 'Currently we only support vector dot products'

        sum = 0.0
        for i in range(a.shape[0]):
            sum += a[i] * b[i]
        return sum
    else:
        return np.dot(a, b)


def apply_transformation(T, pts):
    """Returns T*coords.
    :param T: MxD transformation matrix
    :param pts: DxN or (D-1)xN data points to transform. If dimension is D-1,
        a homogeneous dimension will be added.
    """
    if pts.ndim == 0:
        raise ValueError('Input coordinates must be DxN or (D-1)xN, not scalar.')
    T = T.astype(np.float64)
    pts = pts.astype(np.float64)
    ndim = T.shape[1]
    # Add homogeneous coordinate if necessary
    if pts.shape[0] == ndim-1:
        if pts.ndim == 1:
            npts = 1
        else:
            npts = pts.shape[1]
        pts = np.row_stack((pts, np.ones((1, npts), np.float64)))

    # Dimension check
    if pts.shape[0] != ndim:
        raise ValueError('Dimensions do not match')

    return matmul(T, pts)


# Alias
transform = apply_transformation


def apply_projection(P, pts):
    """Computes P*pts and returns the result after dehomogenization.
    :param P: MxD projection matrix
    :param pts: DxN or (D-1)xN data points to project. If dimension is D-1, a
        homogeneous dimension will be added.
    :returns: (M-1)xN projected points after dehomogenization.
    """
    return apply_dehomogenization(apply_transformation(P, pts))


def apply_dehomogenization(pts):
    """Computes the dehomogeneous points, i.e. dividing by the last dimension.
    :param pts: DxN points
    :returns: (D-1)xN dehomogeneous points

    This will issue a RuntimeWarning if the last dimension is 0. The
    corresponding division result will be set to nan.
    If you're aware of this, you can suppress the warnings, see:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.seterr.html
    """
    return np.divide(pts[:-1, :], pts[-1, :])


def shift_points_along_viewing_rays(pts, distance):
    """Shifts points along the viewing rays in a pinhole camera model. Usually
    used with 3xN pts: x and y coordinates will be computed, such that
    z=distance; distance can be a scalar or a 1xN array
    """
    if pts.ndim != 2 or pts.shape[0] != 3:
        raise ValueError('Input coordinates must be of shape 3xN.')
    pts = pts.astype(np.float64)
    num_pts = pts.shape[1]
    # Leverage similar triangles
    if isinstance(distance, numbers.Number):
        shifted = np.row_stack((pts[:-1, :] / pts[-1, :] * distance,
            np.array([distance]*num_pts, dtype=np.float64)))
    else:
        if distance.shape[1] != num_pts:
            raise ValueError('Number of pts and distances must match: {} points, but {} distances!'.format(
                num_pts, distance.shape[1]))
        distance = distance.astype(np.float64)
        shifted = np.row_stack((np.multiply(pts[:-1, :] / pts[-1, :], distance),
            distance))
    return shifted


def P_from_K_R_t(K, R, t):
    """Returns the 3x4 projection matrix P = K [R | t]."""
    K = K.astype(np.float64)
    R = R.astype(np.float64)
    t = t.astype(np.float64)
    return matmul(K, np.column_stack((R, t)))


def t_from_R_C(R, C):
    """Returns the 3x1 translation vector t = -RC."""
    return -matmul(R, C)


def C_from_R_t(R, t):
    """Returns the origin of the coordinate system, C = -R't."""
    return -matmul(np.transpose(R), t)


def C_from_Rt(Rt):
    """Returns the origin of the coordinate system, C = -R't."""
    return C_from_R_t(Rt[:, :3], Rt[:, 3])


def Rt_from_R_C(R, C):
    """Returns the 3x4 extrinsics [R | t], where t = -RC."""
    return np.column_stack((R, t_from_R_C(R, C)))


def project_world_to_image_K_Rt(K, Rt, world_pts):
    """Projects 3D world points onto the image plane, assuming a calibrated
    camera and rectified inputs.
    """
    K = K.astype(np.float64)
    Rt = Rt.astype(np.float64)
    P = matmul(K, Rt)
    return apply_projection(P, world_pts)


def project_world_to_image_K_R_t(K, R, t, world_pts):
    """Project 3D world points onto the image plane, assuming you already
    calibrated your cameras and rectified your images.

    Don't confuse C (3D position vector of the camera's optical center) and
    t = -RC !!! If you have C, you'll want to use the other
    project_world_to_image().
    """
    R = R.astype(np.float64)
    t = t.astype(np.float64)
    Rt = np.column_stack((R, t))
    return project_world_to_image_K_Rt(K, Rt, world_pts)


def project_world_to_image_K_R_C(K, R, C, world_pts):
    """Project 3D world points onto the image plane, assuming you already
    calibrated your cameras and rectified your images.

    Don't confuse C (3D position vector of the camera's optical center) and
    t = -RC !!! If you have t, you'll want to use
    project_world_to_image_K_R_t().
    """
    R = R.astype(np.float64)
    C = C.astype(np.float64)
    Rt = Rt_from_R_C(R, C)
    return project_world_to_image_K_Rt(K, Rt, world_pts)


def project_world_to_image_with_distortion_K_Rt(K, Rt, dist_coeff, coords):
    """Project 3D world points onto the image plane, compensating for the lens
    distortion.
    Note that Rt = [R, -RC], where C is the position vector of the optical
    center w.r.t. the world/reference coordinate system! Thus, t=-RC.

    See http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
    for notes on the camera model, how the distortion works, etc.
    """
    K = K.astype(np.float64)
    Rt = Rt.astype(np.float64)
    dist_coeff = dist_coeff.astype(np.float64)
    coords = coords.astype(np.float64)

    # Transform points from the world/reference coordinate system to the camera
    # coordinate system:
    coords_cam = apply_transformation(Rt, coords)

    # Perform the "normalized pinhole projection", i.e. shift the points along
    # the viewing rays to the "image plane". This is done via similar triangles:
    coords_normalized = coords_cam[0:2, :] / coords_cam[2, :]

    # Apply distortion model, since we use imperfect optics:
    kappa1, kappa2, rho1, rho2, kappa3 = dist_coeff
    r2 = np.sum(np.power(coords_normalized, 2), axis=0)  # squared radius, r^2 = x^2 + y^2
    r4 = np.multiply(r2, r2)
    r6 = np.multiply(r2, r4)

    rd = kappa1 * r2 + kappa2 * r4 + kappa3 * r6
    radial_distortion = np.multiply(coords_normalized, np.row_stack((rd, rd)))
    xy = np.multiply(coords_normalized[0, :], coords_normalized[1, :])
    x2 = np.power(coords_normalized[0, :], 2)
    y2 = np.power(coords_normalized[1, :], 2)
    tdx = 2.0 * rho1 * xy + rho2 * (r2 + 2.0 * x2)
    tdy = rho1 * (r2 + 2.0 * y2) + 2.0 * rho2 * xy
    tangential_distortion = np.row_stack((tdx, tdy))
    coords_corrected = coords_normalized + radial_distortion + tangential_distortion

    # Finally, project onto image:
    return apply_projection(K, coords_corrected)


# Alias for general projection
project = apply_projection


def project_world_to_image_with_distortion_K_R_C(K, R, C, dist_coeff, coords):
    """Project 3D world points onto the image plane, compensating for the lens
    distortion. See project_world_to_image_with_distortion_K_Rt()
    """
    R = R.astype(np.float64)
    C = C.astype(np.float64)
    Rt = np.column_stack((R, -matmul(R, C)))
    return project_world_to_image_with_distortion_K_Rt(K, Rt, dist_coeff, coords)


def project_world_to_image_with_distortion_K_R_t(K, R, t, dist_coeff, coords):
    """Project 3D world points onto the image plane, compensating for the lens
    distortion. See project_world_to_image_with_distortion_K_Rt()
    """
    R = R.astype(np.float64)
    t = t.astype(np.float64)
    Rt = np.column_stack((R, t))
    return project_world_to_image_with_distortion_K_Rt(K, Rt, dist_coeff, coords)


def normalize_image_coordinates(K, coords):
    """Compute the normalized coordinates of the given pixel coordinates."""
    invK = np.linalg.inv(K.astype(np.float64))
    return apply_transformation(invK, coords)


def normalize_image_coordinates_with_distortion(K, dist_coeff, pixel_coords):
    """Computes the normalized pinhole projection coordinates of the given pixel
    coordinates and compensates for the lens distortion.
    Basically, it computes:
      x_dist = K^-1 * pixel_coords
      x_norm = compensate_distortion(x_dist)
    """
    # There's no closed form (or even nice) algebraic inverse for the distorted
    # projection. This code is ported from Bouguet's MATLAB toolbox normalize.m
    # We also use his variable names and naming conventions
    pixel_coords = pixel_coords.astype(np.float64)
    K = K.astype(np.float64)
    # In Bouguet's toolbox, K[0,:] = [f_x, alpha_c * f_x, cc_x],
    # alpha_c is the skew coefficient
    alpha_c = K[0, 1] / K[0, 0]
    fc = (K[0, 0], K[1, 1])
    cc = (K[0, 2], K[1, 2])
    dist_coeff = dist_coeff.astype(np.float64)

    # Subtract principal point, and divide by the focal length:
    xdx = (pixel_coords[0, :] - cc[0]) / fc[0]
    xdy = (pixel_coords[1, :] - cc[1]) / fc[1]

    # Second, undo skew
    xdx = xdx - alpha_c * xdy
    x_distort = np.row_stack((xdx, xdy))

    # Third, compensate for lens distortion:
    xn = compute_distortion_oulu(x_distort, dist_coeff)
    npts = xn.shape[1]
    return np.row_stack((xn, np.ones((1, npts), dtype=np.float64)))


def compute_distortion_oulu(xd, k):  # pragma: no cover
    """Compensates for radial and tangential distortion. Model From Oulu
    university. This code is a Python port from Bouguet's toolbox, namely
    comp_distortion_oulu.m. We use the same/similar variable names."""
    k1 = k[0]
    k2 = k[1]
    k3 = k[4]
    p1 = k[2]
    p2 = k[3]

    # Initial guess
    x = xd
    num_pts = xd.shape[1]
    iteration = 0
    while iteration < 20:
        r_2 = np.sum(np.power(x, 2), axis=0)
        # r_2 = sum(x.^2);
        k_radial = np.reshape(1.0 + k1 * r_2 + k2 * np.power(r_2, 2) + k3 * np.power(r_2, 3), (1, num_pts))
        _xy = np.multiply(x[0, :], x[1, :])
        dxx = 2.0 * p1 * _xy + p2 * (r_2 + 2.0 * np.power(x[0, :], 2))
        dxy = p1 * (r_2 + 2.0 * np.power(x[1, :], 2)) + 2.0 * p2 * _xy
        delta_x = np.row_stack((dxx, dxy))

        # Compute relative change (to enable early termination)
        magnitude_delta2 = np.sum(np.power(delta_x, 2), axis=0)
        magnitude_x2 = np.sum(np.power(x, 2), axis=0)
        change = np.divide(magnitude_delta2, magnitude_x2)

        x = np.divide(xd - delta_x, np.row_stack((k_radial, k_radial)))
        # Stop early if there is no major improvement:
        if np.max(change) < 1e-5:
            break
        iteration += 1
    return x


def get_groundplane_to_image_homography(P):
    """Given the 3x4 camera projection matrix P, returns the homography
    mapping ground plane points onto the image plane."""
    P = P.astype(np.float64)
    return P[:, [0, 1, 3]]


def get_image_to_groundplane_homography(P):
    """Given the 3x4 camera projection matrix P, returns the homography
    mapping image plane points onto the ground plane."""
    return np.linalg.inv(get_groundplane_to_image_homography(P))


def rotx3d(theta):
    """3D rotation matrix, x-axis, angle in radians."""
    ct = np.cos(theta)
    st = np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, ct, -st],
        [0.0, st, ct]], dtype=np.float64)


def roty3d(theta):
    """3D rotation matrix, y-axis, angle in radians."""
    ct = np.cos(theta)
    st = np.sin(theta)
    return np.array([
        [ct, 0.0, st],
        [0.0, 1.0, 0.0],
        [-st, 0.0, ct]], dtype=np.float64)


def rotz3d(theta):
    """3D rotation matrix, z-axis, angle in radians."""
    ct = np.cos(theta)
    st = np.sin(theta)
    return np.array([
        [ct, -st, 0.0],
        [st, ct, 0.0],
        [0.0, 0.0, 1.0]], dtype=np.float64)


def rot3d(deg_x, deg_y, deg_z):
    """Returns the 3D rotation matrix in ZYX (i.e. yaw-pitch-roll) order."""
    Rx = rotx3d(np.deg2rad(deg_x))
    Ry = roty3d(np.deg2rad(deg_y))
    Rz = rotz3d(np.deg2rad(deg_z))
    R = matmul(Rx, matmul(Ry, Rz))
    return R


def compare_rotation_matrices(R1, R2):
    """Returns the rotation angle (radians) between two 3x3 rotation matrices."""
    # Compute rotation matrix R_1-->2 as R1' * R2
    r_12 = matmul(np.transpose(R1), R2)
    # Compute the axis-angle representation using
    # trace(R_12) = 1 + 2*cos(theta) and return the
    # angle as rotation error/deviation.
    # max(-1, min(1, ...) prevents nan values due to
    # floating point precision.
    return np.arccos(max(-1, min(1, (np.trace(r_12) - 1.0) / 2.0)))

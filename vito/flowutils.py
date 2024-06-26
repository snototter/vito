#!/usr/bin/env python
# coding=utf-8
"""Optical flow I/O and visualization."""

import sys
import numpy as np
from pathlib import Path
from typing import Union
from . import colormaps


def floread(filename: Union[str, Path]) -> np.ndarray:
    """
    Read optical flow (.flo) files stored in Middlebury format.

    Adapted from https://stackoverflow.com/a/28016469/400948

    Args:
      filename: Path to the input .flo file.

    Returns:
      HxWx2 numpy array of type `numpy.float32`.
    """
    if sys.byteorder != 'little':
        raise RuntimeError('Current .flo support requires little-endian architecture!')  # pragma: no cover

    if filename is None:
        return None

    if not Path(filename).exists():
        raise FileNotFoundError('File %s does not exist' % filename)

    if not Path(filename).is_file():
        raise FileNotFoundError('Path %s is not a file' % filename)

    with open(filename, 'rb') as f:
        # Check magic number
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic[0]:
            raise ValueError(
                'Magic number should be 202021.25, but got %f in file "%s"' % (magic[0], filename))
        # Next, get the image dimensions
        w = int(np.fromfile(f, np.int32, count=1))
        h = int(np.fromfile(f, np.int32, count=1))
        # Load the data and reshape
        flow = np.fromfile(f, np.float32, count=2*w*h)
        return np.resize(flow, (h, w, 2))


def flosave(filename: Union[str, Path], flow: np.ndarray) -> None:
    """
    Save HxWx2 optical flow to file in Middlebury format.

    Args:
      filename: Output file path.
      flow: Optical flow components as HxWx2 numpy array of
        type `numpy.float32`
    """
    if len(flow.shape) != 3 or flow.shape[2] != 2:
        raise ValueError('Invalid flow shape!')
    # Prepare data
    h, w = flow.shape[0], flow.shape[1]
    data = np.zeros((h, w*2))
    data[:, np.arange(w)*2] = flow[:, :, 0]
    data[:, np.arange(w)*2 + 1] = flow[:, :, 1]

    with open(filename, 'wb') as f:
        # Write magic number
        np.array([202021.25], np.float32).tofile(f)
        # Write dimensions as W,H (!)
        np.array(w).astype(np.int32).tofile(f)
        np.array(h).astype(np.int32).tofile(f)
        # Write actual data
        data.astype(np.float32).tofile(f)


def colorize_uv(
        u: np.ndarray,
        v: np.ndarray,
        return_rgb: bool = True,
        colorwheel: np.ndarray = colormaps.make_flow_color_wheel()
        ) -> np.ndarray:
    """
    Colorizes the given optical flow using the flow color wheel.
    This function performs no normalization or sanity checks. Usually, you
    should prefer colorize_flow() instead!

    This is a port of the C++/MATLAB code from
    https://people.csail.mit.edu/celiu/OpticalFlow, also based on
    https://github.com/tomrunia/OpticalFlow_Visualization.

    Args:
      u: horizontal flow components as HxW np.ndarray
      v: vertical flow components as HxW np.ndarray
      return_rgb: bool, whether to return RGB or BGR
      colorwheel: provide a num_colors-by-3 color numpy nd.array
        which will be used as color wheel
      convert_to_bgr: bool, whether to change ordering and
        output BGR instead of RGB

    Returns:
      HxWx3 uint8 numpy ndarray.
    """
    vis = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1

        idx = (rad <= 1)
        # Increase saturation with radius:
        col[idx] = 1 - rad[idx] * (1-col[idx])
        # Out of range:
        col[~idx] = col[~idx] * 0.75
        # BGR or RGB?
        ch_idx = i if return_rgb else 2-i
        vis[:, :, ch_idx] = np.floor(255.0 * col)
    return vis


def colorize_flow(
        flow: np.ndarray,
        max_val: float = None,
        return_rgb: bool = True) -> np.ndarray:
    """
    Returns the widely used flow visualization.

    This is a port of the C++/MATLAB code from
    https://people.csail.mit.edu/celiu/OpticalFlow, also based on
    https://github.com/tomrunia/OpticalFlow_Visualization.

    :param flow:       stacked u, v flow components as HxWx2 numpy ndarray.
    :param max_val:    float, if not None, flow values will be clipped to
                       the range [0, max_val].
    :param return_rgb: bool, set to False if you work with BGR images.
    :return:  HxWx3 uint8 numpy ndarray.
    """
    # Sanity check
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError('Input flow must be of shape HxWx2!')
    # Clip values
    if max_val is not None:
        flow = np.clip(flow, 0, max_val)

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Normalize flow
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    if rad_max > 0:
        u = u / rad_max
        v = v / rad_max

    return colorize_uv(u, v, return_rgb)

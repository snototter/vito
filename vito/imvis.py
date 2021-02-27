#!/usr/bin/env python
# coding=utf-8
"""Handy visualization tools."""

import numpy as np
import logging

from . import colormaps
from . import imutils


# Use logging module (fallback to default print) to log imutils set up (e.g. OpenCV vs Pillow).
__log_fx = print if len(logging.getLogger().handlers) == 0 else logging.getLogger().info


try:
    # Try to load OpenCV (in case you installed it in your workspace)
    import cv2
    __log_fx("vito.imvis will use OpenCV to display images via 'imshow'.")  # pragma: no cover
    
    def imshow(img_np, title="Image", flip_channels=False, wait_ms=-1):  # pragma: no cover
        """
        Convenience 1-liner to display image and wait for key input.

        :param img_np:    Should be provided as RGB, otherwise use flip_channels=True
        :param title:     Window title
        :param wait_ms:   cv2.waitKey() input
        :param flip_channels: if you want to display a BGR image properly, set this
                            to True
        :return: Pressed key or -1, i.e. cv2.waitKey() output
        """
        if not flip_channels:
            disp = imutils.flip_layers(img_np)
        else:
            disp = img_np
        cv2.imshow(title, disp)
        return cv2.waitKey(wait_ms)
except:
    from PIL import Image
    __log_fx("vito.imvis will use Pillow to display images via 'imshow' (OpenCV could not be loaded).")  # pragma: no cover

    def imshow(img_np, title="Image", flip_channels=False, **kwargs):  # pragma: no cover
        """
        Convenience 1-liner to display image. This implementation uses
        PIL (which uses your OS default viewer).
        'kwargs' will silently be ignored and are only provided to be
        compatible with the OpenCV-based 'imshow' version (which will be
        loaded in case 'cv2' is installed in your python workspace).

        Note that the window usually doesn't block!

        :param img:    Should be provided as RGB, otherwise use flip_channels=True
        :param title:  Window title
        :param flip_channels: if you want to display a BGR image properly, set this
                            to True
        :return: -1 for compatibility reasons (the same return value as if you
                        used the OpenCV-based version and there was no key press)
        """
        if flip_channels:
            disp = imutils.flip_layers(img_np)
        else:
            disp = img_np
        im = Image.fromarray(disp)
        im.show(title=title)
        return -1


# Colors used to easily distinguish different IDs (e.g. when visualizing
# tracking results).
exemplary_colors = [
    (255,   0,  0),   # red
    (  0, 200,   0),  # light green(ish)
    (  0,   0, 255),  # deep blue
    (230, 230,   0),  # yellow(ish)
    (230,   0, 230),  # magenta(ish)
    (  0, 230, 230),  # cyan(ish)
    (255, 128,   0),  # orange
    (255, 128, 128),  # skin(ish)
    (128,  64,   0),  # brown(ish)
    (160, 160, 160),  # gray(ish)
    (  0, 128, 255),  # light blue
    (153,  77, 204)   # lilac
]


def color_by_id(id, flip_channels=False):
    """Returns a color tuple (rgb) to colorize labels, identities,
    segments, trajectories, etc."""
    col = exemplary_colors[id % len(exemplary_colors)]
    if flip_channels:
        return (col[2], col[1], col[0])
    return col


def pseudocolor(values, limits=[0.0, 1.0], color_map=colormaps.colormap_parula_rgb):
    """
    Return a HxWx3 pseudocolored representation of the input matrix.

    :param values: A single channel, HxW or HxWx1 numpy ndarray.
        NaN or Inf values will be colorized by color_map[0].
    :param limits: [min, max] to clip the input values. If limits is None or
        any of min/max is None, the corresponding limits will be computed from
        the input values.
    :param color_map: The color map to be used, see colormaps.py

    :return: a HxWx3 colorized representation.
    """
    # Sanity checks
    if values is None:
        return None
    values = values.copy()

    if values.ndim != 2:
        if values.ndim > 3 or (values.ndim == 3 and values.shape[2] > 1):
            raise ValueError('Input to pseudocoloring must be a single channel data matrix, shaped (H,W) or (H,W,1)!')
        values = values.reshape((values.shape[0], values.shape[1] if values.ndim > 2 else 1))
    
    valid = np.isfinite(values)
    if not np.any(valid):
        return None

    if limits is None:
        limits = [np.min(values[valid]), np.max(values[valid])]
    if limits[0] is None:
        limits[0] = np.min(values[valid])
    if limits[1] is None:
        limits[1] = np.max(values[valid])
    
    values[np.logical_not(valid)] = limits[0]

    values = values.astype(np.float64)
    lut = np.asarray(color_map)
    interval = (limits[1] - limits[0]) / 255.0
    # Clip values to desired limits
    values[values < limits[0]] = limits[0]
    values[values > limits[1]] = limits[1]
    # Compute lookup values
    if interval > 0:
        lookup_values = np.floor((values - limits[0]) / interval).astype(np.int32)
    else:
        # Fall back to maximum value
        lookup_values = np.zeros(values.shape, dtype=np.int32)
    colorized = lut[lookup_values].astype(np.uint8)
    return colorized


def overlay(img1, alpha1, img2, mask=None):
    """
    Overlay two images with alpha blending, s.t.
        out = img1 * alpha1 + img2 * (1-alpha1).
    Optionally, only overlays those parts from img2 which are
    indicated by non-zero mask pixels:
        out = blended if mask > 0
              img1 else
    Output dtype will be the same as img1.dtype.
    Only float32, float64 and uint8 are supported.
    """
    if alpha1 < 0.0 or alpha1 > 1.0:
        raise ValueError('Weight factor must be in [0,1]')

    # Allow overlaying a grayscale image on top of a color image (and vice versa)
    if len(img1.shape) == 2:
        channels1 = 1
    else:
        channels1 = img1.shape[2]

    if len(img2.shape) == 2:
        channels2 = 1
    else:
        channels2 = img2.shape[2]

    if channels1 != channels2 and not (channels1 == 1 or channels1 == 3):
        raise ValueError('Can only extrapolate single channel image to the other''s dimension')

    if channels1 == 1 and channels2 > 1:
        if img1.ndim == 2:
            img1 = np.repeat(img1[:, :, np.newaxis], channels2, axis=2)
        else:
            img1 = np.repeat(img1[:, :, :], channels2, axis=2)
    if channels2 == 1 and channels1 > 1:
        if img2.ndim == 2:
            img2 = np.repeat(img2[:, :, np.newaxis], channels1, axis=2)
        else:
            img2 = np.repeat(img2[:, :, :], channels1, axis=2)

    num_channels = 1 if len(img1.shape) == 2 else img1.shape[2]

    # Convert to float64, [0,1]
    if img1.dtype == np.uint8:
        scale1 = 255.0
    elif img1.dtype in [np.float32, np.float64]:
        scale1 = 1.0
    else:
        raise ValueError('Datatype {} is not supported'.format(img1.dtype))
    img1 = img1.astype(np.float64) / scale1

    target_dtype = img2.dtype
    if img2.dtype == np.uint8:
        scale2 = 255.0
    elif img2.dtype in [np.float32, np.float64]:
        scale2 = 1.0
    else:
        raise ValueError('Datatype {} is not supported'.format(img2.dtype))
    img2 = img2.astype(np.float64) / scale2

    if mask is None:
        out = alpha1 * img1 + (1. - alpha1) * img2
    else:
        if mask.shape[0] != img1.shape[0] or mask.shape[1] != img1.shape[1] \
                or (mask.ndim > 2 and mask.shape[2] != 1):
            raise ValueError('Mask must be 2D and of same width/height as inputs')
        if num_channels == 1:
            img2 = np.where(mask.reshape(img2.shape) > 0, img2, img1)
        else:
            if mask.ndim == 2:
                _mask = np.repeat(mask[:, :, np.newaxis], num_channels, axis=2)
            else:
                _mask = np.repeat(mask[:, :], num_channels, axis=2)
            img2 = np.where(_mask > 0, img2, img1)
        out = alpha1 * img1 + (1. - alpha1) * img2
    return (scale2 * out).astype(target_dtype)

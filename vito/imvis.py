#!/usr/bin/env python
# coding=utf-8
"""Handy visualization tools."""

import numpy as np
import time
from PIL import Image

from vito import colormaps, imutils



def _pil_imshow(
        img_np: np.ndarray,
        title: str ="Image",
        flip_channels: bool = False,
        wait_ms: int = -1
    ) -> int:  # pragma: no cover
    """
    Convenience 1-liner to display an image. This implementation uses
    PIL (which uses your OS default viewer).

    Args:
        img: Should be provided as RGB, otherwise set `flip_channels=True`.
        title: Window title
        flip_channels: if you want to display a BGR image properly, set this
            to True
        wait_ms: If > 0, this call will block the current thread for the
            specified amount of milliseconds.
    Returns:
        The constant `-1` for compatibility reasons. This is the same return
        value as if you used the OpenCV backend and there was no key press.
    """
    if flip_channels:
        disp = imutils.flip_layers(img_np)
    else:
        disp = img_np
    im = Image.fromarray(disp)
    im.show(title=title)
    if wait_ms > 0:
        time.sleep(wait_ms / 1e3)
    return -1


try:
    # Try to load OpenCV (in case you installed it in your workspace)
    import cv2
    
    def imshow(
            img_np: np.ndarray,
            title: str = "Image",
            flip_channels: bool = False,
            wait_ms: int = -1
        ) -> int:  # pragma: no cover
        """
        Convenience 1-liner to display an image and wait for key input.

        Args:
            img_np: Should be provided as RGB, otherwise set `flip_channels=True`.
            title: Window title
            wait_ms: Number of milliseconds to wait for user input. This will
                be forwarded to the internal `cv2.waitKey()` call.
            flip_channels: If your input is BGR, set this to True for correct
                color display.

        Returns:
            Pressed key code or -1, i.e. the `cv2.waitKey()` output.
        """
        if img_np is None:
            return -1
        
        if not flip_channels:
            disp = imutils.flip_layers(img_np)
        else:
            disp = img_np
        
        # Common mistake: a boolean mask is not a valid input for cv2.imshow,
        # this would cause a cv2.error and we would fall back to PIL's imshow
        # which opens a new window for each image.
        # To avoid this, we convert boolean masks to uint8 images:
        if disp.dtype == bool:
            disp = disp.astype(np.uint8) * 255

        try:
            cv2.imshow(title, disp)
            if wait_ms == 0:
                return -1
            else:
                return cv2.waitKey(wait_ms)
        except cv2.error:
            # If opencv is installed headless, cv2.imshow will raise an error.
            # I couldn't find an easy workaround to check for headless setup
            # before the actual imshow call, thus we catch the error and then
            # change vito's imshow to the pillow fallback:
            global imshow
            imshow = _pil_imshow
            imshow(img_np, title=title, flip_channels=flip_channels)
except:
    imshow = _pil_imshow


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


def color_by_id(id: int, flip_channels: bool = False) -> tuple:
    """
    Returns a color `tuple(int, int, int)` to colorize labels, identities,
    segments, trajectories, etc.

    By default, the returned tuple will hold the RGB values. If you need BGR
    instead, set `flip_channels = True`.
    """
    col = exemplary_colors[id % len(exemplary_colors)]
    if flip_channels:
        return (col[2], col[1], col[0])
    return col


def pseudocolor(
        values: np.ndarray,
        limits: list = [0.0, 1.0],
        color_map: list = colormaps.viridis
    ) -> np.ndarray:
    """
    Returns a HxWx3 pseudocolored representation of the input matrix.

    :param values: A single channel, HxW or HxWx1 numpy ndarray.
        NaN or Inf values will be colorized by color_map[0].
    :param limits: `[min, max]` to clip the input values. If limits is None or
        any of min/max is None, the corresponding limit(s) will be computed
        from the input values.
    :param color_map: The color map to be used, see `vito.colormaps`, i.e. a
        list of RGB colors: list(tuple(int, int, int)).

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


def overlay(
        img1: np.ndarray,
        alpha1: float,
        img2: np.ndarray,
        mask: np.ndarray = None
    ) -> np.ndarray:
    """
    Overlays two images with alpha blending, s.t.
    `out = img1 * alpha1 + img2 * (1-alpha1)`.

    Optionally, only overlays those parts from `img2` which are
    indicated by non-zero mask pixels:
        out = blended  if mask > 0
              img1     else

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

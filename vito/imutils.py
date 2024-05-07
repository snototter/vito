#!/usr/bin/env python
# coding=utf-8
"""Utilities you'll often need when working with images ;-)"""

# TODOs:
# * imread/imsave should use tifffile for tiff files - supports multi-band images

import io
from pathlib import Path
from typing import Any, Sequence, Union
import numpy as np
from PIL import Image
from PIL import ImageFilter


def flip_layers(nparray: np.ndarray) -> np.ndarray:
    """
    Flips the R & B channels.

    Also accepts rgbA/bgrA and single channel images without crashing.
    """
    if nparray is None:
        return None
    if len(nparray.shape) == 3:
        if nparray.shape[2] == 4:
            # We got xyzA, make zyxA
            return nparray[..., [2, 1, 0, 3]]
        else:
            return nparray[:, :, ::-1]
    return nparray


def ensure_c3(nparray: np.ndarray) -> np.ndarray:
    """
    Ensures that the output image has 3 channels.

    Valid inputs: monochrome, 3- and 4-channel images.
    """
    if nparray is None:
        return None
    if nparray.ndim == 2:
        return np.repeat(nparray[:, :, np.newaxis], 3, axis=2)
    if nparray.ndim == 3:
        if nparray.shape[2] == 1:
            return np.repeat(nparray[:, :, :], 3, axis=2)
        elif nparray.shape[2] == 3:
            return nparray
        elif nparray.shape[2] == 4:
            return nparray[:, :, :3]
    raise ValueError("Invalid input to ensure_c3, input must be a 1-, 3- or 4-channel image.")


def rgb2gray(
        nparray: np.ndarray,
        is_bgr: bool = False
        ) -> np.ndarray:
    """
    Converts the RGB image to grayscale
    using `L = 0.2989 R + 0.5870 G + 0.1140 B`.

    Args:
      nparray: HxWxC numpy array.
      is_bgr: If True, the input channels are in BGR format. Otherwise, they
        are assumed to be RGB.
    """
    if nparray is None:
        return None
    if nparray.ndim == 2 or (nparray.ndim == 3 and nparray.shape[2] == 1):
        return nparray
    if nparray.ndim == 3 and nparray.shape[2] == 2:
        raise ValueError('Cannot convert a 2 channel input image to grayscale.')
    if is_bgr:
        return np.dot(nparray[..., :3], [0.1140, 0.5870, 0.2989]).astype(nparray.dtype)
    else:
        return np.dot(nparray[..., :3], [0.2989, 0.5870, 0.1140]).astype(nparray.dtype)


# Alias for convenience
grayscale = rgb2gray


def __imsave_checks(
        filename: Union[str, Path],
        image: np.ndarray
        ) -> None:
    if not isinstance(filename, Path):
        filename = Path(filename)
    if not filename.exists():
        filename.absolute().parents[0].mkdir(parents=True, exist_ok=True)
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f'Image is not a NumPy array, but `{type(image)}`. Check the'
            'parameter order: (`filename`, `image`, `flip_channels`)!')


try:
    # Try to load OpenCV (in case you installed it in your workspace)
    import cv2

    def imsave(
            filename: Union[str, Path],
            image: np.ndarray,
            flip_channels: bool = False
            ) -> None:  # pragma: no cover
        """Stores the image using OpenCV.

        This function *will create the output directory* if it does not
        exist yet.
        If you have a BGR image, set `flip_channels = True` to store it
        correctly.
        """
        __imsave_checks(filename, image)

        # To be compatible with the Pillow/PIL version (see below), we have to
        # invert the flip_channels flag.
        if not flip_channels:
            cv2.imwrite(str(filename), flip_layers(image))
        else:
            cv2.imwrite(str(filename), image)
except:
    # Fall back to Pillow
    def imsave(
            filename: Union[str, Path],
            image: np.ndarray,
            flip_channels: bool = False
            ) -> None:
        """Stores the image using PIL/Pillow.

        This function *will create the output directory* if it does not
        exist yet.
        If you have a BGR image, set `flip_channels = True` to store it
        correctly.
        """
        __imsave_checks(filename, image)
        if flip_channels:
            im_np = flip_layers(image)
        else:
            im_np = image
        Image.fromarray(im_np).save(str(filename))


def imread(
        filename: Union[str, Path],
        flip_channels: bool = False,
        **kwargs
        ) -> np.ndarray:
    """Loads an image (using PIL/Pillow) into a NumPy array.

    Multi-channel images are returned as RGB format unless you
    set `flip_channels=True` (which will swap the R & B channels).

    Optional kwargs will be passed on to PIL's Image.convert(). Thus, you can
    specify PIL's loading 'mode', e.g. 'RGB' for color, 'RGBA' for a
    transparent image and 'L' for grayscale.
    """
    if filename is None:
        return None
    if not isinstance(filename, Path):
        filename = Path(filename)
    if not filename.exists() or not filename.is_file():
        raise FileNotFoundError(f'Image `{filename}` does not exist')
    if not isinstance(flip_channels, bool):
        raise TypeError(
            f'Parameter `flip_channels` must be bool, not `{type(flip_channels)}` '
            '- you probably forgot the keyword for the PIL **kwargs!')
    # PIL loads 16-bit PNGs as np.int32 ndarray. If we need to support other
    # bit depths, we should look into using pypng, see documentation at
    # https://pythonhosted.org/pypng/ex.html#reading
    # For now, PIL provides all we need
    image = np.asarray(Image.open(filename).convert(**kwargs))

    if flip_channels:
        return flip_layers(image)
    else:
        return image


def ndarray2memory_file(
        np_data: np.ndarray,
        format: str = 'png'
        ) -> io.BytesIO:
    """Converts the numpy (image) array to a BytesIO stream.

    Useful to transfer an image via sockets.
    """
    if np_data is None:
        return None
    img = Image.fromarray(np_data)
    img_memory_file = io.BytesIO()
    img.save(img_memory_file, format)
    return img_memory_file


def memory_file2ndarray(
        memfile: io.BytesIO
        ) -> np.ndarray:
    """Loads an image stored in the given BytesIO memory file."""
    if memfile is None:
        return None
    return np.asarray(Image.open(memfile))


def clip_rect_to_image(
        rect: Sequence[Any],
        img_width: int,
        img_height: int
        ) -> tuple:
    """Ensures that the rectangle is within the image boundaries.

    Explicitly casts all entries to integer.

    Args:
      rect: list/tuple (l,t,w,h)
      img_width: int
      img_height: int

    Returns:
      (l, t, w, h)
    """
    l, t, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    # Right/bottom bounds are exclusive
    r = max(0, min(img_width, l+w))
    b = max(0, min(img_height, t+h))
    # Left/top bounds are inclusive
    l = max(0, l)
    t = max(0, t)
    return (l, t, r-l, b-t)


def is_valid_bbox(rect: Sequence[Any]) -> bool:
    """Checks if left/top >= 0 and W/H > 0"""
    return rect[0] >= 0 and rect[1] >= 0 and rect[2] > 0 and rect[3] > 0


def apply_on_bboxes(
        image_np: np.ndarray,
        bboxes: list,
        func,
        *args,
        **func_kwargs
        ) -> np.ndarray:
    """
    Applies the given function (which returns a modified image) on each
    bounding box.

    This function takes care of proper clipping of each bounding box, roi
    extraction and copying back the results.

    Args:
      image_np: numpy ndarray, input image
      bboxes: list of bounding boxes, i.e. [(l, t, w, h), (...)]
      func: function handle to apply on each bounding box
      func_kwargs: optional kwargs passed on to the function

    Returns:
      numpy ndarray, a copy of the input image after applying the given
      function on each (valid) bounding box
    """
    if image_np is None:
        return None
    # Ensure the image is writeable
    image_np = image_np.copy()
    # Prevent invalid memory access
    bboxes = [clip_rect_to_image(bb, image_np.shape[1], image_np.shape[0])
              for bb in bboxes]
    bboxes = [b for b in bboxes if is_valid_bbox(b)]
    for bb in bboxes:
        l, t, w, h = bb
        roi = image_np[t:t+h, l:l+w]
        image_np[t:t+h, l:l+w] = func(roi, *args, **func_kwargs)
    return image_np


def roi(image_np: np.ndarray, rect: Sequence[Any]) -> np.ndarray:
    """Returns the cropped ROI, with rect = [l, t, w, h]."""
    if image_np is None or rect is None or any([r is None for r in rect]):
        return None
    l, t, w, h = rect
    r, b = l+w, t+h
    img_height, img_width = image_np.shape[0], image_np.shape[1]

    # Right/bottom bounds are exclusive; left/top are inclusive
    l = max(0, min(img_width-1, l))
    r = max(0, min(img_width, r))
    t = max(0, min(img_height-1, t))
    b = max(0, min(img_height, b))

    if len(image_np.shape) == 2:
        return image_np[t:b, l:r]
    return image_np[t:b, l:r, :]


# Alias for convenience
crop = roi


def pad(
        image_np: np.ndarray,
        border: int,
        color: tuple = None
        ) -> np.ndarray:
    """Pads the image by 'border' pixels in each direction.

    :param image_np: HxWxC numpy ndarray.
    :param border: int > 0.
    :param color: If None, border will be transparent.
                  Otherwise, the border will be drawn in the
                  corresponding color (RGB/BGR tuple or scalar).
    :return: (H+2*border)x(W+2*border)xC or x4 numpy ndarray.
    """
    if image_np is None:
        return None
    if border < 1:
        raise ValueError("Border must be > 0")
    h, w = image_np.shape[:2]
    c = image_np.shape[2] if image_np.ndim == 3 else 1
    if color is None:
        # RGB/BGR+A output
        out = np.zeros((h + 2*border, w + 2*border, 4), dtype=image_np.dtype)
        mask = np.zeros((h, w), dtype=image_np.dtype)
        mask[border:-border, border:-border] = np.iinfo(image_np.dtype).max
        out[border:-border, border:-border, 3] = mask
        if c in [1, 3, 4]:
            for i in range(max(3, c)):
                out[border:-border, border:-border, i] = image_np if c == 1 else image_np[:, :, i]
            return out
        else:
            raise RuntimeError("Invalid input shape, only 1, 3 or 4 channel images are supported!")
    else:
        out = np.zeros((h + 2*border, w + 2*border, c), dtype=image_np.dtype)
        if isinstance(color, (list, tuple, np.ndarray)):
            for i in range(c):
                out[:, :, i] = 0 if i >= len(color) else color[i]
        else:
            out[:] = color
        for i in range(c):
            out[border:-border, border:-border, i] = image_np if c == 1 else image_np[:, :, i]
        return out


def pixelate(
        image_np: np.ndarray,
        block_width: int = 5,
        block_height: int = -1
        ) -> np.ndarray:
    """
    Pixelate the image into blocks of size WxH.
    If H is -1, then H=W. Otherwise, rectangular blocks will
    be used (unless W is negative, in which case W=H will be used).
    """
    if image_np is None:
        return None
    if block_width < 1 and block_height < 1:
        raise ValueError("Either block width or height must be >= 1")
    block_width = block_height if block_width < 1 else block_width
    block_height = block_width if block_height < 1 else block_height
    resized_height = max(1, image_np.shape[0] // block_height)
    resized_width = max(1, image_np.shape[1] // block_width)
    pixelized_roi = Image.fromarray(image_np) \
        .resize((resized_width, resized_height), resample=Image.NEAREST) \
        .resize((image_np.shape[1], image_np.shape[0]), resample=Image.NEAREST)
    return np.asarray(pixelized_roi)


def gaussian_blur(
        image_np: np.ndarray,
        radius: int = 15
        ) -> np.ndarray:
    """Blurs the image using a Gaussian kernel with the given radius."""
    if image_np is None:
        return None
    img = Image.fromarray(image_np).filter(
        ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img)


def set_to(
        image_np: np.ndarray,
        value: Any
        ) -> np.ndarray:
    """Sets the image pixels to the given value.

    The parameter `value` can either be a tuple of float/int (specifying a
    color) or a scalar value (int/float).
    """
    if image_np is None or value is None:
        return None
    if isinstance(value, tuple):
        channels = 1 if image_np.ndim < 3 else image_np.shape[2]
        if len(value) < channels:
            raise ValueError('Input value has less entries than image channels.')
        if channels == 1:
            image_np[:] = value[0]
        else:
            for ch in range(channels):
                image_np[:, :, ch] = value[ch]
    else:
        image_np[:] = value
    return image_np


def _make_stack_compatible(
        img1: np.ndarray,
        img2: np.ndarray,
        horizontal: bool
        ) -> tuple:
    """
    Returns two "compatible" images to be used for horizontal or vertical
    stacking, i.e. they'll have the correct size, same dtype and number of
    channels.
    """
    # Check for compatible sizes
    if horizontal and img1.shape[0] != img2.shape[0]:
        raise ValueError('Images must have the same height for horizontal concatenation.')
    if not horizontal and img1.shape[1] != img2.shape[1]:
        raise ValueError('Images must have the same width for vertical concatenation.')
    # Check data types
    if img1.dtype != img2.dtype:
        if img1.dtype.itemsize > img2.dtype.itemsize:
            img2 = img2.astype(img1.dtype)
        else:
            img1 = img1.astype(img2.dtype)
    # Same dimensionality
    if img1.ndim < img2.ndim:
        img1 = img1[:, :, np.newaxis]
    elif img2.ndim < img1.ndim:
        img2 = img2[:, :, np.newaxis]
    # Check layers
    if img1.ndim == 3 and img1.shape[2] != img2.shape[2]:
        if img1.shape[2] == 1:
            img1 = np.repeat(img1[:, :, :], img2.shape[2], axis=2)
        elif img2.shape[2] == 1:
            img2 = np.repeat(img2[:, :, :], img1.shape[2], axis=2)
        else:
            raise ValueError('If channels are different, one of the inputs must be single-channel.')
    return img1, img2


def hstack(
        img1: np.ndarray,
        img2: np.ndarray
        ) -> np.ndarray:
    """Horizontally concatenates the two given images."""
    if img1 is None:
        return img2
    if img2 is None:
        return img1
    img1, img2 = _make_stack_compatible(img1, img2, True)
    return np.hstack((img1, img2))


def vstack(
        img1: np.ndarray,
        img2: np.ndarray
        ) -> np.ndarray:
    """Vertically concatenates the two given images."""
    if img1 is None:
        return img2
    if img2 is None:
        return img1
    img1, img2 = _make_stack_compatible(img1, img2, False)
    return np.vstack((img1, img2))


def concat(
        img1: np.ndarray,
        img2: np.ndarray,
        horizontal: bool = True
        ) -> np.ndarray:
    """Concatenates the two given images either horizontally or vertically."""
    if horizontal:
        return hstack(img1, img2)
    else:
        return vstack(img1, img2)


def rotate90(image_np: np.ndarray) -> np.ndarray:
    """Rotates the given image by 90 degrees counter-clockwise."""
    if image_np is None:
        return None
    return np.rot90(image_np)


def rotate180(image_np: np.ndarray) -> np.ndarray:
    """Rotates the given image by 180 degrees."""
    if image_np is None:
        return None
    return np.fliplr(np.flipud(image_np))


def rotate270(image_np: np.ndarray) -> np.ndarray:
    """Rotates the given image by 270 degrees counter-clockwise."""
    if image_np is None:
        return None
    return np.rot90(image_np, -1)


def noop(x: Any) -> Any:
    """No-operation/identity function."""
    return x

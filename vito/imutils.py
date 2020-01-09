#!/usr/bin/env python
# coding=utf-8
"""Utilities you'll often need when working with images ;-)"""

# TODOs:
# * imread/imsave should use tifffile for tiff files - supports multi-band images

import io
import os
import numpy as np
from PIL import Image


def flip_layers(nparray):
    """
    Flip RGB to BGR image data (numpy ndarray).
    Also accepts rgbA/bgrA and single channel images without crashing.
    """
    if len(nparray.shape) == 3:
        if nparray.shape[2] == 4:
            # We got xyzA, make zyxA
            return nparray[..., [2, 1, 0, 3]]
        else:
            return nparray[:, :, ::-1]
    return nparray


try:
    # Try to load OpenCV (in case you installed it in your workspace)
    import cv2

    def imsave(filename, image, flip_channels=False):
        """Store an image using OpenCV."""
        # To be compatible with the Pillow/PIL version (see below), we have to
        # invert the flip_channels flag.
        if not flip_channels:
            cv2.imwrite(filename, flip_layers(image))
        else:
            cv2.imwrite(filename, image)
except:
    # Fall back to Pillow
    def imsave(filename, image, flip_channels=False):
        """Store an image using PIL/Pillow, optionally flipping layers, i.e. BGR -> RGB."""
        if flip_channels:
            im_np = flip_layers(image)
        else:
            im_np = image
        Image.fromarray(im_np).save(filename)


def imread(filename, flip_channels=False, **kwargs):
    """Load an image (using PIL) into a NumPy array.
    Multi-channel images are returned as RGB unless you set flip_channels=True.
    
    Optional kwargs will be passed on to PIL's Image.convert(). Thus, you can
    specifiy PIL's loading 'mode', e.g. 'RGB' for color, 'RGBA' for a
    transparent image and 'L' for grayscale.
    """
    if filename is None:
        return None
    if not os.path.exists(filename):
        raise FileNotFoundError('Image %s does not exist' % filename)
    # PIL loads 16-bit PNGs as np.int32 ndarray. If we need to support other
    # bit depths, we should look into using pypng, see documentation at
    # https://pythonhosted.org/pypng/ex.html#reading
    # For now, PIL provides all we need
    image = np.asarray(Image.open(filename).convert(**kwargs))
    
    if flip_channels:
        return flip_layers(image)
    else:
        return image


def ndarray2memory_file(np_data, format='png'):
    """Convert numpy (image) array to ByteIO stream.
    Useful to stream an image via sockets."""
    img = Image.fromarray(np_data)
    img_memory_file = io.BytesIO()
    img.save(img_memory_file, format)
    return img_memory_file


def memory_file2ndarray(memfile):
    """Load an image stored in the given io.BytesIO memory file memfile."""
    if memfile is None:
        return None
    return np.asarray(Image.open(memfile))


def clip_rect_to_image(rect, img_width, img_height):
    """Ensure that the rectangle is within the image boundaries.
    Explicitly casts all entries to integer.
    :param rect: list/tuple (l,t,w,h)
    :param img_width: int
    :param img_height: int
    :return: (l, t, w, h)
    """
    l, t, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    # Right/bottom bounds are exclusive
    r = max(0, min(img_width, l+w))
    b = max(0, min(img_height, t+h))
    # Left/top bounds are inclusive
    l = max(0, l)
    t = max(0, t)
    return (l, t, r-l, b-t)


def is_valid_bbox(rect):
    """Left/top must be >= 0, W/H must be > 0"""
    return rect[0] >= 0 and rect[1] >= 0 and rect[2] > 0 and rect[3] > 0


def apply_on_bboxes(image_np, bboxes, func, **func_kwargs):
    """Applies the function func (which returns a modified image) on each
    bounding box. Takes care of proper clipping, roi extraction and copying
    back the results.
    :param image_np:    numpy ndarray, input image
    :param bboxes:      list of bounding boxes, i.e. [(l, t, w, h), (...)]
    :param func:        function handle to apply on each bounding box
    :param func_kwargs: optional kwargs passed on to the function
    :return: numpy ndarray, a copy of the input image after applying the given
             function on each (valid) bounding box
    """
    # Ensure the image is writeable
    image_np = image_np.copy()
    # Prevent invalid memory access
    bboxes = [clip_rect_to_image(bb, image_np.shape[1], image_np.shape[0])
            for bb in bboxes]
    bboxes = [b for b in bboxes if is_valid_bbox(b)]
    for bb in bboxes:
        l, t, w, h = bb
        roi = image_np[t:t+h, l:l+w]
        image_np[t:t+h, l:l+w] = func(roi, **func_kwargs)
    return image_np


def roi(image_np, rect):
    """Returns the cropped ROI, with rect = [l, t, w, h]."""
    if rect is None or any([r is None for r in rect]):
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

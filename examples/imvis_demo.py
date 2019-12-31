#!/usr/bin/env python
# coding=utf-8
"""
Showcasing data inspection

This example script needs PIL (Pillow package) to load images from disk.
"""

import os
import sys

from PIL import Image
import numpy as np

# Extend the python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from vito import colormaps
from vito import imutils
from vito import imvis


if __name__ == "__main__":
    # Standard loading/display
    lena = imutils.imread('lena.jpg', mode='RGB')
    imvis.imshow(lena)

    # Load as BGR
    lena = imutils.imread('lena.jpg', mode='RGB', flip_channels=True)
    imvis.imshow(lena)

    # Load a single-channel image
    peaks = imutils.imread('peaks.png', mode='L')
    # Colorize it
    colorized = imvis.pseudocolor(peaks, limits=None, color_map=colormaps.colormap_parula_rgb)
    imvis.imshow(colorized)
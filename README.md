# vito - Vision Tools
[![View on PyPI](https://img.shields.io/pypi/v/vito.svg)](https://pypi.org/project/vito)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/snototter/vito/blob/master/LICENSE?raw=true)

Python utilities for common computer vision tasks.
The goal of this package is to provide a lightweight, python-only package helping you with standard/recurring image manipulation tasks.

## Dependencies
* `numpy`
* `PIL`

## Examples
* Pseudocoloring:
```python
# Load a single-channel image
peaks = imutils.imread('peaks.png', mode='L')
# Colorize it
colorized = imvis.pseudocolor(peaks, limits=None, color_map=colormaps.colormap_parula_rgb)
imvis.imshow(colorized)
```

## Changelog
* Upcoming `0.1.2` (`0.2.0` if flow is included)
  * Optical flow (Middlebury .flo format) I/O
  * Support saving images
  * Colorization to visualize tracking results
* `0.1.1`
  * Changed supported python versions for legacy tests
* `0.1.0`
  * First actually useful release
  * Contains most of the functionality of `pvt` (a library I developed throughout my studies)
    * `cam_projections` - projective geometry, lens distortion/rectification (Plumb Bob model), etc.
    * `colormaps` - colormap definitions for visualization (jet, parula, magma, viridis, etc.)
    * `imutils` - image loading, conversion, RoI handling (e.g. apply functions on several patches of an image)
    * `imvis` - visualization helpers, e.g. pseudocoloring or overlaying images
    * `pyutils` - common python functions (timing code, string manipulation, list sorting/search, etc.)
* `0.0.1`
  * Initial public release
  * Contains common python/language and camera projection utils

## TODO List
* flow - visualize
* anonymization utils
* augmentation

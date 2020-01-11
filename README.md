# vito - Vision Tools
[![View on PyPI](https://badge.fury.io/py/vito.svg)](https://pypi.org/project/vito)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/snototter/vito/blob/master/LICENSE?raw=true)

Python utilities for common computer vision tasks.
The goal of this package is to provide a lightweight package helping you with standard/recurring image manipulation tasks.


## Dependencies
* `numpy`
* `Pillow`


## Examples
* Pseudocoloring:
```python
from vito import imutils
from vito import imvis

# Load a single-channel image
peaks = imutils.imread('peaks.png', mode='L')
# Colorize it
colorized = imvis.pseudocolor(peaks, limits=None, color_map=colormaps.colormap_parula_rgb)
imvis.imshow(colorized)
```
* Optical flow:
```python
from vito import flowutils
from vito import imvis

# Load optical flow file
flow = flowutils.floread('color_wheel.flo')
# Colorize it
colorized = flowutils.colorize_flow(flow)
imvis.imshow(colorized)
```
* Depth image stored as 16-bit PNG:
```python
from vito import imread
from vito import imvis

# Load 16-bit depth (will be of type np.int32)
depth = imutils.imread('depth.png')
# Colorize it
colorized = imvis.pseudocolor(depth, limits=None, color_map=colormaps.colormap_turbo_rgb)
imvis.imshow(colorized)
```


## Changelog
* `1.0.3`
  * Minor bug fix: handle invalid user inputs in `imvis`.
* `1.0.2`
  * Additional tests and minor improvements (potential bug fixes, especially for edge case inputs).
  * Ensure default image I/O parametrization always returns/saves/loads color images as RGB (even if OpenCV is available/used on your system).
* `1.0.1`
  * Fix colorizing boolean masks (where mask[:] = True or mask[:] = False).
* `1.0.0`
  * Rename flow package to `flowutils`.
* `0.3.2`
  * Rename colorization for optical flow.
* `0.3.1`
  * Fix `colormaps.by_name()` for grayscale.
* `0.3.0`
  * `apply_on_bboxes()` now supports optional kwargs to be passed on to the user-defined function handle.
  * Changed `imread()`'s default `mode` parameter to optional kwargs which are passed on to Pillow.
  * Raising error for non-existing files in `imread()`
  * Added `colormaps.by_name()` functionality.
  * Fixed bounding box clipping off-by-one issue.
  * Added `imutils` tests ensuring proper data types.
* `0.2.0`
  * Optical flow (Middlebury .flo format) I/O and visualization.
  * Support saving images.
  * Colorization to visualize tracking results.
* `0.1.1`
  * Changed supported python versions for legacy tests.
* `0.1.0`
  * First actually useful release.
  * Contains most of the functionality of `pvt` (a library I developed throughout my studies).
    * `cam_projections` - projective geometry, lens distortion/rectification (Plumb Bob model), etc.
    * `colormaps` - colormap definitions for visualization (jet, parula, magma, viridis, etc.)
    * `imutils` - image loading, conversion, RoI handling (e.g. apply functions on several patches of an image).
    * `imvis` - visualization helpers, e.g. pseudocoloring or overlaying images.
    * `pyutils` - common python functions (timing code, string manipulation, list sorting/search, etc.)
* `0.0.1`
  * Initial public release.
  * Contains common python/language and camera projection utils.

## TODO List
* anonymization utils
* augmentation

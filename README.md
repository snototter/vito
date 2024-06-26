# vito - Vision Tools
[![View on PyPI](https://badge.fury.io/py/vito.svg)](https://pypi.org/project/vito)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/vito.svg)](https://pypi.org/project/vito)
[![Build Status](https://app.travis-ci.com/snototter/vito.svg?branch=master)](https://app.travis-ci.com/snototter/vito)
[![Coverage Status](https://coveralls.io/repos/github/snototter/vito/badge.svg?branch=master)](https://coveralls.io/github/snototter/vito?branch=master)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/snototter/vito/blob/master/LICENSE?raw=true)

Python utilities for common computer vision tasks.
The goal of this package is to provide a lightweight python-only package for very basic image manipulation tasks, optical flow handling, pseudocoloring and camera geometry.


## Examples
* **Pseudocoloring:** 
  ```python
  from vito import imutils
  from vito import imvis

  # Load a single-channel image (data.dtype will be numpy.uint8)
  peaks = imutils.imread('peaks.png', mode='L')
  # Colorize it
  colorized = imvis.pseudocolor(peaks, limits=None, color_map=colormaps.colormap_viridis_rgb)
  imvis.imshow(colorized)

  # Load 16-bit depth stored as PNG (data.dtype will be numpy.int32)
  depth = imutils.imread('depth.png')
  # Colorize it
  colorized = imvis.pseudocolor(depth, limits=None, color_map=colormaps.colormap_turbo_rgb)
  imvis.imshow(colorized)

  ```
  Exemplary visualizations: colorizing via the `turbo` rainbow colormap (left); same data reduced to 11 bins colorized using `viridis` (right). Input data is obtained from two translated and scaled Gaussian distributions.
  ![Pseudocoloring Example](https://github.com/snototter/vito/raw/master/examples/visualizations/example-pseudocolor.png)
* **Optical flow:**
  ```python
  from vito import flowutils
  from vito import imvis

  # Load optical flow file
  flow = flowutils.floread('color_wheel.flo')
  # Colorize it
  colorized = flowutils.colorize_flow(flow)
  imvis.imshow(colorized)
  ```
  Exemplary visualization: Optical flow (standard color wheel visualization) and corresponding RGB frame for one frame of the [MPI Sintel Flow](http://sintel.is.tue.mpg.de) dataset.
  ![Optical Flow Example](https://github.com/snototter/vito/raw/master/examples/visualizations/example-flowvis.png)
* **Pixelation:**
  ```python
  from vito import imutils
  from vito import imvis

  img = imutils.imread('homer.png')
  rects = [(80, 50, 67, 84), (257, 50, 82, 75)]  # (Left, Top, Width, Height)
  anon = imutils.apply_on_bboxes(img, rects, imutils.pixelate)
  imvis.imshow(anon)
  ```
  Exemplary visualization: anonymization example using `imutils.apply_on_bboxes()` as shown above, with Gaussian blur kernel (`imutils.gaussian_blur()`, left) and pixelation (`imutils.pixelate()`, right), respectively.
  ![Anonymization Example](https://github.com/snototter/vito/raw/master/examples/visualizations/example-anon.png)
* For more examples (or if you prefer having a simple GUI to change visualization/analyze your data), see also the [**iminspect**](https://pypi.org/project/iminspect) package (which uses `vito` under the hood).


## Dependencies
* `numpy`
* `Pillow`


## Changelog
* `1.6.1`
  * House keeping: Internal adaptations to fix deprecated (NumPy) and changed (PIL) library usage.
  * CI updates to [PyPI's trusted publishing](https://docs.pypi.org/trusted-publishers/).
  * Code clean up (added type hints, docstrings, ...).
* `1.6.0`
  * Remove deprecated functionality (workarounds for Python versions older than 3.6).
  * Add type hints and update documentation.
  * Add parameter order check to `imsave` (that's the one thing I alwys mess up
    when coding without IDE).
  * `imsave` now creates the output directory structure if it did not exist.
* `1.5.3`
  * Mute initial notification about `imshow` backend.
  * Catch exception and adjust `imshow` backend if used with headless OpenCV.
* `1.5.2`
  * Changed `imshow` to skip waiting for user input if `wait_ms=0`.
  * Added (RGB) colormap handles for convenience.
  * Drop support for deprecated Python version 3.5.
* `1.5.1`
  * Changed handling of `None` inputs for image concatenation/stacking.
* `1.5.0`
  * Extends the `imutils` submodule (rotation, concatenation & noop).
  * Fix deprecation warnings with newer NumPy versions.
  * Extended test suite.
* `1.4.1`
  * Removes f-strings to fix compatibility for older python 3.5.
* `1.4.0`
  * Changed `imvis.overlay` to use a more intuitive signature.
  * Aliases for some `cam_projections` functions.
  * Spell-checked all files via `pyspelling`.
* `1.3.4`
  * Extended input handling for `imutils` (support single channel input to rgb2gray).
  * Aliases for some `imutils` functions.
  * Cleaning up tests, documentation, etc.
* `1.3.3`
  * Prevent nan values caused by floating point precision issues in the `cam_projections` submodule.
  * Remove the (empty) `tracking` submodule (to be added in a future release).
  * Update the submodule list.
* `1.3.2`
  * Support custom label maps in `detection2d` module.
  * Construct `BoundingBox`es from relative representations.
* `1.3.1`
  * Relative `BoundingBox` representation.
  * Support label lookup for `Detection` instances directly.
* `1.3.0`
  * Common representations and utilities for 2D object detection via the `detection2d` module.
    * `Detection` class to encapsulate object detections.
    * `BoundingBox` class to work with axis-aligned bounding boxes.
* `1.2.3`
  * Support sampling from colormaps.
  * Adjust tests to updated PIL version.
* `1.2.2`
  * Use explicit copies in `pseudocolor()` to prevent immutable assignment destination errors.
* `1.2.1`
  * Explicitly handle invalid (NaN and infinite) inputs to `pseudocolor()`.
* `1.2.0`
  * Add pixelation functionality for anonymization via `imutils`.
  * Add Gaussian blur to `imutils`.
* `1.1.5`
  * Extend projection utils.
* `1.1.4`
  * Explicitly handle `None` inputs to `imutils`.
* `1.1.3`
  * Fix transparent borders when padding.
* `1.1.2`
  * Add sanity checks to `imutils` which prevent interpreting optional PIL/cv2 parameters as custom parameters.
  * Add grayscale conversion to `imutils`.
* `1.1.1`
  * Maximum alpha channel value derived from data type.
* `1.1.0`
  * Added padding functionality.
* `1.0.4`
  * Improved test coverage.
  * Fixed potential future bugs - explicit handling of wrong/unexpected user inputs.
* `1.0.3`
  * Minor bug fix: handle invalid user inputs in `imvis`.
* `1.0.2`
  * Additional tests and minor improvements (potential bug fixes, especially for edge case inputs).
  * Ensure default image I/O parametrization always returns/saves/loads color images as RGB (even if OpenCV is available/used on your system).
* `1.0.1`
  * Fix colorizing boolean masks (where `mask[:] = True` or `mask[:] = False`).
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
  * Visualization utils for tracking results.
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

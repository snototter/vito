import numpy as np
import os
import pytest
from ..imutils import flip_layers, imread, apply_on_bboxes


def test_flip_layers():
    # Single channel image
    x = np.random.rand(3, 3)
    xf = flip_layers(x)
    assert np.all(x == xf)

    # rgb to bgr conversion
    x = np.random.rand(16, 32, 3)
    xf = flip_layers(x)
    for c in range(x.shape[2]):
        assert np.all(x[:, :, c] == xf[:, :, x.shape[2]-c-1])

    # rgbA should be flipped to bgrA
    x = np.random.rand(17, 5, 4)
    xf = flip_layers(x)
    for c in range(x.shape[2]):
        complement = 2-c if c < 3 else 3
        assert np.all(x[:, :, c] == xf[:, :, complement])


def test_imread():
    exdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples')
    assert imread(None) is None
    for fn in ['', 'a-non-existing.file']:
        with pytest.raises(FileNotFoundError):
            imread(fn)
    # Load RGB JPG
    img = imread(os.path.join(exdir, 'lena.jpg'))
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8
    # Same image, but BGR
    img_flipped = imread(os.path.join(exdir, 'lena.jpg'), flip_channels=True)
    for c in range(img.shape[2]):
        complement = 2-c if c < 3 else 3
        assert np.all(img[:, :, c] == img_flipped[:, :, complement])
    # Same image, but monochrome
    img = imread(os.path.join(exdir, 'lena.jpg'), mode='L')
    assert img.ndim == 2 or img.shape[2] == 1
    assert img.dtype == np.uint8
    # Load 8-bit single-channel PNG
    img = imread(os.path.join(exdir, 'peaks.png'))
    assert img.ndim == 2 or img.shape[2] == 1
    assert img.dtype == np.uint8
    # ... now enforce loading it as RGB/BGR
    img = imread(os.path.join(exdir, 'peaks.png'), mode='RGB')
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8
    # ... all channels should contain the same information
    for c in [1, 2]:
        assert np.all(img[:, :, c] == img[:, :, 0])
    # Load 16-bit PNG
    img = imread(os.path.join(exdir, 'depth.png'))
    assert img.ndim == 2 or img.shape[2] == 1
    assert img.dtype == np.int32


def test_apply_on_bboxes():
    # Single and multi-channel test data
    x1 = np.zeros((5, 5), dtype=np.uint8)
    x3 = np.zeros((5, 5, 3), dtype=np.uint8)
    # Example boxes
    boxes = [
        (0, 0, 1, 2),
        (5, 5, 3, 2),   # outside image
        (2, 3, 10, 9),  # should be clipped
        (3, 0, 0, 0),   # invalid
        (3, 1, 5, 1)    # should be clipped
    ]
    # Expected results
    e255 = x3.copy()
    e255[0:2, 0, :] = 255
    e255[3:, 2:, :] = 255
    e255[1, 3:, :] = 255
    e42 = x3.copy()
    e42[0:2, 0, :] = 42
    e42[3:, 2:, :] = 42
    e42[1, 3:, :] = 42

    # Exemplary functions
    def _set(img, value):
        img[:] = value
        return img

    def _set255(img):
        return _set(img, 255)

    # No kwargs:
    r1 = apply_on_bboxes(x1, boxes, _set255)
    assert np.all(r1 == e255[:, :, 0])
    r3 = apply_on_bboxes(x3, boxes, _set255)
    assert np.all(r3 == e255)
    # With kwargs
    r1 = apply_on_bboxes(x1, boxes, _set, value=42)
    assert np.all(r1 == e42[:, :, 0])
    r3 = apply_on_bboxes(x3, boxes, _set, value=42)
    assert np.all(r3 == e42)

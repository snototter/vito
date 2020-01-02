import numpy as np
import os
from ..imutils import flip_layers, imread


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
    assert imread('') is None
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

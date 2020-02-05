import numpy as np
import os
import pytest
from vito.imutils import flip_layers, imread, imsave, apply_on_bboxes, \
    ndarray2memory_file, memory_file2ndarray, roi, pad, rgb2gray
from vito.pyutils import safe_shell_output


def test_flip_layers():
    # Invalid input gracefully handled
    assert flip_layers(None) is None

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
    exdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples')
    assert imread(None) is None
    for fn in ['', 'a-non-existing.file']:
        with pytest.raises(FileNotFoundError):
            imread(fn)
    # Forget the keyword to be passed on to PIL
    with pytest.raises(ValueError):
        imread(os.path.join(exdir, 'flamingo.jpg'), 'RGB')
    # Load RGB JPG
    img = imread(os.path.join(exdir, 'flamingo.jpg'))
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8
    # Same image, but BGR
    img_flipped = imread(os.path.join(exdir, 'flamingo.jpg'), flip_channels=True)
    for c in range(img.shape[2]):
        complement = 2-c if c < 3 else 3
        assert np.all(img[:, :, c] == img_flipped[:, :, complement])
    # Same image, but monochrome
    img = imread(os.path.join(exdir, 'flamingo.jpg'), mode='L')
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
    # Load 16-bit PNG, specifying mode manually:
    img = imread(os.path.join(exdir, 'depth.png'), mode='I')
    assert img.ndim == 2 or img.shape[2] == 1
    assert img.dtype == np.int32
    # Load 1-bit PNG (requires specifying the mode!)
    img = imread(os.path.join(exdir, 'space-invader.png'), mode='L')
    assert img.ndim == 2 or img.shape[2] == 1
    assert img.dtype == np.uint8


def test_imsave(tmp_path):
    # This test will only work on Unix-based test environments because we
    # use the 'file' command to ensure the stored file is correct.
    exdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples')
    out_fn = str(tmp_path / 'test.png')
    ##########################################################################
    # Test RGB
    img_in = imread(os.path.join(exdir, 'flamingo.jpg'))
    assert img_in.ndim == 3 and img_in.shape[2] == 3
    assert img_in.dtype == np.uint8
    # Save (lossless) and reload
    imsave(out_fn, img_in)
    _, finfo = safe_shell_output('file', out_fn)
    assert finfo.split(':')[1].strip() == 'PNG image data, 400 x 400, 8-bit/color RGB, non-interlaced'
    img_out = imread(out_fn)
    assert img_out.ndim == 3 and img_out.shape[2] == 3
    assert img_out.dtype == np.uint8
    assert np.all(img_in[:] == img_out[:])
    ##########################################################################
    # Test RGB with flipping channels
    img_in = imread(os.path.join(exdir, 'flamingo.jpg'))
    assert img_in.ndim == 3 and img_in.shape[2] == 3
    assert img_in.dtype == np.uint8
    # Save (lossless) and reload
    imsave(out_fn, img_in, flip_channels=True)
    _, finfo = safe_shell_output('file', out_fn)
    assert finfo.split(':')[1].strip() == 'PNG image data, 400 x 400, 8-bit/color RGB, non-interlaced'
    img_out = imread(out_fn)
    assert img_out.ndim == 3 and img_out.shape[2] == 3
    assert img_out.dtype == np.uint8
    for c in range(3):
        assert np.all(img_in[:, :, c] == img_out[:, :, 2-c])
    ##########################################################################
    # Test monochrome 8 bit
    img_in = imread(os.path.join(exdir, 'peaks.png'))
    assert img_in.ndim == 2 or img_in.shape[2] == 1
    assert img_in.dtype == np.uint8
    # Save (lossless) and reload
    imsave(out_fn, img_in)
    _, finfo = safe_shell_output('file', out_fn)
    assert finfo.split(':')[1].strip() == 'PNG image data, 256 x 256, 8-bit grayscale, non-interlaced'
    img_out = imread(out_fn)
    assert img_out.ndim == 2 or img_out.shape[2] == 1
    assert img_out.dtype == np.uint8
    assert np.all(img_in[:] == img_out[:])
    ##########################################################################
    # Test monochrome 16 bit (will be loaded as np.int32, using PIL's 'I' mode)
    img_in = imread(os.path.join(exdir, 'depth.png'))
    assert img_in.ndim == 2 or img_in.shape[2] == 1
    assert img_in.dtype == np.int32
    # Explicitly cast to uint16
    img_in = img_in.astype(np.uint16)
    # Save (lossless) and reload
    imsave(out_fn, img_in)
    _, finfo = safe_shell_output('file', out_fn)
    assert finfo.split(':')[1].strip() == 'PNG image data, 800 x 600, 16-bit grayscale, non-interlaced'
    img_out = imread(out_fn)
    assert img_out.ndim == 2 or img_out.shape[2] == 1
    # Loading, however, will still produce a int32 image.
    assert img_out.dtype == np.int32
    assert np.all(img_in[:] == img_out[:])
    ##########################################################################
    # Test 1-bit PNG (requires specifying the mode!)
    img_in = imread(os.path.join(exdir, 'space-invader.png'), mode='L').astype(np.bool)
    assert img_in.ndim == 2 or img_in.shape[2] == 1
    assert img_in.dtype == np.bool
    imsave(out_fn, img_in)
    _, finfo = safe_shell_output('file', out_fn)
    assert (finfo.split(':')[1].strip() == 'PNG image data, 200 x 149, 1-bit colormap, non-interlaced' or
        finfo.split(':')[1].strip() == 'PNG image data, 200 x 149, 1-bit grayscale, non-interlaced')
    img_out = imread(out_fn, mode='L').astype(np.bool)
    assert img_out.ndim == 2 or img_out.shape[2] == 1
    assert np.all(img_in[:] == img_out[:])


def test_apply_on_bboxes():
    # Single and multi-channel test data
    x1 = np.zeros((5, 5), dtype=np.uint8)
    x2 = np.zeros((5, 5, 1), dtype=np.int32)
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

    assert apply_on_bboxes(None, boxes, _set255) is None

    # No kwargs:
    r1 = apply_on_bboxes(x1, boxes, _set255).copy()
    assert np.all(r1 == e255[:, :, 0])
    assert r1.dtype == np.uint8

    r2 = apply_on_bboxes(x2, boxes, _set255)
    e = e255[:, :, 0]
    assert np.all(r2.reshape(e.shape) == e)
    assert r2.dtype == np.int32

    r3 = apply_on_bboxes(x3, boxes, _set255)
    assert np.all(r3 == e255)
    assert r3.dtype == np.uint8

    # With kwargs
    r1 = apply_on_bboxes(x1, boxes, _set, value=42)
    assert np.all(r1 == e42[:, :, 0])
    assert r1.dtype == np.uint8

    r2 = apply_on_bboxes(x2, boxes, _set, value=42)
    assert np.all(r2 == e42)
    assert r2.dtype == np.int32

    r3 = apply_on_bboxes(x3, boxes, _set, value=42)
    assert np.all(r3 == e42)
    assert r3.dtype == np.uint8


def test_np2mem():
    assert memory_file2ndarray(None) is None
    assert ndarray2memory_file(None) is None

    shapes = [(20, 128), (32, 64, 3), (64, 32, 3)]
    for s in shapes:
        if isinstance(s, tuple):
            x = (255.0 * np.random.rand(*s)).astype(np.uint8)
        else:
            x = (255.0 * np.random.rand(s)).astype(np.uint8)
        mfpng = ndarray2memory_file(x, format='png')
        y = memory_file2ndarray(mfpng)
        assert np.all(x[:] == y[:])
        # Encode as JPG, this should decrease quality - at least we're not
        # able to decode the original input data
        mfjpg = ndarray2memory_file(x, format='jpeg')
        y = memory_file2ndarray(mfjpg)
        assert not np.all(x[:] == y[:])
    # PIL assumes a 2D image and performs no checking:
    with pytest.raises(IndexError):
        _ = ndarray2memory_file(np.random.rand(3).astype(np.uint8))


def test_roi():
    assert roi(np.zeros((3, 3)), None) is None
    assert roi(np.zeros((3, 3)), [1, 2, 3, None]) is None
    assert roi(None, [1, 2, 3, 4]) is None
    invalid = roi(np.zeros((3, 3), dtype=np.uint8), [1, 1, 0, 1])
    assert invalid.shape[0] == 1 and invalid.shape[1] == 0
    invalid = roi(np.zeros((3, 3), dtype=np.uint8), [1, 1, 1, 0])
    assert invalid.shape[0] == 0 and invalid.shape[1] == 1

    # Proper clipping of corners:
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    clipped = roi(x, [-1, -3, 2, 4])
    assert clipped.shape[0] == 1 and clipped.shape[1] == 1
    assert clipped[0, 0] == x[0, 0]
    clipped = roi(x, [3, -2, 2, 3])
    assert clipped.shape[0] == 1 and clipped.shape[1] == 1
    assert clipped[0, 0] == x[0, 2]

    rects = [
        (2, 5, 1, 2),
        (20, 20, 100, 100),
        (-5, 2, 1, 2)
    ]
    for shape in [(10, 10), (10, 10, 2)]:
        x = np.random.rand(*shape)
        for rect in rects:
            roi_ = roi(x, rect)
            l, t, w, h = rect
            r, b = l + w, t + h
            l = max(0, min(x.shape[1]-1, l))
            r = max(0, min(x.shape[1], r))
            t = max(0, min(x.shape[0]-1, t))
            b = max(0, min(x.shape[0], b))
            if x.ndim == 2:
                expected = x[t:b, l:r]
            else:
                expected = x[t:b, l:r, :]
            assert np.all(expected[:] == roi_[:])


def test_pad():
    assert pad(None, 3) is None
    data = np.random.randint(0, 255, (2, 3))
    with pytest.raises(ValueError):
        pad(data, 0)
    with pytest.raises(ValueError):
        pad(data, -17)
    padded = pad(data, 1, color=None)
    assert padded.shape[0] == data.shape[0] + 2
    assert padded.shape[1] == data.shape[1] + 2
    assert padded.ndim == 3 and padded.shape[2] == 4
    assert np.all(padded[:, 0] == 0) and np.all(padded[0, :] == 0)
    assert np.all(padded[1, 1:-1, 0] == data[0, :])
    with pytest.raises(RuntimeError):
        pad(np.dstack((data, data)), 1, None)
    # Provide valid color as scalar and tuple
    padded = pad(data, 2, color=3)
    assert padded.shape[0] == data.shape[0] + 4
    assert padded.shape[1] == data.shape[1] + 4
    assert padded.ndim == data.ndim or (padded.ndim == 3 and padded.shape[2] == 1)
    assert np.all(padded[:, 0] == 3) and np.all(padded[0, :] == 3)
    assert np.all(padded[2:-2, 2:-2, 0] == data)
    # ... tuple
    padded = pad(data, 8, color=(200, 200, 200))
    assert padded.shape[0] == data.shape[0] + 16
    assert padded.shape[1] == data.shape[1] + 16
    assert padded.ndim == data.ndim or (padded.ndim == 3 and padded.shape[2] == 1)
    assert np.all(padded[:, 0] == 200) and np.all(padded[0, :] == 200)
    assert np.all(padded[8:-8, 8:-8, 0] == data)
    # Multi-channel image
    data = np.random.randint(0, 255, (2, 3, 3))
    color = np.random.randint(0, 255, (3,))
    padded = pad(data, 1, color=color)
    assert padded.shape[0] == data.shape[0] + 2
    assert padded.shape[1] == data.shape[1] + 2
    assert padded.ndim == data.ndim
    for c in range(3):
        assert np.all(padded[:, 0, c] == color[c]) and np.all(padded[0, :, c] == color[c])
        assert np.all(padded[1:-1, 1:-1, c] == data[:, :, c])
    # ... 4-channel
    data = np.random.randint(0, 255, (2, 3, 4))
    color = np.random.randint(0, 255, (4,))
    padded = pad(data, 1, color=color)
    assert padded.shape[0] == data.shape[0] + 2
    assert padded.shape[1] == data.shape[1] + 2
    assert padded.ndim == data.ndim and padded.shape[2] == 4
    for c in range(3):
        assert np.all(padded[:, 0, c] == color[c]) and np.all(padded[0, :, c] == color[c])
        assert np.all(padded[1:-1, 1:-1, c] == data[:, :, c])


def test_rgb2gray():
    assert rgb2gray(None) is None
    x = np.zeros((2, 3, 4), dtype=np.uint8)
    frgb = [0.2989, 0.5870, 0.1140]
    fbgr = frgb[::-1]
    for c in range(3):
        y = x.copy()
        y[:, :, c] = 255
        g = rgb2gray(y, False)
        assert g.ndim == 2 or (g.ndim == 3 and g.shape[2] == 1)
        assert np.all(g[:] == np.uint8(frgb[c]*255.0))
        g = rgb2gray(y, True)
        assert np.all(g[:] == np.uint8(fbgr[c]*255.0))

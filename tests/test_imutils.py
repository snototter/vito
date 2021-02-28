import numpy as np
import os
import pytest
from vito.imutils import flip_layers, imread, imsave, apply_on_bboxes, \
    ndarray2memory_file, memory_file2ndarray, roi, crop, pad, rgb2gray, \
    grayscale, pixelate, gaussian_blur, set_to, ensure_c3, concat, \
    rotate90, rotate180, rotate270, noop
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
    img_in = imread(os.path.join(exdir, 'space-invader.png'), mode='L').astype(bool)
    assert img_in.ndim == 2 or img_in.shape[2] == 1
    assert img_in.dtype == bool
    imsave(out_fn, img_in)
    _, finfo = safe_shell_output('file', out_fn)
    assert (finfo.split(':')[1].strip() == 'PNG image data, 200 x 149, 1-bit colormap, non-interlaced' or
            finfo.split(':')[1].strip() == 'PNG image data, 200 x 149, 1-bit grayscale, non-interlaced')
    img_out = imread(out_fn, mode='L').astype(bool)
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
    r1 = apply_on_bboxes(x1, boxes, _set, 42)
    assert np.all(r1 == e42[:, :, 0])
    assert r1.dtype == np.uint8

    r2 = apply_on_bboxes(x2, boxes, _set, 42)
    assert np.all(r2 == e42)
    assert r2.dtype == np.int32

    r3 = apply_on_bboxes(x3, boxes, _set, 42)
    assert np.all(r3 == e42)
    assert r3.dtype == np.uint8


def test_np2mem():
    assert memory_file2ndarray(None) is None
    assert ndarray2memory_file(None) is None

    shapes = [(20, 128), (32, 64, 3), (64, 32, 3)]
    for s in shapes:
        x = (255.0 * np.random.randint(0, 255, s)).astype(np.uint8)
        mfpng = ndarray2memory_file(x, format='png')
        y = memory_file2ndarray(mfpng)
        assert np.array_equal(x, y)
        # Encode as JPG, this should decrease quality - at least we're not
        # able to decode the original input data
        mfjpg = ndarray2memory_file(x, format='jpeg')
        y = memory_file2ndarray(mfjpg)
        assert not np.array_equal(x, y)
    # Newer PIL versions can also work with 1D data:
    x = np.random.randint(0, 255, 3).astype(np.uint8)
    mf1d = ndarray2memory_file(x)
    y = memory_file2ndarray(mf1d)
    assert y.shape[0] == 3 and y.shape[1] == 1
    assert y[0, 0] == x[0] and y[1, 0] == x[1] and y[2, 0] == x[2]


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
    # Ensure crop() is an alias to roi()
    assert crop == roi


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
    # Single channel input v1
    x = np.random.randint(0, 255, size=(17, 23), dtype=np.uint8)
    g = rgb2gray(x)
    assert np.array_equal(x, g)
    # Single channel input v2
    x = np.random.randint(0, 255, size=(17, 23, 1), dtype=np.uint8)
    g = rgb2gray(x)
    assert np.array_equal(x, g)
    # Single channel float
    x = np.random.rand(17, 23).astype(np.float32)
    g = rgb2gray(x)
    assert np.array_equal(x, g)
    # Cannot convert dual-channel to gray:
    x = np.random.randint(0, 255, size=(15, 10, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        g = rgb2gray(x)
    # RGB(A) input (uint8 and float32)
    inputs = [np.zeros((2, 3, 3), dtype=np.uint8),
              np.zeros((5, 17, 4), dtype=np.uint8),
              np.zeros((23, 5, 3), dtype=np.float32),
              np.zeros((18, 12, 4), dtype=np.float32)]
    for x, scalar, cast in zip(inputs, [255.0, 255.0, 1.0, 1.0], [np.uint8, np.uint8, np.float32, np.float32]):
        frgb = [0.2989, 0.5870, 0.1140]
        fbgr = frgb[::-1]
        for c in range(3):
            y = x.copy()
            y[:, :, c] = scalar
            g = rgb2gray(y, False)
            assert g.ndim == 2 or (g.ndim == 3 and g.shape[2] == 1)
            assert np.all(g[:] == cast(frgb[c]*scalar))
            g = rgb2gray(y, True)
            assert np.all(g[:] == cast(fbgr[c]*scalar))
    # Ensure grayscale is an alias to rgb2gray
    assert grayscale == rgb2gray


def test_pixelate():
    assert pixelate(None) is None
    with pytest.raises(ValueError):
        pixelate(np.zeros((17, 12)), 0, -1)
    x = np.random.randint(0, 255, (3, 10), dtype=np.uint8)
    res = pixelate(x)
    assert res.shape == x.shape
    assert np.all(res[:, :5] == res[0, 0])
    assert np.all(res[:, 5:] == res[-1, -1])

    x = np.random.randint(0, 255, (3, 12), dtype=np.uint8)
    res = pixelate(x)
    assert res.shape == x.shape
    # If the image size is not a multiple of the block size,
    # the blocks will be slightly larger (to avoid having small
    # -- recognizable -- blocks at the border)
    assert np.all(res[:, :6] == res[0, 0])
    assert np.all(res[:, 6:] == res[-1, -1])


def test_gaussian_blur():
    assert gaussian_blur(None) is None
    exdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples')
    img = imread(os.path.join(exdir, 'flamingo.jpg'))
    res1 = gaussian_blur(img, 3)
    res2 = gaussian_blur(img, 15)
    assert res1.shape == img.shape
    assert res2.shape == img.shape
    # We use PIL's ImageFilter module, so just resort to a simple sanity
    # check, i.e. whether blurring the image with a larger kernel degrades
    # its visual quality more than using a smaller kernel.
    diff1 = np.sqrt(np.sum((img - res1)**2))
    diff2 = np.sqrt(np.sum((img - res2)**2))
    assert diff1 > 0
    assert diff2 > diff1


def test_set_to():
    assert set_to(None, 3) is None
    assert set_to(np.zeros((2, 3)), None) is None

    # Test scalar
    res = set_to(np.zeros((2,)), 13)
    assert np.all(res[:] == 13) and res.shape == (2,)
    res = set_to(np.zeros((17, 12), dtype=np.uint8), 200)
    assert np.all(res[:] == 200) and res.shape == (17, 12)
    res = set_to(np.zeros((17, 12, 4), dtype=np.int32), -123)
    assert np.all(res[:] == -123) and res.shape == (17, 12, 4)

    # Test tuple
    with pytest.raises(ValueError):
        set_to(np.zeros((2, 3)), ())
    res = set_to(np.zeros((2,)), (13, 74, 99))
    assert np.all(res[:] == 13) and res.shape == (2,)
    res = set_to(np.zeros((3, 5), dtype=np.uint8), (78,))
    assert np.all(res[:] == 78) and res.shape == (3, 5)
    with pytest.raises(ValueError):
        set_to(np.zeros((17, 12, 4), dtype=np.int32), (1, 2, 3))
    val = (-123, 77, -23, 2**10)
    res = set_to(np.zeros((5, 6, 4), dtype=np.int32), val)
    assert res.shape == (5, 6, 4)
    for ch in range(res.shape[2]):
        assert np.all(res[:, :, ch] == val[ch])


def test_ensure_c3():
    # Invalid inputs
    assert ensure_c3(None) is None
    for invalid in [np.zeros(17), np.ones((4, 3, 2)), np.zeros((2, 2, 5))]:
        with pytest.raises(ValueError):
            ensure_c3(invalid)
    # Grayscale image (2-dim)
    x = np.random.randint(0, 255, (20, 30))
    c3 = ensure_c3(x)
    assert c3.ndim == 3 and c3.shape[2] == 3
    for c in range(3):
        assert np.array_equal(x, c3[:, :, c])
    # Grayscale image (3-dim, 1-channel)
    x = np.random.randint(0, 255, (10, 5, 1))
    c3 = ensure_c3(x)
    assert c3.ndim == 3 and c3.shape[2] == 3
    for c in range(3):
        assert np.array_equal(x[:, :, 0], c3[:, :, c])
    # RGB(A) inputs
    for x in [np.random.randint(0, 255, (12, 23, 3)), np.random.randint(0, 255, (12, 23, 4))]:
        c3 = ensure_c3(x)
        assert c3.ndim == 3 and c3.shape[2] == 3
        assert np.array_equal(x[:, :, :3], c3)


def test_concat():
    # Invalid inputs
    x = np.zeros((3,5))
    y = np.ones((5,7))
    assert concat(None, y, True) is None
    assert concat(x, None, False) is None
    with pytest.raises(ValueError):
        concat(x, y, True)
    with pytest.raises(ValueError):
        concat(x, y, False)
    
    # ndim == 2, different dtype
    x = np.random.randint(0, 255, (5, 17), dtype=np.uint8)
    y = np.random.randint(0, 255, (5, 2), dtype=np.int32)
    c = concat(x, y, True)
    # The concatenation should have the "better" dtype of the two inputs
    assert c.dtype == y.dtype
    assert np.array_equal(x.astype(c.dtype), c[:, :x.shape[1]])
    assert np.array_equal(y, c[:, x.shape[1]:])

    # ndim == 2, same dtype
    y = np.random.randint(0, 255, (5, 2), dtype=x.dtype)
    c = concat(x, y, True)
    assert c.dtype == x.dtype
    assert np.array_equal(x, c[:, :x.shape[1]])
    assert np.array_equal(y, c[:, x.shape[1]:])

    # ndim == 2 vs ndim == 3, single channel
    x = np.random.randint(0, 255, (5, 2, 1), dtype=np.uint8)
    y = np.random.randint(0, 255, (3, 2), dtype=np.int32)
    c = concat(x, y, False)
    assert c.dtype == y.dtype
    assert c.ndim == 3 and c.shape[2] == 1
    assert np.array_equal(x.astype(c.dtype), c[:x.shape[0], :, :])
    assert np.array_equal(y[:, :], c[x.shape[0]:, :, 0])

    # ndim == 2 vs ndim == 3, single vs multi-channel (compatible)
    x = np.random.randint(0, 255, (5, 2), dtype=np.int32)
    y = np.random.randint(0, 255, (3, 2, 4), dtype=np.uint8)
    c = concat(x, y, False)
    assert c.dtype == x.dtype
    assert c.ndim == 3 and c.shape[2] == y.shape[2]
    for l in range(c.shape[2]):
        assert np.array_equal(x[:, :], c[:x.shape[0], :, l])
    assert np.array_equal(y[:, :, :], c[x.shape[0]:, :, :])

    # ndim == 3, multi-channel
    # --> Incompatible inputs
    x = np.random.randint(0, 255, (5, 2, 2), dtype=np.uint8)
    y = np.random.randint(0, 255, (3, 2, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        concat(x, y, False)
    # --> same number of channels
    x = np.random.randint(0, 255, (6, 2, 3), dtype=np.uint64)
    y = np.random.randint(0, 255, (6, 9, 3), dtype=np.int32)
    c = concat(x, y, True)
    assert c.dtype == x.dtype
    assert np.array_equal(x, c[:, :x.shape[1], :])
    assert np.array_equal(y, c[:, x.shape[1]:, :])
    # --> different number of channels (but compatible)
    x = np.random.randint(0, 255, (6, 2, 1), dtype=np.uint64)
    y = np.random.randint(0, 255, (6, 9, 3), dtype=np.int32)
    c = concat(x, y, True)
    assert c.dtype == x.dtype
    for l in range(3):
        assert np.array_equal(x[:, :, 0], c[:, :x.shape[1], l])
    assert np.array_equal(y.astype(c.dtype), c[:, x.shape[1]:])
    # ... switch parameter order
    c = concat(y, x, True)
    assert c.dtype == x.dtype
    assert np.array_equal(y.astype(c.dtype), c[:, :y.shape[1]])
    for l in range(3):
        assert np.array_equal(x[:, :, 0], c[:, y.shape[1]:, l])


def test_noop():
    for x in [None, 1, 3.7, np.zeros((3, 2)), 'test']:
        if isinstance(x, np.ndarray):
            assert np.array_equal(noop(x), x)
        else:
            assert noop(x) == x


def test_rotation():
    # Invalid inputs
    assert rotate90(None) is None
    assert rotate180(None) is None
    assert rotate270(None) is None

    for x in [np.array([[1, 2], [3, 4]], dtype=np.int32),
              np.random.randint(0, 255, (2, 2, 1), dtype=np.uint8),
              np.random.randint(0, 255, (2, 2, 3), dtype=np.uint8),
              np.random.randint(0, 255, (2, 2, 3), dtype=np.int64)]:
        if x.ndim == 2:
            y = rotate90(x)
            assert y[0, 0] == x[0, 1] and \
                y[0, 1] == x[1, 1] and \
                y[1, 0] == x[0, 0] and \
                y[1, 1] == x[1, 0]

            y = rotate180(x)
            assert y[0, 0] == x[1, 1] and \
                y[0, 1] == x[1, 0] and \
                y[1, 0] == x[0, 1] and \
                y[1, 1] == x[0, 0]

            y = rotate270(x)
            assert y[0, 0] == x[1, 0] and \
                y[0, 1] == x[0, 0] and \
                y[1, 0] == x[1, 1] and \
                y[1, 1] == x[0, 1]
        else:
            for l in range(x.shape[2]):
                y = rotate90(x)
                assert y[0, 0, l] == x[0, 1, l] and \
                    y[0, 1, l] == x[1, 1, l] and \
                    y[1, 0, l] == x[0, 0, l] and \
                    y[1, 1, l] == x[1, 0, l]

                y = rotate180(x)
                assert y[0, 0, l] == x[1, 1, l] and \
                    y[0, 1, l] == x[1, 0, l] and \
                    y[1, 0, l] == x[0, 1, l] and \
                    y[1, 1, l] == x[0, 0, l]

                y = rotate270(x)
                assert y[0, 0, l] == x[1, 0, l] and \
                    y[0, 1, l] == x[0, 0, l] and \
                    y[1, 0, l] == x[1, 1, l] and \
                    y[1, 1, l] == x[0, 1, l]

import numpy as np
import pytest
from vito.imvis import pseudocolor, color_by_id, exemplary_colors, overlay
from vito import colormaps


def assert_color_equal(a, b, flip=False):
    for c in range(3):
        if flip:
            assert a[c] == b[2-c]
        else:
            assert a[c] == b[c]


def test_pseudocolor():
    data = np.array([[0, 1, 2], [255, 42, 0]], dtype=np.uint8)
    for cm in [colormaps.colormap_parula_rgb, colormaps.colormap_jet_rgb,
            colormaps.colormap_magma_rgb, colormaps.colormap_gray]:
        pc = pseudocolor(data, limits=None, color_map=cm)
        assert np.all(pc[0, 0, :] == pc[1, 2, :])
        assert_color_equal(pc[0, 1, :], cm[1])
        assert_color_equal(pc[0, 2, :], cm[2])
        assert_color_equal(pc[1, 0, :], cm[-1])
        assert_color_equal(pc[1, 1, :], cm[42])
        assert_color_equal(pc[1, 2, :], cm[0])

        pc = pseudocolor(data, limits=[0, 0], color_map=cm)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                assert_color_equal(pc[r, c, :], cm[0])

    data = np.random.rand(3, 7, 2)
    with pytest.raises(ValueError):
        _ = pseudocolor(data)

    data = np.zeros((2, 3, 1), dtype=np.uint8)
    data[0, 0] = 20
    data[1, 2] = 10
    pc = pseudocolor(data, limits=None)
    cm = colormaps.colormap_parula_rgb
    assert_color_equal(pc[0, 0, :], cm[-1])
    assert_color_equal(pc[1, 0, :], cm[0])
    assert_color_equal(pc[1, 2, :], cm[127])

    # Test clipping
    pc = pseudocolor(data, limits=[2, 4])
    assert_color_equal(pc[0, 0, :], cm[-1])
    assert_color_equal(pc[1, 0, :], cm[0])
    assert_color_equal(pc[1, 2, :], cm[-1])

    pc = pseudocolor(data, limits=[30, 40])
    assert_color_equal(pc[0, 0, :], cm[0])
    assert_color_equal(pc[1, 0, :], cm[0])
    assert_color_equal(pc[1, 2, :], cm[0])

    # Test clipping by providing only a single value
    pc = pseudocolor(data, limits=[None, 4])
    assert_color_equal(pc[0, 0, :], cm[-1])
    assert_color_equal(pc[1, 0, :], cm[0])
    assert_color_equal(pc[1, 2, :], cm[-1])

    pc = pseudocolor(data, limits=[20, None])
    assert_color_equal(pc[0, 0, :], cm[0])
    assert_color_equal(pc[1, 0, :], cm[0])
    assert_color_equal(pc[1, 2, :], cm[0])


def test_colormap_by_name():
    cm = colormaps.by_name('HSV')
    assert cm == colormaps.colormap_hsv_rgb
    cm = colormaps.by_name('HSV', return_rgb=False)
    assert cm == colormaps.colormap_hsv_bgr

    cm = colormaps.by_name('turbo')
    assert cm == colormaps.colormap_turbo_rgb
    cm = colormaps.by_name('TURBO', return_rgb=False)
    assert cm == colormaps.colormap_turbo_bgr

    cm = colormaps.by_name('gray')
    assert cm == colormaps.colormap_gray
    cm = colormaps.by_name('grayscale', return_rgb=False)
    assert cm == colormaps.colormap_gray
    cm = colormaps.by_name('grey')
    assert cm == colormaps.colormap_gray
    cm = colormaps.by_name('greyscale', return_rgb=False)
    assert cm == colormaps.colormap_gray

    with pytest.raises(KeyError):
        colormaps.by_name('foo')


def test_color_by_id():
    assert_color_equal(color_by_id(0), exemplary_colors[0])
    nc = len(exemplary_colors)
    assert_color_equal(color_by_id(nc), exemplary_colors[0])
    assert_color_equal(color_by_id(nc-1), exemplary_colors[-1])
    assert_color_equal(color_by_id(-1), exemplary_colors[-1])
    assert_color_equal(color_by_id(-3), exemplary_colors[-3])
    assert_color_equal(color_by_id(nc-1, flip_channels=True), exemplary_colors[-1], flip=True)
    assert_color_equal(color_by_id(-3, flip_channels=True), exemplary_colors[-3], flip=True)


def test_overlay():
    img1 = np.zeros((3, 3), dtype=np.uint8)
    img2 = 255 * np.ones((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        _ = overlay(img1, img2, -1)
    with pytest.raises(ValueError):
        _ = overlay(img1, img2, 1.1)

    # Invalid/Incompatible layers
    with pytest.raises(ValueError):
        _ = overlay(np.zeros((3, 3, 2)), np.zeros((3, 3, 1)), 0)

    # Overlay same channels (but not 1- or 3-channel)
    out = overlay(np.zeros((3, 3, 2)), np.ones((3, 3, 2)), 0.5)
    assert np.all(out[:] == pytest.approx(0.5))
    assert out.ndim == 3 and out.shape[2] == 2

    # Overlay gray on gray
    out = overlay(img1, img2, 1)
    assert np.all(img1[:] == out[:])
    out = overlay(img1, img2, 0)
    assert np.all(img2[:] == out[:])

    # Overlay gray on color
    img1 = np.zeros((3, 3, 3), dtype=np.uint8)
    out = overlay(img1, img2, 1)
    assert np.all(img1[:] == out[:])
    out = overlay(img2, img1, 1)
    assert np.all(img2[:] == out[:])
    img2 = 255 * np.ones((3, 3, 1), dtype=np.uint8)
    out = overlay(img1, img2, 0)
    assert np.all(img2[:] == out[:])
    out = overlay(img2, img1, 1)
    assert np.all(img2[:] == out[:])

    # Overlay color on color
    img2 = 255 * np.ones((3, 3, 3), dtype=np.uint8)
    out = overlay(img1, img2, 0)
    assert np.all(img2[:] == out[:])
    out = overlay(img1, img2, 0.8)
    assert np.all(50 == out[:])
    out = overlay(img1, img2, 0.2)
    assert np.all(204 == out[:])

    # Test different data type combination
    data_types = [np.uint8, np.float32, np.float64]
    expected_vals = {
        np.uint8: {
            np.uint8: 152,
            np.float32: pytest.approx(101.8),
            np.float64: pytest.approx(101.8)
        },
        np.float32: {
           np.uint8: 50,
           np.float32: pytest.approx(152.6),
           np.float64: pytest.approx(152.6)
        },
        np.float64: {
           np.uint8: 50,
           np.float32: pytest.approx(152.6),
           np.float64: pytest.approx(152.6)
        }
    }
    for dt1 in data_types:
        img1 = 255 * np.ones((10, 10, 3), dtype=dt1)
        for dt2 in data_types:
            img2 = 127 * np.ones((10, 10, 1), dtype=dt2)
            out = overlay(img1, img2, 0.2)
            assert np.all(out[:] == expected_vals[dt1][dt2])
            assert out.dtype == dt2

    # Unsupported dtypes
    with pytest.raises(ValueError):
        _ = overlay(np.zeros((3, 3), dtype=np.int32), img1, 0.7)
    with pytest.raises(ValueError):
        _ = overlay(img1, np.zeros((3, 3), dtype=np.int32), 0.2)

    # Test masking
    with pytest.raises(ValueError):
        _ = overlay(img1, img1, 0.1, np.ones((img1.shape[0]+1, img1.shape[1])))
    for imshape in [(2, 3), (2, 3, 1), (3, 4, 3)]:
        img1 = np.ones(imshape, dtype=np.uint8)
        if img1.ndim > 2:
            img1[:, :, 0] = 50
        img2 = 2 * img1
        for dt in [np.uint8, np.float32]:
            for maskshape in [0, 1]:
                if maskshape == 0:
                    mask = np.zeros((imshape[0], imshape[1]), dtype=dt)
                else:
                    mask = np.zeros((imshape[0], imshape[1], 1), dtype=dt)
                for maskval in [1, 255]:
                    mask[1, 1] = maskval
                    out = overlay(img1, img2, 0.2, mask)
                    expval = np.uint8(
                            255 * (0.2 * (img1[1, 1] / 255.0) + 0.8 * (img2[1, 1] / 255.0)))
                    assert np.all(out[:, 0] == img2[:, 0])
                    assert np.all(out[:, 2] == img2[:, 2])
                    assert np.all(out[0, :] == img2[0, :])
                    assert np.all(out[1, 1] == expval)

import numpy as np
import pytest
from vito.imvis import pseudocolor, color_by_id, exemplary_colors
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

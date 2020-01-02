import numpy as np
from ..imvis import pseudocolor, color_by_id, exemplary_colors
from .. import colormaps


def assert_color_equal(a, b):
    for c in range(3):
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


def test_color_by_id():
    assert_color_equal(color_by_id(0), exemplary_colors[0])
    nc = len(exemplary_colors)
    assert_color_equal(color_by_id(nc), exemplary_colors[0])
    assert_color_equal(color_by_id(nc-1), exemplary_colors[-1])
    assert_color_equal(color_by_id(-1), exemplary_colors[-1])
    assert_color_equal(color_by_id(-3), exemplary_colors[-3])

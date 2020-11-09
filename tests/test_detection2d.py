import pytest
import numpy as np
from vito.detection2d import BoundingBox


def test_bbox_init():
    bb = BoundingBox.from_corner_repr([0, 0, 2, 4])
    assert bb.width == 2
    assert bb.height == 4
    assert bb.left == 0
    assert bb.top == 0

    bb_corner = BoundingBox.from_corner_repr([17, 42, 123, 99])
    assert bb_corner.width == 106
    assert bb_corner.height == 57
    assert bb_corner.left == 17
    assert bb_corner.top == 42

    bb_minmax = BoundingBox.from_minmax_repr([17, 123, 42, 99])
    assert bb_corner == bb_minmax

    bb_centroid = BoundingBox.from_centroid_repr([17+53, 42+28.5, 106, 57])
    assert bb_corner == bb_centroid

    bb_rect = BoundingBox.from_rect_repr(bb_centroid.to_rect_repr())
    assert bb_corner == bb_rect


def test_bbox_conversion():
    bb = BoundingBox.from_rect_repr(np.random.randint(0, 1e6, 4))
    
    assert BoundingBox.from_rect_repr(bb.to_rect_repr()) == bb
    assert BoundingBox.from_corner_repr(bb.to_corner_repr()) == bb
    assert BoundingBox.from_centroid_repr(bb.to_centroid_repr()) == bb
    assert BoundingBox.from_minmax_repr(bb.to_minmax_repr()) == bb


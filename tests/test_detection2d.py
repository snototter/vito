import pytest
import numpy as np
from vito.detection2d import BoundingBox, iou


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


def test_bbox_iou():
    bb = BoundingBox.from_rect_repr(np.random.randint(0, 1e6, 4))
    # Ensure w/h > 0
    bb.width += 1
    bb.height += 1
    # Input checks
    assert iou(None, bb) == 0.0
    assert iou(bb, None) == 0.0
    assert iou(None, None) == 0.0
    assert bb.iou(None) == 0.0
    assert bb.iou(bb) == iou(bb, bb)
    assert bb.iou(bb) == 1.0
    bb1 = BoundingBox.from_rect_repr([-10, -5, 7, 3])
    assert bb.iou(bb1) == 0.0
    assert bb1.iou(bb) == 0.0
    assert iou(bb1, bb) == 0.0
    bb2 = BoundingBox.from_rect_repr([-3, -3, 1, 2])
    assert bb1.iou(bb2) == 0.0

    bb1 = BoundingBox.from_rect_repr([10, 20, 30, 40])
    bb2 = BoundingBox.from_rect_repr([25, 20, 15, 40])
    assert bb1.iou(bb2) == 0.5
    bb1.width = 45
    assert bb1.iou(bb2) == 1/3


def test_bbox_area():
    bb = BoundingBox.from_rect_repr(np.random.randint(0, 1e6, 4))
    assert bb.area() == bb.width * bb.height
    bb.width = 0
    bb.height = 1
    assert bb.area() == 0

import pytest
import numpy as np
from vito.detection2d import BoundingBox, Detection, Size, iou, filter_detection_classes,\
    LabelMapVOC07, LabelMapCOCO, LabelMap


def test_size():
    sz1 = Size.from_hw(60, 30)
    sz2 = Size.from_wh(30, 60)
    assert sz1 == sz2
    assert sz1.area() == 1800
    assert sz1.width == 30
    assert sz1.height == 60


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


def bb_approx_equal(a, b):
    """Test if two bounding boxes are approximately equal (up to rounding precision)."""
    assert a.left == pytest.approx(b.left)
    assert a.top == pytest.approx(b.top)
    assert a.width == pytest.approx(b.width)
    assert a.height == pytest.approx(b.height)


def test_bbox_conversion():
    bb = BoundingBox.from_rect_repr(np.random.randint(0, 1e6, 4))
    assert BoundingBox.from_rect_repr(bb.to_rect_repr()) == bb
    assert BoundingBox.from_corner_repr(bb.to_corner_repr()) == bb
    assert BoundingBox.from_centroid_repr(bb.to_centroid_repr()) == bb
    assert BoundingBox.from_minmax_repr(bb.to_minmax_repr()) == bb

    sz = Size(bb.width, bb.height)
    bb_rel = bb.to_rect_repr(sz)
    assert bb_rel[2] == 1
    assert bb_rel[3] == 1

    bb = BoundingBox.from_rect_repr([10, 70, 50, 140])
    bb_rel = bb.to_rect_repr(Size(100, 210))
    assert bb_rel[0] == 0.1
    assert bb_rel[1] == pytest.approx(1/3)
    assert bb_rel[2] == 0.5
    assert bb_rel[3] == pytest.approx(2/3)

    bb_rel = bb.to_corner_repr(Size(200, 210))
    assert bb_rel[0] == 0.05  # xmin
    assert bb_rel[1] == pytest.approx(1/3) # ymin
    assert bb_rel[2] == 0.30  # xmax
    assert bb_rel[3] == 1  # ymax

    bb_rel = bb.to_minmax_repr(Size(100, 210))
    assert bb_rel[0] == 0.1  # xmin
    assert bb_rel[1] == 0.6 # xmax
    assert bb_rel[2] == pytest.approx(1/3)  # ymin
    assert bb_rel[3] == 1  # ymax

    bb_rel = bb.to_centroid_repr(Size(200, 210))
    assert bb_rel[0] == 0.175  # cx
    assert bb_rel[1] == pytest.approx(2/3)  # cy
    assert bb_rel[2] == 0.25  # w
    assert bb_rel[3] == pytest.approx(2/3)  # h

    with pytest.raises(AttributeError):
        bb_abs = BoundingBox.from_centroid_repr(bb_rel, [100, 200])
    bb_abs = BoundingBox.from_centroid_repr(bb_rel, Size(200, 210))
    bb_approx_equal(bb_abs, bb)


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


def test_detection2d():
    d1 = Detection(2, BoundingBox.from_rect_repr(np.random.randint(0, 1e6, 4)), 0.5)
    d2 = Detection("person", BoundingBox.from_rect_repr(np.random.randint(0, 1e6, 4)), 1e7)

    assert d1.class_id == 2
    assert d1.score == 0.5
    assert d2.class_id == 'person'
    assert d2.score == 1e7

    dets = [
        d1, d2,
        Detection(20, BoundingBox.from_rect_repr(np.random.randint(0, 1e6, 4)), 0.5),
        Detection(2, BoundingBox.from_rect_repr(np.random.randint(0, 1e6, 4)), 0.5),
        Detection("car", BoundingBox.from_rect_repr(np.random.randint(0, 1e6, 4)), 0.5),
        Detection("person", BoundingBox.from_rect_repr(np.random.randint(0, 1e6, 4)), 0.5),
        Detection("person", BoundingBox.from_rect_repr(np.random.randint(0, 1e6, 4)), 0.5)
    ]
    ls = filter_detection_classes(dets, 2)
    assert len(ls) == 2
    # Filtering should keep the original order
    assert ls[0] == d1

    ls = filter_detection_classes(dets, "car")
    assert len(ls) == 1
    assert ls[0].class_id == 'car'

    ls = filter_detection_classes(dets, ["person", 2])
    assert len(ls) == 5
    assert ls[0] == d1
    assert ls[1] == d2
    assert ls[2].class_id == 2
    assert ls[3].class_id == 'person'
    assert ls[4].class_id == 'person'

    ls = filter_detection_classes(dets, [])
    assert len(ls) == 0
    ls = filter_detection_classes(dets, -2)
    assert len(ls) == 0

    d1 = Detection(2, BoundingBox.from_rect_repr([17, 3, 40, 10]), 0.5)
    d2 = d1.scale(3)
    assert d1.class_id == 2
    assert d1.score == 0.5
    assert d1.bounding_box.left == 51
    assert d1.bounding_box.width == 120
    assert d1.bounding_box.top == 9
    assert d1.bounding_box.height == 30
    assert d2 == d1

    d1.scale(0.5, 2)
    assert d1.class_id == 2
    assert d1.score == 0.5
    assert d1.bounding_box.left == 25.5
    assert d1.bounding_box.width == 60
    assert d1.bounding_box.top == 18
    assert d1.bounding_box.height == 60


def test_coco_labels():
    with pytest.raises(ValueError):
        LabelMapCOCO.class_id('foobar')
    with pytest.raises(ValueError):
        LabelMapCOCO.class_id(None)
    assert LabelMapCOCO.class_id('Person') == 1
    assert LabelMapCOCO.class_id('toothbrush') == 90

    det = Detection(2, BoundingBox.from_rect_repr(np.random.randint(0, 1e6, 4)), 0.5)
    assert LabelMapCOCO.label(det) == 'bicycle'

    with pytest.raises(ValueError):
        LabelMapCOCO.label(-1)
    assert LabelMapCOCO.label(1) == 'person'
    assert LabelMapCOCO.label(2) == 'bicycle'
    assert LabelMapCOCO.label(90) == 'toothbrush'


def test_voc07_labels():
    lbl_list = LabelMapVOC07.label_map.values()
    lm = LabelMap.from_list(lbl_list)

    with pytest.raises(ValueError):
        lm.class_id('foobar')
    with pytest.raises(ValueError):
        lm.class_id(None)

    det = Detection(3, BoundingBox.from_rect_repr(np.random.randint(0, 1e6, 4)), 0.5)
    assert lm.label(det) == 'bird'

    assert lm.class_id('Person') == 15
    assert lm.class_id('background') == 0
    assert lm.class_id('cow') == 10

    with pytest.raises(ValueError):
        lm.label(-1)
    assert lm.label(1) == 'aeroplane'
    assert lm.label(10) == 'cow'
    assert lm.label(20) == 'tvmonitor'

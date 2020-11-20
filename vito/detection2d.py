# import numpy as np
from types import SimpleNamespace


def iou(bbox1, bbox2):
    """
    Computes the intersection over union for the two bounding boxes
    given in rect representation: [left, top, width, height].
    """
    if bbox1 is None or bbox2 is None:
        return 0.0
    if isinstance(bbox1, BoundingBox):
        bbox1 = bbox1.to_rect_repr()
    if isinstance(bbox2, BoundingBox):
        bbox2 = bbox2.to_rect_repr()
    left1, top1, width1, height1 = bbox1
    left2, top2, width2, height2 = bbox2
    right1 = left1 + width1
    right2 = left2 + width2
    bottom1 = top1 + height1
    bottom2 = top2 + height2
    # Compute the coordinates of the intersection rectangle
    left = max(left1, left2)
    top = max(top1, top2)
    right = min(right1, right2)
    bottom = min(bottom1, bottom2)
    # Abort early if we don't have a valid intersection rectangle
    if right < left or bottom < top:
        return 0.0
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box. Right and bottom coordinates are exclusive.
    intersection_area = (right - left) * (bottom - top)
    # Compute the area of both AABBs
    bb1_area = width1 * height1
    bb2_area = width2 * height2
    # Area of the union
    union_area = float(bb1_area + bb2_area - intersection_area)
    if union_area > 0.0:
        return intersection_area / union_area
    return 0.0  # pragma: no cover


def filter_detection_classes(detections, class_filter):
    """
    Returns the filtered list of detections that satisfy the given 'class_filter'.

    detections:     A list of Detection instances.

    class_filter:   Either 1) a list of object class IDs or labels
                    or 2) a single object class ID or label.
    """
    return [d for d in detections if d.is_class(class_filter)]


class Size(SimpleNamespace):
    @classmethod
    def from_hw(cls, height, width):
        return cls(width, height)

    @classmethod
    def from_wh(cls, width, height):
        return cls(width, height)

    def __init__(self, width, height):
        super().__init__(width=width, height=height)

    def area(self):
        return self.width * self.height


class Detection(SimpleNamespace):
    """Axis-aligned bounding box with object class ID and detection confidence/score."""
    def __init__(self, class_id, bounding_box, score):
        super().__init__(class_id=class_id, bounding_box=bounding_box, score=score)

    def is_class(self, class_filter):
        """
        Checks if this instance satisfies the given 'class_filter'.

        class_filter:  Either 1) a list of object class IDs or labels
                        or 2) a single object class ID or label.
        """
        if isinstance(class_filter, list):
            return any([self.class_id == c for c in class_filter])
        else:
            return self.class_id == class_filter
    
    def scale(self, scale_x, scale_y=None):
        """Scale the bounding box."""
        self.bounding_box.scale(scale_x, scale_y)
        return self


class BoundingBox(SimpleNamespace):
    """Axis-aligned bounding box."""

    @classmethod
    def from_corner_repr(cls, corner_repr, img_size=None):
        """
        Returns a BoundingBox from the 4-element argument corner_repr = [xmin, ymin, xmax, ymax].
        If img_size is not None, the representation will be scaled by the image dimension,
        i.e. img_size.width and img_size.height.
        """
        xmin, ymin, xmax, ymax = corner_repr
        w, h = xmax - xmin, ymax - ymin
        return cls(left=xmin, top=ymin, width=w, height=h, img_size=img_size)

    @classmethod
    def from_minmax_repr(cls, minmax_repr, img_size=None):
        """
        Returns a BoundingBox from the 4-element argument minmax_repr = [xmin, xmax, ymin, ymax].
        If img_size is not None, the representation will be scaled by the image dimension,
        i.e. img_size.width and img_size.height.
        """
        xmin, xmax, ymin, ymax = minmax_repr
        w, h = xmax - xmin, ymax - ymin
        return cls(left=xmin, top=ymin, width=w, height=h, img_size=img_size)

    @classmethod
    def from_centroid_repr(cls, centroid_repr, img_size=None):
        """
        Returns a BoundingBox from the 4-element argument centroid_repr = [cx, cy, w, h].
        If img_size is not None, the representation will be scaled by the image dimension,
        i.e. img_size.width and img_size.height.
        """
        cx, cy, w, h = centroid_repr
        left, top = cx - w/2, cy - h/2
        return cls(left=left, top=top, width=w, height=h, img_size=img_size)

    @classmethod
    def from_rect_repr(cls, rect_repr, img_size=None):
        """
        Returns a BoundingBox from the 4-element argument rect_repr = [left, top, width, height].
        If img_size is not None, the representation will be scaled by the image dimension,
        i.e. img_size.width and img_size.height.
        """
        return cls(left=rect_repr[0], top=rect_repr[1], width=rect_repr[2], height=rect_repr[3], img_size=img_size)

    def __init__(self, left, top, width, height, img_size=None):
        if img_size is not None:
            left *= img_size.width
            width *= img_size.width
            top *= img_size.height
            height *= img_size.height
        super().__init__(left=left, top=top, width=width, height=height)

    def scale(self, scale_x, scale_y=None):
        if scale_y is None:
            scale_y = scale_x
        self.left *= scale_x
        self.width *= scale_x
        self.top *= scale_y
        self.height *= scale_y

    def to_corner_repr(self, img_size=None):
        """
        Returns [xmin, ymin, xmax, ymax] as absolute values.
        If img_size is not None, the coordinates will be scaled by the image
        dimension, i.e. in the range [0, 1].
        """
        if img_size is None:
            return [self.left, self.top, self.left + self.width, self.top + self.height]
        else:
            return [self.left / img_size.width, self.top / img_size.height,
                    (self.left + self.width) / img_size.width,
                    (self.top + self.height) / img_size.height]

    def to_minmax_repr(self, img_size=None):
        """
        Returns [xmin, xmax, ymin, ymax] as absolute values.
        If img_size is not None, values will be given relative to the (image) size.
        """
        corner = self.to_corner_repr(img_size=img_size)
        return [corner[0], corner[2], corner[1], corner[3]]

    def to_centroid_repr(self, img_size=None):
        """
        Returns [center_x, center_y, width, height] as absolute values.
        If img_size is not None, values will be given relative to the (image) size.
        """
        cx = self.left + self.width/2
        cy = self.top + self.height/2
        if img_size is None:
            return [cx, cy, self.width, self.height]
        else:
            return [cx / img_size.width, cy / img_size.height,
                    self.width / img_size.width, self.height / img_size.height]

    def to_rect_repr(self, img_size=None):
        """
        Returns [xmin, ymin, width, height] in absolute values.
        If img_size is not None, the representation will be given
        relative to the (image) size.
        """
        if img_size is None:
            return [self.left, self.top, self.width, self.height]
        else:
            return [self.left / img_size.width, self.top / img_size.height,
                    self.width / img_size.width, self.height / img_size.height]

    def iou(self, other):
        """Returns the intersection over union."""
        return iou(self.to_rect_repr(), other)

    def area(self):
        return self.width * self.height


# Object categories from MS COCO
CATEGORIES_COCO = {
    0: 'background',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}


# Object categories from PASCAL VOC 2007-2012
CATEGORIES_VOC07 = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor"
}


# TODO encapsulate coco/voc via
# LabelMapVOC = LabelMap.from_map(...)
class LabelMap(object):
    @classmethod
    def from_map(cls, label_map, name=None):
        """Creates a label map from a dictionary of { class_id : label }."""
        return cls(label_map, name)
    
    @classmethod
    def from_list(cls, label_list, name=None):
        """Creates a label map from a list of category labels."""
        lm = { i: lbl for i,lbl in enumerate(label_list) }
        return cls(lm, name)
    
    def __init__(self, label_map, name):
        self.label_map = label_map
        self.name = name
    
    def label(self, c):
        """Returns the class label (string) for the given class ID or Detection instance 'c'."""
        if isinstance(c, Detection):
            cid = c.class_id
        else:
            cid = c
        if cid in self.label_map:
            return self.label_map[cid]
        raise ValueError()

    def class_id(self, lbl):
        """Returns the class ID (integer) for the given label (string)."""
        if lbl is None:
            raise ValueError()
        return list(self.label_map.keys())[list(self.label_map.values()).index(lbl.lower())]


LabelMapVOC07 = LabelMap.from_map(CATEGORIES_VOC07)


LabelMapCOCO = LabelMap.from_map(CATEGORIES_COCO)

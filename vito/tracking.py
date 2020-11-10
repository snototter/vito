# import numpy as np
from types import SimpleNamespace


def iou(bbox1, bbox2):
    """Compute intersection over union for the two bounding boxes given in rect representation: [left, top, width, height]."""
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


class Target(object):
    """A tracked target encapsulating the previous trajectory and state information."""
    def __init__(self, object_id, class_id):
        """
        object_id: unique identifier for the tracked object
        class_id: ID of the (semantic) object class
        """
        pass

    def trajectory(self):
        pass

    def dominant_motion(self, num_steps=5):
        pass

    def similarity_iou(self, box):
        pass
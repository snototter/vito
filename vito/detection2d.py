import numpy as np
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


class BoundingBox(SimpleNamespace):
    @classmethod
    def from_corner_repr(cls, corner_repr):
        """Returns a BoundingBox from the 4-element argument corner_repr = [xmin, ymin, xmax, ymax]."""
        xmin, ymin, xmax, ymax = corner_repr
        w, h = xmax - xmin, ymax - ymin
        return cls(left=xmin, top=ymin, width=w, height=h)
    
    @classmethod
    def from_minmax_repr(cls, minmax_repr):
        """Returns a BoundingBox from the 4-element argument minmax_repr = [xmin, xmax, ymin, ymax]."""
        xmin, xmax, ymin, ymax = minmax_repr
        w, h = xmax - xmin, ymax - ymin
        return cls(left=xmin, top=ymin, width=w, height=h)
    
    @classmethod
    def from_centroid_repr(cls, centroid_repr):
        """Returns a BoundingBox from the 4-element argument centroid_repr = [cx, cy, w, h]."""
        cx, cy, w, h = centroid_repr
        left, top = cx - w/2, cy - h/2
        return cls(left=left, top=top, width=w, height=h)

    @classmethod
    def from_rect_repr(cls, rect_repr):
        """Returns a BoundingBox from the 4-element argument rect_repr = [left, top, width, height]."""
        return cls(left=rect_repr[0], top=rect_repr[1], width=rect_repr[2], height=rect_repr[3])
    
    def __init__(self, left, top, width, height):
        super().__init__(left=left, top=top, width=width, height=height)
    
    def to_corner_repr(self):
        """Returns [xmin, ymin, xmax, ymax]."""
        return [self.left, self.top, self.left + self.width, self.top + self.height]
    
    def to_minmax_repr(self):
        """Returns [xmin, xmax, ymin, ymax]."""
        corner = self.to_corner_repr()
        return [corner[0], corner[2], corner[1], corner[3]]
    
    def to_centroid_repr(self):
        """Returns [center_x, center_y, width, height]."""
        cx = self.left + self.width/2
        cy = self.top + self.height/2
        return [cx, cy, self.width, self.height]

    def to_rect_repr(self):
        """Returns [xmin, ymin, width, height]."""
        return [self.left, self.top, self.width, self.height]
    
    def iou(self, other):
        if other is None:
            return 0.0
        return iou(self.to_rect_repr(), other.to_rect_repr())
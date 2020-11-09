import numpy as np
from types import SimpleNamespace

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
        #TODO
        pass
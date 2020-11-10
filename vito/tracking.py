# import numpy as np
# from types import SimpleNamespace


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
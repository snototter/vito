import sys
import numpy as np


def floread(filename):
    """
    Read optical flow (.flo) files stored in Middlebury format.

    Adapted from https://stackoverflow.com/a/28016469/400948
    """
    if sys.byteorder != 'little':
        raise RuntimeError('Current .flo support requires little-endian architecture!')

    with open(filename, 'rb') as f:
        # Check magic number
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            raise ValueError('Invalid magic number in file "%s".' % filename)
        # Next, get the image dimensions
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # Load the data and reshape
        flow = np.fromfile(f, np.float32, count=2*w*h)
        return np.resize(flow, (h, w, 2))


def flosave(filename, flow):
    """Save HxWx2 optical flow to file in Middlebury format."""
    if len(flow.shape) != 3 or flow.shape[2] != 2:
        raise ValueError('Invalid flow shape!')
    # Prepare data
    h, w = flow.shape[0], flow.shape[1]
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    data = np.zeros((h, w*2))
    data[:, np.arange(w)*2] = u
    data[:, np.arange(w)*2 + 1] = v

    with open(filename, 'wb') as f:
        # Write magic number
        np.array([202021.25], np.float32).tofile(f)
        # Write dimensions as W,H (!)
        np.array(w).astype(np.int32).tofile(f)
        np.array(h).astype(np.int32).tofile(f)
        # Write actual data
        data.astype(np.float32).tofile(f)

import numpy as np
import os
import pytest
from ..flowutils import floread, flosave


def test_floread():
    exfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples', 'color_wheel.flo')
    assert floread(None) is None
    for fn in ['', 'a-non-existing.file']:
        with pytest.raises(FileNotFoundError):
            floread(fn)
    # Load example .flo
    flow = floread(exfile)
    assert flow.ndim == 3 and flow.shape[2] == 2
    assert flow.shape[0] == 151
    assert flow.shape[1] == 151
    assert flow.dtype == np.float32


def test_flosave(tmp_path):
    exfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples', 'color_wheel.flo')
    # Load example .flo
    flow_in = floread(exfile)
    assert flow_in.ndim == 3 and flow_in.shape[2] == 2
    assert flow_in.dtype == np.float32

    # Save and reload
    out_fn = str(tmp_path / 'test.flo')
    flosave(out_fn, flow_in)
    flow_out = floread(out_fn)
    assert flow_out.ndim == 3 and flow_out.shape[2] == 2
    assert flow_out.dtype == np.float32
    assert np.all(flow_in[:] == flow_out[:])

    # Try saving float64 (will be loaded as float32)
    flow_in64 = flow_in.astype(np.float64)
    assert flow_in64.dtype == np.float64
    flosave(out_fn, flow_in64)
    flow_out = floread(out_fn)
    assert flow_out.ndim == 3 and flow_out.shape[2] == 2
    assert flow_out.dtype == np.float32
    assert np.all(flow_in[:] == flow_out[:])

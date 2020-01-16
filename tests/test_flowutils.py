import numpy as np
import os
import pytest
from vito.flowutils import floread, flosave, colorize_flow


def test_floread():
    exfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples', 'color_wheel.flo')
    eximg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples', 'flamingo.jpg')
    assert floread(None) is None
    for fn in ['', 'a-non-existing.file']:
        with pytest.raises(FileNotFoundError):
            floread(fn)
    # Load image as flow file
    with pytest.raises(ValueError):
        floread(eximg)
    # Load example .flo
    flow = floread(exfile)
    assert flow.ndim == 3 and flow.shape[2] == 2
    assert flow.shape[0] == 151
    assert flow.shape[1] == 151
    assert flow.dtype == np.float32


def test_flosave(tmp_path):
    exfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples', 'color_wheel.flo')
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

    # Try saving an array of invalid shape as flow
    invalid = [
        np.zeros((3,), dtype=np.int32),
        np.zeros((15, 1), dtype=np.uint8),
        np.ones((10, 2), dtype=np.float32),
        np.zeros((20, 30, 1), dtype=np.float32),
        np.ones((4, 7, 3), dtype=np.float64)
    ]
    for i in invalid:
        with pytest.raises(ValueError):
            flosave(out_fn, i)


def assert_color_equal(a, b):
    for c in range(3):
        assert a[c] == b[c]


def test_colorize():
    # Try invalid inputs
    invalid = [
        np.zeros((3,), dtype=np.int32),
        np.zeros((15, 1), dtype=np.uint8),
        np.ones((10, 2), dtype=np.float32),
        np.zeros((20, 30, 1), dtype=np.float32),
        np.ones((4, 7, 3), dtype=np.float64)
    ]
    for i in invalid:
        with pytest.raises(ValueError):
            colorize_flow(i)

    examples = [
        np.zeros((10, 10, 2), dtype=np.uint8),
        np.zeros((10, 10, 2), dtype=np.float32),
        np.zeros((10, 10, 2), dtype=np.float64)
    ]
    for ex in examples:
        cf = colorize_flow(ex, max_val=None)
        assert np.all(cf[:] == 255)

    cf = colorize_flow(np.ones((10, 10, 2), dtype=np.float32))
    assert np.all(cf[:, :, 0] == 255)
    assert np.all(cf[:, :, 1] == 114)
    assert np.all(cf[:, :, 2] == 0)
    cf = colorize_flow(np.ones((10, 10, 2), dtype=np.float32), return_rgb=False)
    assert np.all(cf[:, :, 2] == 255)
    assert np.all(cf[:, :, 1] == 114)
    assert np.all(cf[:, :, 0] == 0)
    cf = colorize_flow(-np.ones((10, 10, 2), dtype=np.float32))
    assert np.all(cf[:, :, 0] == 0)
    assert np.all(cf[:, :, 1] == 52)
    assert np.all(cf[:, :, 2] == 255)
    # max_val should clip values into [0, max_val]
    cf = colorize_flow(-np.ones((10, 10, 2), dtype=np.float32), max_val=0)
    assert np.all(cf[:] == 255)

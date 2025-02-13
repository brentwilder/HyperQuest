import hyperquest
import numpy as np
from spectral import *


HDR_PATH = './tests/test_data.hdr'

def test_smile_metric():
    o2_mean, co2_mean, o2_std, co2_std = hyperquest.smile_metric(HDR_PATH)

    # output is 1d arry
    assert isinstance(o2_mean, np.ndarray)
    assert isinstance(co2_mean, np.ndarray)
    assert isinstance(o2_std, np.ndarray)
    assert isinstance(co2_std, np.ndarray)
    assert o2_mean.ndim == 1
    assert co2_mean.ndim == 1
    assert o2_std.ndim == 1
    assert co2_std.ndim == 1


def test_sub_pixel_shift():
    x_shift, y_shift = hyperquest.sub_pixel_shift(HDR_PATH)

    # output is a float and not nan
    assert isinstance(x_shift, float)
    assert isinstance(y_shift, float)
    assert not np.isnan(x_shift)
    assert not np.isnan(y_shift)
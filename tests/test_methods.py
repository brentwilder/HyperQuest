import hyperquest
import numpy as np
from spectral import *


PATH_TO_DATA = './tests/test_data.hdr'


def test_sub_pixel_shift():
    x_shift, y_shift = hyperquest.sub_pixel_shift(PATH_TO_DATA, band_index_vnir=0, band_index_vswir=250)

    # output is a float and not nan
    assert isinstance(x_shift, float)
    assert isinstance(y_shift, float)
    assert not np.isnan(x_shift)
    assert not np.isnan(y_shift)
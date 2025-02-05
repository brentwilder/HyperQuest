import numpy as np
from skimage.registration import phase_cross_correlation
from spectral import *

from .utils import *



def sub_pixel_shift(hdr_path, band_index_vnir, band_index_vswir, no_data_value=-9999, upsample_factor=5000):
    '''
    A wrapper function for skimage.registration's `phase_cross_correlation`



    Parameters:
        hdr_path (str): Path to the .hdr file..
        band_index_vnir (int): Band index for VNIR camera , assuming the first band is 0.
        band_index_vswir (int): Band index for VSWIR camera , assuming the first band is 0.
        no_data_value (int): Assumed to be -9999.
        upsample_factor (int): Upsampling factor. Images will be registered to within 1 / upsample_factor of a pixel. 

    Returns:
        Tuple containing shift in the X direction and shift in the Y direction (in pixels)
    '''

    # Load image data
    img_path = get_img_path_from_hdr(hdr_path)
    array = np.array(envi.open(hdr_path, img_path).load(), dtype=np.float64)

    # Select the desired bands (VNIR and VSWIR)
    vnir_band = array[:, :, band_index_vnir]
    vswir_band = array[:, :, band_index_vswir]
    
    # Mask no data values
    vnir_band = np.ma.masked_equal(vnir_band, no_data_value)
    vswir_band = np.ma.masked_equal(vswir_band, no_data_value)

    # Compute the shift using phase_cross_correlation
    estimated_shift, error, diffphase = phase_cross_correlation(vnir_band, vswir_band, 
                                                                upsample_factor=upsample_factor,
                                                                space = 'real')
    
    return estimated_shift[1], estimated_shift[0]






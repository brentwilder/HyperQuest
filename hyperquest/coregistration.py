import numpy as np
from skimage.registration import phase_cross_correlation
from spectral import *

from .utils import *


def sub_pixel_shift(path_to_data, band_index_vnir, band_index_vswir, no_data_value=-9999, upsample_factor=5000):
    '''
    A wrapper function for skimage.registration's `phase_cross_correlation` 

    Parameters
    ----------
    path_to_data : str
        Path to the .hdr or .nc
    band_index_vnir : int
        Band index for VNIR camera , assuming the first band is 0.
    band_index_vswir : int
        Band index for VSWIR camera , assuming the first band is 0.
    no_data_value : int
        Assumed to be -9999.
    upsample_factor : int
        Upsampling factor. Images will be registered to within 1 / upsample_factor of a pixel. 

    Returns
    -------
    tuple
        Tuple containing shift in the X direction, shift in the Y direction (in pixels), and wavelength for VNIR, wavelength for VSWIR
    '''

    # Identify data type
    if path_to_data.lower().endswith('.nc'):
        array, _, w, _ = retrieve_data_from_nc(path_to_data)
    else:
        # Load raster 
        img_path = get_img_path_from_hdr(path_to_data)
        array = np.array(envi.open(path_to_data, img_path).load(), dtype=np.float64)
        # get wavelengths
        w, _, _ = read_hdr_metadata(path_to_data)

    # Select the desired bands (VNIR and VSWIR)
    vnir_band = array[:, :, band_index_vnir]
    vswir_band = array[:, :, band_index_vswir]

    # Save wavelength values for output
    w_vnir = w[band_index_vnir]
    w_vswir = w[band_index_vswir] 
    
    # Mask no data values
    vnir_band = np.ma.masked_equal(vnir_band, no_data_value)
    vswir_band = np.ma.masked_equal(vswir_band, no_data_value)

    # Compute the shift using phase_cross_correlation
    estimated_shift, error, diffphase = phase_cross_correlation(vnir_band, vswir_band, 
                                                                upsample_factor=upsample_factor,
                                                                space = 'real')
    
    return estimated_shift[1], estimated_shift[0], w_vnir, w_vswir


import numpy as np
from spectral import *

from .utils import *

def smile_metric(hdr_path, mask_waterbodies=True, no_data_value=-9999):
    '''
    TODO

    dBand = (Band1 - Band2) / mean(FWHM from Band1 and Band2)

    Band1 is absorption band (either CO2 or O2)
    Band2 is the following band
    dBand is the computed derivative along the column.


    computes the column mean derivatives, and their standard deviations, for the O2 and CO2 absorption features 


    '''

    # Load raster
    img_path = get_img_path_from_hdr(hdr_path)
    array = np.array(envi.open(hdr_path, img_path).load(), dtype=np.float64)

    # mask waterbodies
    if mask_waterbodies is True:
        array = mask_water_using_ndwi(array, hdr_path)

    # Mask no data values
    array = np.ma.masked_equal(array, no_data_value)
  
    # get wavelengths
    w = read_center_wavelengths(hdr_path)

    # get fwhm
    fwhm = read_fwhm(hdr_path)

    # ensure these are the same length
    if len(w) != len(fwhm):
        raise ValueError('Wavelength and FWHM arrays have different lengths.')

    #  first, ensure the wavelengths covered the span of o2 and co2 features
    # If they do not, then these are filled with -9999.
    if np.max(w) < 800:
        co2_mean = np.full(array.shape[1], fill_value=-9999)
        co2_std = np.full(array.shape[1], fill_value=-9999)
        o2_mean = np.full_like(o2_mean, fill_value=-9999)
        o2_std = np.full_like(o2_mean, fill_value=-9999)
        return o2_mean, co2_mean, o2_std, co2_std

    # Find closest band to co2 and O3
    # based on Dadon et al. (2010)
    # o2 :  B1=772-nm   B2=next 
    # co2 : B1=2012-nm  B2=next 
    o2_index = np.argmin(np.abs(w - 772))
    co2_index = np.argmin(np.abs(w - 2012))

    # compute derivative
    o2_b1 = array[:, :, o2_index] 
    o2_b2 = array[:, :, o2_index+1] 
    fwhm_bar_o2 = np.nanmean([fwhm[o2_index], fwhm[o2_index+1]])
    o2_dband = (o2_b1 + o2_b2) / fwhm_bar_o2

    # Compute cross-track (columnwise) means and standard deviation
    o2_mean = np.nanmean(o2_dband, axis=0)
    o2_std = np.nanstd(o2_dband, axis=0)

    # likely has enough data to find CO2
    if np.max(w)>2100: 
        co2_b1 = array[:, :, co2_index] 
        co2_b2 = array[:, :, co2_index+1]

        fwhm_bar_co2 = np.nanmean([fwhm[co2_index], fwhm[co2_index+1]])

        co2_dband = (co2_b1 + co2_b2) / fwhm_bar_co2

        co2_mean = np.nanmean(co2_dband, axis=0)
        co2_std = np.nanstd(co2_dband, axis=0)

    else: # return -9999 just for the co2 data
        co2_mean = np.full(array.shape[1], fill_value=-9999)
        co2_std = np.full(array.shape[1], fill_value=-9999)


    return o2_mean, co2_mean, o2_std, co2_std

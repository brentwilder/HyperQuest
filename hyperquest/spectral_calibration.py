import numpy as np
from spectral import *

from utils import *


def wavelength_dispersion(hdr_path, sensor_zenith_angle, sensor_azimuth_angle, mask_waterbodies=True, no_data_value=-9999):
    '''
    TODO

    inputs
        radiative_transfer: MODTRAN OR libRadtran OR pre-built LUT

        path_to_radiative_transfer: file path to install of radiative transfer model

        method: OE

        SNR : optional else 200...


    
        
    Reduces to 1 row as in Thompson et al. 2024

    Runs required table of libRadtran etc... (can be done in parallel - dependent on RAM)

    Runs OE for the row (can be done in parallel)

    Finish..


    NOTE: 
        - SNR can be matrix of 200 (Thompson 2024) or perhaps I can have an input for this too.
        - no-data can also be pulled from .hdr file



    '''
    
    # Load raster
    img_path = get_img_path_from_hdr(hdr_path)
    array = np.array(envi.open(hdr_path, img_path).load(), dtype=np.float64)

    # mask waterbodies
    if mask_waterbodies is True:
        array = mask_water_using_ndwi(array, hdr_path)

    # Mask no data values
    array = np.ma.masked_equal(array, no_data_value)

    # Average in down-track direction (reduce to 1 row)
    array = np.nanmean(array, axis=0)
    print(array.shape)

    # Get Copernicus DEM data based on bounding box in hdr

    # Average elevation (assume mus = mu0 ; flat assumption)

    # Get aquisition time from hdr

    # Plug all information into radiative transfer model
    # generate tables where the following vary:
    #   H20: []
    #   AOT???
    #
    # Parallel computing here


    # "At runtime, we calculate an initial guess of surface reflectance using a band ratio water vapor retrieval as in Schläpfer et al. (1998). "


    # Feed everything into OE method.





    return


# TESTING
hdr_path = '/Users/brent/Code/HyperQuest/tests/data/SISTER_EMIT_L1B_RDN_20220827T091626_000.hdr'



wavelength_dispersion(hdr_path)
import numpy as np
import pandas as pd
from spectral import *
from dem_stitcher import stitch_dem

from utils import *
from leastsquares import *


def cogliati_2021(hdr_path, path_to_rtm_output_csv, 
                  absorption_band_target=760, mask_waterbodies=True, no_data_value=-9999):
    '''
    TODO

    NOTE / TODO:
    1) very sensitive to surface reflectance... 
     
    2) still have not figured out to bring in tau_550.. main issue is that libRadtran is slow with fine` activated. This would also help #1 
    if I had the full spectrum computed.

    3) 

    Reduces to 1 row as in Thompson et al. 2024

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

    # Value for surface reflectance given
    rho_surface = 0.7

    # Get data from hdr
    w, fwhm, obs_time = read_hdr_metadata(hdr_path)

    # find closest band match to 760 for O2-A.
    o2A_index = np.argmin(np.abs(w - absorption_band_target))
    l_toa_observed = array[:, o2A_index]

    # Gather initial vector
    # [CWL, fwhm]
    x0 = [w[o2A_index], fwhm[o2A_index]]
    print(x0)

    # Read out the results from rtm 
    # l0, t_up, sph_alb, s_total
    df = pd.read_csv(path_to_rtm_output_csv)
    s_total = df['e_dir'].values + df['e_diff'].values
    rtm_wave = df['Wavelength'].values
    rtm_radiance = df['l0'].values + (1/np.pi) * ((rho_surface * s_total* df['t_up'].values) / (1 - df['s'].values * rho_surface))


    # loop across-track
    for l in l_toa_observed:
        if l > 0:
            cwl,fwhm = invert_cwl_and_fwhm(x0, l, rtm_radiance, rtm_wave)
            print(cwl, fwhm)
            break
        else:
            print(l)



    return


# TESTING
hdr_path = '/Users/brent/Code/HyperQuest/tests/data/SISTER_EMIT_L1B_RDN_20220827T091626_000.hdr'

path_to_rtm_output_csv = "/Users/brent/Code/HyperQuest/tests/data/rtm-SISTER_EMIT_L1B_RDN_20220827T091626_000/radiative_transfer_output.csv"

cogliati_2021(hdr_path, path_to_rtm_output_csv, absorption_band_target=760, mask_waterbodies=True, no_data_value=-9999)
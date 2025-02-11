import numpy as np
import pandas as pd
from spectral import *
from dem_stitcher import stitch_dem

from utils import *
from leastsquares import *



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
    w, fwhm, obs_time = read_hdr_metadata(hdr_path)

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




def wavelength_shift_o2a(hdr_path, path_to_rtm_output_csv, 
                         mask_waterbodies=True, no_data_value=-9999):
    '''
    TODO

    NOTE / TODO:

    1) compute 549 - 850 nm for libradtran, also get these wavelengths range from Sensor.

    2) Solve analytically L-TOA for surface reflectance based on atmos inputs.

    3) Change x0 to be [dlambda-shift, FWHM_i] 
        -> dlambda will change the few bands CWL in that window together.
        -> FWHM_i is each of the bands FWHM
        --> with SRF, and convuloution, these can produce a new Band radiance.

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
    absorption_band_target = 760 # nm
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

wavelength_shift_o2a(hdr_path, path_to_rtm_output_csv, absorption_band_target=760, mask_waterbodies=True, no_data_value=-9999)
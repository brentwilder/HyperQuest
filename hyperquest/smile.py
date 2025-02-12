import numpy as np
import pandas as pd
from spectral import *
from dem_stitcher import stitch_dem

from utils import *
from optimization import *



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
                         surface_reflectance,
                         mask_waterbodies=True, no_data_value=-9999):
    '''
    TODO

    '''
    
    # Load raster
    img_path = get_img_path_from_hdr(hdr_path)
    array = np.array(envi.open(hdr_path, img_path).load(), dtype=np.float64)

    # mask waterbodies
    if mask_waterbodies is True:
        array = mask_water_using_ndwi(array, hdr_path)

    # Mask no data values
    array[array <= -9999] = np.nan

    # Average in down-track direction (reduce to 1 row)
    array = np.nanmean(array, axis=0)

    # Get data from hdr
    w_sensor, fwhm, obs_time = read_hdr_metadata(hdr_path)

    # Only include window for o2-a
    window = (w_sensor >= 650) & (w_sensor <= 870)
    w_sensor = w_sensor[window]
    fwhm = fwhm[window]
    l_toa_observed = array[:, window]

    # Read out the results from rtm 
    # l0, t_up, sph_alb, s_total
    df = pd.read_csv(path_to_rtm_output_csv)
    df = df[(df['Wavelength'] >= 700) & (df['Wavelength'] <= 800)]
    s_total = df['e_dir'].values + df['e_diff'].values
    w_rtm = df['Wavelength'].values
    t_up = df['t_up'].values
    sph_alb = df['s'].values
    l0 = df['l0'].values
    surface_reflectance =  np.full_like(s_total, fill_value=0.4)
    l_toa_rtm = l0 + (1/np.pi) * ((surface_reflectance * s_total* t_up) / (1 - sph_alb * surface_reflectance))


    # Still todo:
    # this is getting better..
    # but can possibly normalize it?
    #  also   should limit the window in inversion to be actually just 50-60nm range
        # should reviist that Guanter paper next time


    # 
    # Next steps for optimization
    # Gather initial vector  [CWL, FWHM]
    x0 = [5] + fwhm.tolist()
    # next radiance is computed assuming gaussian SRF
    # repeats cross-track.
    for l in l_toa_observed:
        if l[0]>0:
            invert_cwl_and_fwhm(x0,l, l_toa_rtm, w_rtm, w_sensor)
            break






    return






# TESTING
hdr_path = '/Users/brent/Code/HyperQuest/tests/data/SISTER_EMIT_L1B_RDN_20220827T091626_000.hdr'

path_to_rtm_output_csv = "/Users/brent/Code/HyperQuest/tests/data/rtm-SISTER_EMIT_L1B_RDN_20220827T091626_000/radiative_transfer_output.csv"

wavelength_shift_o2a(hdr_path, path_to_rtm_output_csv, surface_reflectance=0.4,
                  mask_waterbodies=True, no_data_value=-9999)
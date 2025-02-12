import numpy as np
import pandas as pd
from spectral import *
from joblib import Parallel, delayed

from .utils import *
from .optimization import *



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
    array[array <= no_data_value] = np.nan
  
    # get wavelengths
    w, fwhm, obs_time = read_hdr_metadata(hdr_path)

    # set up outputs
    co2_mean = np.full(array.shape[1], fill_value=np.nan)
    co2_std = np.full(array.shape[1], fill_value=np.nan)
    o2_mean = np.full(array.shape[1], fill_value=np.nan)
    o2_std = np.full(array.shape[1], fill_value=np.nan)

    #  first, ensure the wavelengths covered the span of o2 and co2 features
    if np.max(w) < 800:
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

    # Compute cross-track (columnwise) means and standard deviation (w/respect to camera)
    o2_mean, o2_std = cross_track_stats(o2_dband)
    o2_mean = o2_mean.flatten()
    o2_std = o2_std.flatten()

    # likely has enough data to find CO2
    if np.max(w)>2100: 
        co2_b1 = array[:, :, co2_index] 
        co2_b2 = array[:, :, co2_index+1]
        fwhm_bar_co2 = np.nanmean([fwhm[co2_index], fwhm[co2_index+1]])
        co2_dband = (co2_b1 + co2_b2) / fwhm_bar_co2
        co2_mean, co2_std = cross_track_stats(co2_dband)
        co2_mean = co2_mean.flatten()
        co2_std = co2_std.flatten()

    return o2_mean, co2_mean, o2_std, co2_std


def nodd_o2a(hdr_path, path_to_rtm_output_csv, ncpus=1,rho_s=0.15, mask_waterbodies=True, no_data_value=-9999):
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
    array[array <= no_data_value] = np.nan

    # Average in down-track direction (reduce to 1 row)
    array,_ = cross_track_stats(array)

    # Get data from hdr
    w_sensor, fwhm, obs_time = read_hdr_metadata(hdr_path)

    # Only include window for o2-a
    window = (w_sensor >= 730) & (w_sensor <= 790)
    w_sensor = w_sensor[window]
    fwhm = fwhm[window]
    l_toa_observed = array[:, window]

    # Read out the results from rtm 
    # l0, t_up, sph_alb, s_total
    df = pd.read_csv(path_to_rtm_output_csv)
    df = df[(df['Wavelength'] >= 730) & (df['Wavelength'] <= 790)]
    s_total = df['e_dir'].values + df['e_diff'].values
    w_rtm = df['Wavelength'].values
    t_up = df['t_up'].values
    sph_alb = df['s'].values
    l0 = df['l0'].values
    rho_s =  np.full_like(s_total, fill_value=rho_s)
    l_toa_rtm = l0 + (1/np.pi) * ((rho_s * s_total* t_up) / (1 - sph_alb * rho_s))

    # Next steps for optimization
    # Gather initial vector  [dlambda, dFWHM]
    dfwhm = np.full_like(fwhm, fill_value= 0.0)
    x0 = [0.0] + dfwhm.tolist()

    # paralell cross-track CWL and FHWM
    results = Parallel(n_jobs=ncpus)(
        delayed(invert_cwl_and_fwhm)(x0, l, l_toa_rtm, w_rtm, w_sensor, fwhm) 
        for l in l_toa_observed
    )

    # Convert results to arrays
    cwl_opt, fwhm_opt = map(np.array, zip(*results))

    # for user show the band that is closest to 760 that is being referred to.
    o2_a =  np.argmin(np.abs(w_sensor-760))
    sensor_band_near_760 = w_sensor[o2_a]
    fwhm_near_760 = fwhm[o2_a]


    return cwl_opt, fwhm_opt, sensor_band_near_760, fwhm_near_760
import numpy as np
import pandas as pd
from spectral import *
from joblib import Parallel, delayed

from .utils import *
from .optimization import *


def smile_metric(path_to_data, rotate, mask_waterbodies=True):
    '''
    This is an exact remake of the MATLAB method smileMetric(), https://www.mathworks.com/help/images/ref/smilemetric.html.

    This is based from the work presented in Dadon et al. (2010),

    Dadon, A., Ben-Dor, E., & Karnieli, A. (2010). Use of derivative calculations and minimum noise
    fraction transformfor detecting and correcting the spectral curvature effect (smile) in Hyperion images. 
    IEEE Transactions on Geoscience and Remote Sensing, 48(6), 2603-2612.

    Parameters
    ----------
    path_to_data : str
        Path to the .hdr or .nc
    rotate : int
        rotate counter clockwise, either 0, 90, 180, or 270.
    mask_waterbodies : bool, optional
        Whether to mask water bodies based on NDWI threshold of 0.25. Default is True.

    Returns
    -------
        o2_mean, co2_mean, o2_std, co2_std: 1d array of cross-track mean do2, mean dco2, std do2, std dco2

    '''

    # Identify data type
    if path_to_data.lower().endswith('.nc'):
        array, fwhm, w, obs_time = retrieve_data_from_nc(path_to_data)
    else:
        # Load raster
        img_path = get_img_path_from_hdr(path_to_data)
        array = np.array(envi.open(path_to_data, img_path).load(), dtype=np.float64)
        
        # get wavelengths
        w, fwhm, obs_time = read_hdr_metadata(path_to_data)

    # ensure this is the raw data and has not been georeferenced.
    if (array[:,:,0]<0).sum() > 0:
        raise Exception('Please provide data that has not been geo-referenced.')

    # ensure data is 3d
    if len(array.shape) != 3:
        raise Exception('Data needs to be a 3D array.')
    
    # ensure data is hyperspectral 
    if (np.max(w) - np.min(w)) / len(w) > 50: # assume hyperspectral data not coarser than 50 nm spec res
        raise Exception('Data needs to be a hyperspectral image.')
    
    # Perform rotation if needed
    if rotate != 0 and rotate != 90 and rotate != 180 and rotate != 270:
        raise ValueError('rotate must be 90, 180, or 270.')
    if rotate>=90:
        array = np.rot90(array, axes=(0,1), k=1)
    elif rotate>=180:
        array = np.rot90(array, axes=(0,1), k=2)
    elif rotate>=270:
        array = np.rot90(array, axes=(0,1), k=3)

    # mask waterbodies
    if mask_waterbodies is True:
        array = mask_water_using_ndwi(array, w, no_data_value=np.nan)
  
    # set up outputs
    co2_mean = np.full(array.shape[1], fill_value=np.nan)
    co2_std = np.full(array.shape[1], fill_value=np.nan)
    o2_mean = np.full(array.shape[1], fill_value=np.nan)
    o2_std = np.full(array.shape[1], fill_value=np.nan)

    #  first, ensure the wavelengths covered the span of o2 and co2 features
    if np.max(w) < 800: # this will capture even VNIR only like DESIS (400-1000nm)
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
    o2_mean = np.nanmean(o2_dband, axis=0)
    o2_std = np.nanmean(o2_dband, axis=0)
    o2_mean = o2_mean.flatten()
    o2_std = o2_std.flatten()

    # likely has enough data to find CO2
    if np.max(w)>2100: # this is a bit arbitrary, but assessing if there is VSWIR data.
        co2_b1 = array[:, :, co2_index] 
        co2_b2 = array[:, :, co2_index+1]
        fwhm_bar_co2 = np.nanmean([fwhm[co2_index], fwhm[co2_index+1]])
        co2_dband = (co2_b1 + co2_b2) / fwhm_bar_co2
        co2_mean = np.nanmean(co2_dband, axis=0)
        co2_std = np.nanmean(co2_dband, axis=0)
        co2_mean = co2_mean.flatten()
        co2_std = co2_std.flatten()

    return o2_mean, co2_mean, o2_std, co2_std


def nodd_o2a(path_to_data, rotate, path_to_rtm_output_csv, ncpus=1,rho_s=0.15, mask_waterbodies=True):
    '''
    Similar to method in Felde et al. (2003) to solve for nm shift at O2-A across-track. Requires radiative transfer model run.

    NODD stands for Normalized Optical Depth Derivative, and was introduced in Felde et al. (2003). It is a method that is largely insensitive to
    surface reflectance and the molecular column density. The actual equations for NODD are present in nodd_sse_min() in the optimization file, and are copied here for reference.

    # gradient of negative natural log (gets at transmittance)
    dtau_obs = np.gradient(-np.log(l), w_sensor)
    dtau_model = np.gradient(-np.log(l_toa_model), w_sensor)

    # offset mean
    dtau_obs -= np.mean(dtau_obs)
    dtau_model -= np.mean(dtau_model)

    # normalize by RMS
    nodd_obs = dtau_obs / np.sqrt(np.mean(dtau_obs**2))
    nodd_model = dtau_model / np.sqrt(np.mean(dtau_model**2))

    # compute residual and SSE
    residual =  nodd_model - nodd_obs
    sse = np.sum(residual**2)

    
    Parameters
    ----------
    path_to_data : str
        Path to the .hdr or .nc
    rotate : int
        rotate counter clockwise, either 0, 90, 180, or 270.
    path_to_rtm_output_csv : str
        Path to output from radiative transfer.
    ncpus : int, optional
        Number of CPUs for parallel processing. Default is 1.
    rho_s : float
        value from 0-1. As stated, this does not influence nodd method very much and 0.15 is common in literature.
    mask_waterbodies : bool, optional
        Whether to mask water bodies based on NDWI threshold of 0.25. Default is True.

    Returns
    -------
        cwl_opt, fwhm_opt, sensor_band_near_760, fwhm_near_760: 1d array of cross-track CWL, 1d array of cross-track FWHM, band near 760, fwhm near 760

    '''
    
    # Identify data type
    if path_to_data.lower().endswith('.nc'):
        array, fwhm, w_sensor, obs_time = retrieve_data_from_nc(path_to_data)
    else:
        # Load raster
        img_path = get_img_path_from_hdr(path_to_data)
        array = np.array(envi.open(path_to_data, img_path).load(), dtype=np.float64)
        
        # get wavelengths
        w_sensor, fwhm, obs_time = read_hdr_metadata(path_to_data)

    # ensure data is hyperspectral 
    if (np.max(w_sensor) - np.min(w_sensor)) / len(w_sensor) > 50: # assume hyperspectral data not coarser than 50 nm spec res
        raise Exception('Data needs to be a hyperspectral image.')

    # ensure data has wavelength range in O2-A band
    if np.max(w_sensor) < 790: # 790 is max window used in fitting procedure
        raise Exception(f'Wavelength range of {np.min(w_sensor)}-{np.max(w_sensor)} nm is not appropriate for this method.')

    # ensure this is the raw data and has not been georeferenced.
    if (array[:,:,0]<0).sum() > 0:
        raise Exception('Please provide data that has not been geo-referenced.')

    # ensure data is 3d
    if len(array.shape) != 3:
        raise Exception('Data needs to be a 3D array.')
    
    # Perform rotation if needed
    if rotate != 0 and rotate != 90 and rotate != 180 and rotate != 270:
        raise ValueError('rotate must be 90, 180, or 270.')
    if rotate>=90:
        array = np.rot90(array, axes=(0,1), k=1)
    elif rotate>=180:
        array = np.rot90(array, axes=(0,1), k=2)
    elif rotate>=270:
        array = np.rot90(array, axes=(0,1), k=3)

    # mask waterbodies
    if mask_waterbodies is True:
        array = mask_water_using_ndwi(array, w_sensor, no_data_value=np.nan)

    # Average in down-track direction (reduce to 1 row)
    array = np.nanmean(array, axis=0)

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
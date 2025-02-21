import numpy as np
from spectral import *
from joblib import Parallel, delayed
from skimage.segmentation import slic
from sklearn.decomposition import PCA

from .utils import *
from .mlr import *


def rlsd(path_to_data, block_size, nbins=150, ncpus=1, snr_in_db = False, mask_waterbodies=True, no_data_value=-9999):
    '''
    Residual-scaled local standard deviation (Gao et al., 2007).

    Neighbor pixel not included in MLR, ( p_k-1 , p_k+1).
    
    Parameters
    ----------
    path_to_data : str
        Path to the .hdr or .nc file that contains hyperspectral data.
    block_size : int
        Block size for partitioning (e.g., 5 would be 5x5 pixels).
    nbins : int, optional
        Number of bins for histogram analysis. Default is 150.
    ncpus : int, optional
        Number of CPUs for parallel processing. Default is 1.
    snr_in_db : bool, optional
        Whether the SNR is in dB. Default is False.
    mask_waterbodies : bool, optional
        Whether to mask water bodies based on an NDWI threshold of 0.25. Default is True.
    no_data_value : int or float
        Value used to describe no data regions.

    Returns
    -------
    tuple
        A tuple containing the Signal-to-Noise Ratio (SNR) and noise variance with respect to wavelength.

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

    # ensure data is hyperspectral 
    if (np.max(w) - np.min(w)) / len(w) > 50: # assume hyperspectral data not coarser than 50 nm spec res
        raise Exception('Data needs to be a hyperspectral image.')

    # mask waterbodies
    if mask_waterbodies is True:
        array = mask_water_using_ndwi(array, w)

    # Mask no data values
    array[array <= no_data_value] = np.nan

    # Pad image to ensure divisibility by block_size
    array = pad_image(array, block_size)

    # get tasks (number of blocks)
    tasks = get_blocks(array, block_size)
    
    # Parallel processing of blocks using joblib
    results = Parallel(n_jobs=ncpus)(delayed(mlr_spectral)(block) for block in tasks)

    # Create empty lists
    local_mu = []
    local_sigma = []

    # Collect results
    for block_idx, (m, s) in enumerate(results):
        local_mu.append(m)
        local_sigma.append(s)
    local_mu = np.array(local_mu)
    local_sigma = np.array(local_sigma)

    # Bin and compute SNR
    mu, sigma = binning(local_mu, local_sigma, nbins)

    # remove atmos windows
    mu = mask_atmos_windows(mu, w)
    sigma = mask_atmos_windows(sigma, w)

    # Compute SNR
    snr = np.divide(mu, sigma, out=np.zeros_like(mu), where=(sigma != 0))
    snr[sigma == 0] = np.nan

    # check to convert to db
    if snr_in_db is True:
        snr = linear_to_db(snr)

    # convert noise to variance
    noise_variance = np.square(sigma, dtype=np.float64)

    return snr, noise_variance


def ssdc(path_to_data, block_size, nbins=150, ncpus=1, snr_in_db = False, mask_waterbodies=True, no_data_value=-9999):
    '''
    Spectral and spatial de-correlation (Roger & Arnold, 1996).

    Neighbor pixel with band K is included in MLR, ( p_k-1 , p_k+1, p+1_k ).

    Parameters
    ----------
    path_to_data : str
        Path to the .hdr or .nc file that contains hyperspectral data.
    block_size : int
        Block size for partitioning (e.g., 5 would be 5x5 pixels).
    nbins : int, optional
        Number of bins for histogram analysis. Default is 150.
    ncpus : int, optional
        Number of CPUs for parallel processing. Default is 1.
    snr_in_db : bool, optional
        Whether the SNR is in dB. Default is False.
    mask_waterbodies : bool, optional
        Whether to mask water bodies based on an NDWI threshold of 0.25. Default is True.
    no_data_value : int or float
        Value used to describe no data regions.

    Returns
    -------
    tuple
        A tuple containing the Signal-to-Noise Ratio (SNR) and noise variance with respect to wavelength.

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

    # ensure data is hyperspectral 
    if (np.max(w) - np.min(w)) / len(w) > 50: # assume hyperspectral data not coarser than 50 nm spec res
        raise Exception('Data needs to be a hyperspectral image.')

    # mask waterbodies
    if mask_waterbodies is True:
        array = mask_water_using_ndwi(array, w, no_data_value=no_data_value)

    # Mask no data values
    array[array <= no_data_value] = np.nan

    # Pad image to ensure divisibility by block_size
    array = pad_image(array, block_size)

    # get tasks (number of blocks)
    tasks = get_blocks(array, block_size)
    
    # Parallel processing of blocks using joblib
    results = Parallel(n_jobs=ncpus)(delayed(mlr_spectral_spatial)(block) for block in tasks)

    # Create empty lists
    local_mu = []
    local_sigma = []

    # Collect results
    for block_idx, (m, s) in enumerate(results):
        local_mu.append(m)
        local_sigma.append(s)
    local_mu = np.array(local_mu)
    local_sigma = np.array(local_sigma)

    # Bin and compute SNR
    mu, sigma = binning(local_mu, local_sigma, nbins)

    # remove atmos windows
    mu = mask_atmos_windows(mu, w)
    sigma = mask_atmos_windows(sigma, w)

    # Compute SNR
    snr = np.divide(mu, sigma, out=np.zeros_like(mu), where=(sigma != 0))
    snr[sigma == 0] = np.nan

    # check to convert to db
    if snr_in_db is True:
        snr = linear_to_db(snr)

    # convert noise to variance
    noise_variance = np.square(sigma, dtype=np.float64)

    return snr, noise_variance


def hrdsdc(path_to_data, n_segments=200, compactness=0.1, n_pca=3, ncpus=1, 
           snr_in_db=False, mask_waterbodies=True, no_data_value=-9999):
    '''
    Homogeneous regions division and spectral de-correlation (Gao et al., 2008). 
    
    Neighbor pixel with band K is included in MLR, ( p_k-1 , p_k+1, p+1_k ).

        
    Parameters
    ----------
    path_to_data : str
        Path to the .hdr or .nc file that contains hyperspectral data.
    n_segments : int
        The (approximate) number of labels in the segmented output image. see skimage.segmentation.slic for more.
    compactness : float
        Balances color proximity and space proximity. Higher values give more weight to space proximity, making superpixel shapes more square/cubic.see skimage.segmentation.slic for more.
    ncpus : int, optional
        Number of CPUs for parallel processing. Default is 1.
    include_neighbor_pixel_in_mlr : bool, optional
        If True, neighbor pixel is used in MLR (for k`). Else, MLR only contains spectral data (k+1, k-1).
    snr_in_db : bool, optional
        Whether the SNR is in dB. Default is False.
    mask_waterbodies : bool, optional
        Whether to mask water bodies based on an NDWI threshold of 0.25. Default is True.
    no_data_value : int or float
        Value used to describe no data regions.

    Returns
    -------
    tuple
        A tuple containing the Signal-to-Noise Ratio (SNR) and noise variance with respect to wavelength.

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

    # ensure data is hyperspectral 
    if (np.max(w) - np.min(w)) / len(w) > 50: # assume hyperspectral data not coarser than 50 nm spec res
        raise Exception('Data needs to be a hyperspectral image.')

    # mask waterbodies
    if mask_waterbodies is True:
        array = mask_water_using_ndwi(array, w, no_data_value=no_data_value)

    # Mask no data values (to negative 9999 for PCA and SLIC to work)
    array[array <= no_data_value] = -9999

    # Apply PCA 
    pca = PCA(n_components=n_pca)
    rows, cols, bands = array.shape
    array_reshaped = array.reshape(-1, bands)
    array_pca = pca.fit_transform(array_reshaped).reshape(rows, cols, -1)

    # SLIC
    segments = slic(array_pca, 
                    n_segments=n_segments, 
                    compactness=compactness)

    # find unique SLIC segments
    unique_segments = np.unique(segments)

    # Prepare SLIC segements for MLR in parallel
    def process_segment(u):
        test_mask = (segments == u)
        test_segment = array[test_mask]
        test_segment = test_segment[test_segment[:, 0] > -99] #spatial nan
        if test_segment.shape[0] != 0:
            return test_segment
        else:
            return None
    segment_data = Parallel(n_jobs=ncpus)(delayed(process_segment)(u) for u in unique_segments)
    segment_data = [seg for seg in segment_data if seg is not None]

    # Parallel processing of all segments 
    # perform spectral-spatial MLR using k` nearby neighbor.
    results = Parallel(n_jobs=ncpus, 
                        timeout=None)(delayed(mlr_spectral_spatial)(segment) for segment in segment_data) 
    
    # Aggregate results
    local_mu = np.array([res[0] for res in results])
    local_sigma = np.array([res[1] for res in results])

    # Average over segments for each band
    # first and last are empty due to k-1 k+1 in regression...
    mu_valid = np.nanmean(local_mu[:, 1:-1], axis=0)
    sigma_valid = np.nanmean(local_sigma[:, 1:-1], axis=0)
    mu = np.concatenate(([np.nan], mu_valid, [np.nan]))
    sigma = np.concatenate(([np.nan], sigma_valid, [np.nan]))

    # remove atmos windows
    mu = mask_atmos_windows(mu, w)
    sigma = mask_atmos_windows(sigma, w)

    # Compute SNR
    snr = np.divide(mu, sigma, out=np.zeros_like(mu), where=(sigma != 0))
    snr[sigma == 0] = np.nan

    # check to convert to db
    if snr_in_db is True:
        snr = linear_to_db(snr)

    # convert noise to variance
    noise_variance = np.square(sigma, dtype=np.float64)

    return snr, noise_variance
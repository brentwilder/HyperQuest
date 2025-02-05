import numpy as np
from spectral import *
from joblib import Parallel, delayed
from skimage.segmentation import slic
from sklearn.decomposition import PCA

from .utils import *
from .mlr import *


def rlsd(hdr_path, block_size, nbins=150, ncpus=1, output_all=False, snr_in_db = False, mask_waterbodies=True):
    '''
    Residual-scaled local standard deviation (Gao et al., 2007)

    Parameters:
        hdr_path (str): Path to the .hdr file.
        block_size (int): Block size for partitioning (for example 5 would be 5x5 pixels).
        nbins (int, optional): Number of bins for histogram analysis. Default is 150.
        ncpus (int, optional): Number of CPUs for parallel processing. Default is 1.
        output_all (bool, optional): Whether to return all outputs. Default is False returing SNR, True returns mu and sigma.
        snr_in_db (bool, optional): Whether SNR is in dB. Default is False.
        mask_waterbodies (bool, optional): Whether to mask water bodies based on NDWI threshold of 0. Default is True.

    Returns:
        out: either an ndarray of SNR, or a tuple containing (mu, sigma, SNR) with respect to wavelength.

    '''

    # Load raster
    img_path = get_img_path_from_hdr(hdr_path)
    array = np.array(envi.open(hdr_path, img_path).load(), dtype=np.float64)

    # mask waterbodies
    if mask_waterbodies is True:
        array = mask_water_using_ndwi(array, hdr_path)
    
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
    w = read_center_wavelengths(hdr_path)
    mu = mask_atmos_windows(mu, w)
    sigma = mask_atmos_windows(sigma, w)

    # division (watching out for zero in denominator)
    out = np.divide(mu, sigma, out=np.zeros_like(mu), where=(sigma != 0))
    out[sigma == 0] = np.nan

    # check to convert to db
    if snr_in_db is True:
        out = linear_to_db(out)
    
    # check to have full output
    if output_all is True:
        out = (mu, sigma, out)

    return out


def ssdc(hdr_path, block_size, nbins=150, ncpus=1, output_all=False, snr_in_db = False, mask_waterbodies=True):
    '''
    Spectral and spatial de-correlation (Roger & Arnold, 1996)

    Parameters:
        hdr_path (str): Path to the .hdr file.
        block_size (int): Block size for partitioning (for example 5 would be 5x5 pixels).
        nbins (int, optional): Number of bins for histogram analysis. Default is 150.
        ncpus (int, optional): Number of CPUs for parallel processing. Default is 1.
        output_all (bool, optional): Whether to return all outputs. Default is False returing SNR, True returns mu and sigma.
        snr_in_db (bool, optional): Whether SNR is in dB. Default is False.
        mask_waterbodies (bool, optional): Whether to mask water bodies based on NDWI threshold of 0. Default is True.

    Returns:
        out: either an ndarray of SNR, or a tuple containing (mu, sigma, SNR) with respect to wavelength.

    '''

    # Load raster
    img_path = get_img_path_from_hdr(hdr_path)
    array = np.array(envi.open(hdr_path, img_path).load(), dtype=np.float64)

    # mask waterbodies
    if mask_waterbodies is True:
        array = mask_water_using_ndwi(array, hdr_path)

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
    w = read_center_wavelengths(hdr_path)
    mu = mask_atmos_windows(mu, w)
    sigma = mask_atmos_windows(sigma, w)

    # division (watching out for zero in denominator)
    out = np.divide(mu, sigma, out=np.zeros_like(mu), where=(sigma != 0))
    out[sigma == 0] = np.nan

    # check to convert to db
    if snr_in_db is True:
        out = linear_to_db(out)
    
    # check to have full output
    if output_all is True:
        out = (mu, sigma, out)

    return out


def hrdsdc(hdr_path, n_segments=200, compactness=0.1, n_pca=3, ncpus=1,
           include_neighbor_pixel_in_mlr=True, output_all=False, 
           snr_in_db=False, mask_waterbodies=True):
    '''
    Homogeneous regions division and spectral de-correlation (Gao et al., 2008)

    Parameters:
        hdr_path (str): Path to the .hdr file.
        n_segments (int):  The (approximate) number of labels in the segmented output image. see skimage.segmentation.slic for more.
        compactness (float):Balances color proximity and space proximity. Higher values give more weight to space proximity, making superpixel shapes more square/cubic.see skimage.segmentation.slic for more.
        n_pca (int): Number of PCAs to compute and provide to SLIC segmentation.
        ncpus (int, optional): Number of CPUs for parallel processing. Default is 1.
        include_neighbor_pixel_in_mlr (bool, optional): If True, neighbor pixel is used in MLR (for k`). Else, MLR only contains spectral data (k+1, k-1).
        output_all (bool, optional): Whether to return all outputs. Default is False returing SNR, True returns mu and sigma.
        snr_in_db (bool, optional): Whether SNR is in dB. Default is False.
        mask_waterbodies (bool, optional): Whether to mask water bodies based on NDWI threshold of 0. Default is True.

    Returns:
        out: either an ndarray of SNR, or a tuple containing (mu, sigma, SNR) with respect to wavelength.

    '''


    # Load raster
    img_path = get_img_path_from_hdr(hdr_path)
    array = np.array(envi.open(hdr_path, img_path).load(), dtype=np.float64)

    # mask waterbodies
    if mask_waterbodies is True:
        array = mask_water_using_ndwi(array, hdr_path)

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
        test_segment = test_segment[test_segment[:, 0] > -99]
        if test_segment.shape[0] != 0:
            return test_segment
        else:
            return None
    segment_data = Parallel(n_jobs=ncpus)(delayed(process_segment)(u) for u in unique_segments)
    segment_data = [seg for seg in segment_data if seg is not None]

    # Parallel processing of all segments depending on method selected
    if include_neighbor_pixel_in_mlr == False:
        # Perform just spectral MLR
        results = Parallel(n_jobs=ncpus, 
                           timeout=None)(delayed(mlr_spectral)(segment) for segment in segment_data)
        
    else: # perform spectral-spatial MLR using k` nearby neighbor.
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
    w = read_center_wavelengths(hdr_path)
    mu = mask_atmos_windows(mu, w)
    sigma = mask_atmos_windows(sigma, w)

    # Compute SNR
    out = np.divide(mu, sigma, out=np.zeros_like(mu), where=(sigma != 0))
    out[sigma == 0] = np.nan

    # check to convert to db
    if snr_in_db is True:
        out = linear_to_db(out)

    # Output full results if requested
    if output_all:
        out = (mu, sigma, out)

    return out
import numpy as np
import rasterio
import rasterio.mask
from joblib import Parallel, delayed
from skimage.segmentation import slic
from sklearn.decomposition import PCA

from .utils import *



def rlsd(img_path, block_size, nbins=150, ncpus=1, output_all=False, 
         snr_in_db = False, mask_waterbodies=True):
    '''
    TODO

    ndwi bands start from 1 .. can also use helper function


    
    '''
    # Load raster
    with rasterio.open(img_path) as src:
        array = src.read()

    # mask waterbodies
    if mask_waterbodies is True:
        array = mask_water_using_ndwi(array, img_path)
    
    # Pad image to ensure divisibility by block_size
    array = pad_image(array, block_size)

    # get tasks (number of blocks)
    tasks = get_blocks(array, block_size)
    
    # Parallel processing of blocks using joblib
    results = Parallel(n_jobs=ncpus)(delayed(block_regression_spectral)(block) for block in tasks)

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






def ssdc(img_path, block_size, nbins=150, ncpus=1, output_all=False, 
         snr_in_db = False, mask_waterbodies=True):
    '''
    TODO
    
    '''
    # Load raster
    with rasterio.open(img_path) as src:
        array = src.read()

    # mask waterbodies
    if mask_waterbodies is True:
        array = mask_water_using_ndwi(array, img_path)

    # Pad image to ensure divisibility by block_size
    array = pad_image(array, block_size)

    # get tasks (number of blocks)
    tasks = get_blocks(array, block_size)
    
    # Parallel processing of blocks using joblib
    results = Parallel(n_jobs=ncpus)(delayed(block_regression_spectral_spatial)(block) for block in tasks)

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



def hrdsdc(img_path,n_segments=200, 
           compactness=0.1, n_pca = 3, ncpus=1, 
           output_all=False, snr_in_db=False, mask_waterbodies=True):
    """
    TODO
    """
    # Load raster
    with rasterio.open(img_path) as src:
        array = src.read() 

    # mask waterbodies
    if mask_waterbodies is True:
        array = mask_water_using_ndwi(array, img_path)

    # Rearrange to (rows, cols, bands) for segmentation
    array = np.moveaxis(array, 0, -1)

    # Apply PCA 
    pca = PCA(n_components=n_pca)
    rows, cols, bands = array.shape
    array_reshaped = array.reshape(-1, bands)
    array_pca = pca.fit_transform(array_reshaped).reshape(rows, cols, -1)
    
    # SLIC
    segments = slic(array_pca, 
                    n_segments=n_segments, 
                    compactness=compactness)


    # Plot the clusters overlaid on the image
    #import matplotlib.pyplot as plt
    #from skimage.segmentation import mark_boundaries
    #plt.figure(figsize=(10, 10))
    #plt.imshow(mark_boundaries(array_pca[:,:,0:3], segments, color=(0,0,0)))
    #plt.title("SLIC Clusters")
    #plt.axis("off")
    #plt.show()

    # Process each segment
    unique_segments = np.unique(segments)
    
    # function here to access array in multiprocessing
    def process_segment(segment_id):
        mask = (segments == segment_id)
        segment_data = array[mask]  
        
        mu_segment = np.full(segment_data.shape[1], np.nan)
        sigma_segment = np.full(segment_data.shape[1], np.nan)
        
        for k in range(1, segment_data.shape[1] - 1):
            X = np.vstack((segment_data[:, k - 1], segment_data[:, k + 1])).T
            y = segment_data[:, k]
            
            if not np.any(np.isnan(y)):
                valid_mask_x = ~np.isnan(X[:, 0]) & ~np.isnan(X[:, 1])
                X_valid = X[valid_mask_x]
                y_valid = y[valid_mask_x]

                if len(y_valid) > 50: # from Gao, at least 50
                    coef, _ = nnls(X_valid, y_valid)
                    y_pred = X_valid @ coef
                    # 3 DOF because of MLR
                    sigma_segment[k] = np.nanstd(y_valid - y_pred, ddof=3)
                    mu_segment[k] = np.nanmean(y_valid)

        return mu_segment, sigma_segment

    # Parallel processing of all segments
    results = Parallel(n_jobs=ncpus)(delayed(process_segment)(seg_id) for seg_id in unique_segments)

    # Aggregate results
    local_mu = np.array([res[0] for res in results])
    local_sigma = np.array([res[1] for res in results])

    # Average over segments for each band
    # first and last are empty due to k-1 k+1 in regression...
    mu_valid = np.nanmean(local_mu[:, 1:-1], axis=0)
    sigma_valid = np.nanmean(local_sigma[:, 1:-1], axis=0)
    mu = np.concatenate(([np.nan], mu_valid, [np.nan]))
    sigma = np.concatenate(([np.nan], sigma_valid, [np.nan]))

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
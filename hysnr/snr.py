import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from joblib import Parallel, delayed

from hysnr.utils import *



def rlsd(img_path, block_size, nbins=150, ncpus=1, output_all=False, snr_in_db = False):
    '''
    TODO

    SNR (or tuple of Signal,Noise,SNR)
    
    '''
    # Load raster
    with rasterio.open(img_path) as src:
        array = src.read()

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






def ssdc(img_path, block_size, nbins=150, ncpus=1, output_all=False, snr_in_db = False):
    '''
    TODO
    
    '''

    return




def hrdsdc(img_path, segmentation_method='slic', ncpus=1, output_all=False, snr_in_db = False):
    '''
    TODO
    
    '''
    return
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.linear_model import LinearRegression
import rasterio
import rasterio.mask
from joblib import Parallel, delayed

from common import *


def homogeneous_area(img_path, geometry, output_all=False, snr_in_db = False):
    """
    Signal-to-Noise Ratio (SNR) for each region of interest using basic homogeneous area.
    This is the simplest method and should be used with extreme caution.

    Args:
    img_path: Path to the hyperspectral image
    geometry: geometry from geopandas (e.g., gdf['geometry'])

    Returns:
    SNR (or tuple of Signal,Noise,SNR)
    """
    # Load raster and geometry data
    with rasterio.open(img_path) as src:

        # clip image
        out_image, _ = rasterio.mask.mask(src, geometry, crop=True, nodata=-9999)

        # Flatten
        flat = out_image.reshape(out_image.shape[0], -1).T

        # Remove NoData values
        flat = flat[flat[:, -1] != -9999]

        # Calculate mean (signal) and std (noise) for each band
        mu = np.nanmean(flat, axis=0)
        sigma = np.nanstd(flat, axis=0)

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




def geostatistical(raster, geometry, geometry_attribute):
    '''
    TODO
    
    '''
    return






def lmlsd(img_path, block_size, nbins=150, ncpus=1, output_all=False, snr_in_db = False):
    """
    Compute local mean and standard deviation-based SNR for hyperspectral data.

    Parameters:
    img_path (str): Path to the hyperspectral image.
    block_size (int): Size of the square blocks (e.g., 3 for (285, 3, 3)).
    nbins (int): Number of bins.
    ncpus (int): Number of parallel processes.

    Returns:
    pd.DataFrame: DataFrame with wavelength and SNR.
    """
    # Load raster and geometry data
    with rasterio.open(img_path) as src:
        array = src.read()

    # Pad image to ensure divisibility by block_size
    array = pad_image(array, block_size)

    # get tasks (number of blocks)
    tasks = get_blocks(array, block_size)
    
    # Parallel processing of blocks using joblib
    results = Parallel(n_jobs=ncpus)(delayed(block_stats)(block) for block in tasks)

    # Create empty lists
    local_mu = []
    local_sigma = []

    # Collect results
    for block_idx, (m, s) in enumerate(results):
        if m is not None and s is not None:
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




def rlsd(img_path, block_size, nbins=150, ncpus=1):
    '''
    TODO
    
    '''
     # Load raster and geometry data
    with rasterio.open(img_path) as src:
        array = src.read()
    wavelengths = read_center_wavelength(img_path)

    # Initialize dataframe
    df = pd.DataFrame(data=wavelengths, columns=['Wavelength'])

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
        if m is not None and s is not None:
            local_mu.append(m)
            local_sigma.append(s)
    local_mu = np.array(local_mu)
    local_sigma = np.array(local_sigma)

    # Bin and compute SNR
    signal, noise = binning(local_mu, local_sigma, wavelengths, nbins)

    # Add to dataframe
    df['Signal'] = signal
    df['Noise'] = noise
    df['SNR'] = df.Signal / df.Noise

    # Set index
    df = df.set_index('Wavelength')

    return df
















def ssdc(data, block_size, nbins=150):
    '''
    TODO
    
    '''
    #Load user data
    datacube = load_image(data)
    wavelengths = datacube.bands.centers

    # start dataframe
    df = pd.DataFrame(data=wavelengths, columns=['Wavelength'])

    # convert datacube to numpy array
    array = datacube.open_memmap(writeable=False)

    # block it 
    array = pad_image(array, block_size)
    blocked_image = view_as_blocks(array, (block_size, block_size, wavelengths))
    
    # loop through each block and compute mu local and sigmal local for all wavelengths
    local_sigma_dict = {w: [] for w in wavelengths}
    local_mu_dict = {w: [] for w in wavelengths}
    
    for i in range(blocked_image.shape[0]):
        for j in range(blocked_image.shape[1]):
            ij = blocked_image[i,j,:]
            #ij = ij.reshape(ij.shape[0], -1) 
            #ij = ij[:, ~np.isnan(ij).any(axis=0)]

            # TODO:
            # can't flatten because need to retain neighbor pixel...

            # Skip empty blocks
            if ij.size == 0:
                continue

     

                # Store values for this wavelength
                #local_sigma_dict[wavelengths[k]].append(sigma_local)
                #local_mu_dict[wavelengths[k]].append(mu_local)

    # bin 
    signal, noise = binning(local_mu_dict, local_mu_dict, wavelengths)

    # compute ratio
    snr = signal / noise
    df['SNR'] = snr

    return df



def hrdsdc(raster, segmentation_method):
    '''
    TODO
    
    '''
    return
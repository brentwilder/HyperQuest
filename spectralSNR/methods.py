import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
from skimage.util import view_as_blocks
from sklearn.linear_model import LinearRegression

from utils import *


def homogeneous_area(data, geometry_path, geometry_attribute):

    """
    Calculate Signal-to-Noise Ratio (SNR) for each region of interest (ROI) in the shapefile, 
    based on the mean and standard deviation of pixel intensities from the raster data.
    
    Args:
    data: Path to the hyperspectral envi data or ENVI data
    geometry_path: Path to the geometry (shp, json, etc).
    geometry_attribute: Attribute that uniquely identifies each region.
    
    Returns:
    A pandas DataFrame with SNR with respect to wavelengths
    """

    #Load user data
    datacube = load_data(data)
    roi = gpd.read_file(geometry_path)
    wavelengths = datacube.bands.centers

    # start dataframe
    df = pd.DataFrame(data=wavelengths, column='Wavelength')

    # Loop through each ROI in the shapefile
    for _, region in roi.iterrows():
        region_geometry = region['geometry']
        region_id = region[geometry_attribute]
        
        # Mask raster with the current region's geometry
        out_image, _ = mask(datacube, [region_geometry], crop=True)
        flat_image = out_image.reshape(out_image.shape[0], -1)  # Flatten for each band

        # Remove NoData values
        flat_image = flat_image[:, ~np.isnan(flat_image).any(axis=0)]

        # Calculate mean (signal) and std (noise) for each band
        mu = np.nanmean(flat_image, axis=1)
        sigma = np.nanstd(flat_image, axis=1)
        snr = mu / sigma

        # Add to dataframe
        df[f'{region_id}_SNR'] = snr

    return df



def geostatistical(raster, geometry, geometry_attribute):
    '''
    TODO
    
    '''
    return




def lmlsd(data, block_size, nbins=150):
    '''
    TODO

    A region-based block compressive sensing algorithm for plant hypers
    
    '''
    #Load user data
    datacube = load_data(data)
    wavelengths = datacube.bands.centers

    # start dataframe
    df = pd.DataFrame(data=wavelengths, column='Wavelength')

    # convert datacube to numpy array
    array = None

    # block it 
    array = pad_image(array, block_size)
    blocked_image = view_as_blocks(array, (block_size, block_size, wavelengths))
    
    # loop through each block and compute mu local and sigmal local for all wavelengths
    local_sigma_dict = {w: [] for w in wavelengths}
    local_mu_dict = {w: [] for w in wavelengths}
    for i in range(blocked_image.shape[0]):
        for j in range(blocked_image.shape[1]):
            ij = blocked_image[i,j,:]
            ij = ij.reshape(ij.shape[0], -1) 
            ij = ij[:, ~np.isnan(ij).any(axis=0)]
            mu_local = np.nanmean(ij, axis=1)
            sigma_local = np.nanstd(ij, axis=1)
            for k, w in enumerate(wavelengths):
                local_sigma_dict[w].append(sigma_local[k])
                local_mu_dict[w].append(mu_local[k])

    # bin 
    signal, noise = binning(local_mu_dict, local_mu_dict, wavelengths)

    # compute ratio
    snr = signal / noise
    df['SNR'] = snr
    
    return df






def rlsd(data, block_size, nbins=150):
    '''
    TODO
    
    '''
    #Load user data
    datacube = load_data(data)
    wavelengths = datacube.bands.centers

    # start dataframe
    df = pd.DataFrame(data=wavelengths, column='Wavelength')

    # convert datacube to numpy array
    array = None

    # block it 
    array = pad_image(array, block_size)
    blocked_image = view_as_blocks(array, (block_size, block_size, wavelengths))
    
    # loop through each block and compute mu local and sigmal local for all wavelengths
    local_sigma_dict = {w: [] for w in wavelengths}
    local_mu_dict = {w: [] for w in wavelengths}
    
    for i in range(blocked_image.shape[0]):
        for j in range(blocked_image.shape[1]):
            ij = blocked_image[i,j,:]
            ij = ij.reshape(ij.shape[0], -1) 
            ij = ij[:, ~np.isnan(ij).any(axis=0)]

            # Skip empty blocks
            if ij.size == 0:
                continue

            # Iterate through wavelengths, skipping first and last
            for k in range(1, len(wavelengths) - 1):
                #x* = a + bx(k-1) + cx(k+1)... 
                X = np.vstack((ij[k - 1], ij[k + 1])).T 
                y = ij[k]
                reg = LinearRegression()
                reg.fit(X, y)
                y_pred = reg.predict(X)
                variance = np.var(np.sum((y - y_pred)**2))
                sigma_local = np.sqrt(variance)
                mu_local = np.mean(y)

                # Store values for this wavelength
                local_sigma_dict[wavelengths[k]].append(sigma_local)
                local_mu_dict[wavelengths[k]].append(mu_local)

    # bin 
    signal, noise = binning(local_mu_dict, local_mu_dict, wavelengths)

    # compute ratio
    snr = signal / noise
    df['SNR'] = snr
  

    return df






def ssdc(data, block_size, nbins=150):
    '''
    TODO
    
    '''
  #Load user data
    datacube = load_data(data)
    wavelengths = datacube.bands.centers

    # start dataframe
    df = pd.DataFrame(data=wavelengths, column='Wavelength')

    # convert datacube to numpy array
    array = None

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
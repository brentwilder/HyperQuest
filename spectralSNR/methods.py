import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
from skimage.util import view_as_blocks

from common import *


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




def lmlsd(data, block_size, nbins):
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
            ij = ij[:, ~np.isnan(ij).any(axis=0)]
            mu_local = np.nanmean(ij, axis=1)
            sigma_local = np.nanstd(ij, axis=1)
            for k, w in enumerate(wavelengths):
                local_sigma_dict[w].append(sigma_local[k])
                local_mu_dict[w].append(mu_local[k])

    signal = []
    noise = []

    for w in wavelengths:
        lsd_values = np.array(local_sigma_dict[w])
        lmu_values = np.array(local_mu_dict[w])
        
        # Create bins based on min max
        bin_min = np.nanmin(lsd_values)
        bin_max = np.nanmax(lsd_values)
        bin_edges = np.linspace(bin_min, bin_max, nbins+1)
        
        # Count blocks in bins
        bin_counts, _ = np.histogram(lsd_values, bins=bin_edges)
        
        # Find the bin with the most blocks
        max_bin_idx = np.argmax(bin_counts)
        selected_bin_min = bin_edges[max_bin_idx]
        selected_bin_max = bin_edges[max_bin_idx + 1]

        # Get LSD values in the selected bin
        selected_sd = lsd_values[(lsd_values >= selected_bin_min) & (lsd_values < selected_bin_max)]
        selected_mu = lmu_values[(lsd_values >= selected_bin_min) & (lsd_values < selected_bin_max)]
        
        # Calculate means
        noise_value = np.nanmean(selected_sd)
        noise.append(noise_value)
        signal_value = np.nanmean(selected_mu)
        signal.append(signal_value)

    # to array
    signal = np.array(signal)
    noise = np.array(noise)

    # compute ratio
    snr = signal / noise
    df['SNR'] = snr
    
    return df






def rlsd(raster, block_size):
    '''
    TODO
    
    '''
    return



def ssdc(raster, block_size):
    '''
    TODO
    
    '''
    return



def hrdsdc(raster, segmentation_method):
    '''
    TODO
    
    '''
    return
import numpy as np

import geopandas as gpd
import rasterio
import pandas as pd
import numpy as np
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
    datacube = load_envi(data)
    roi = gpd.read_file(geometry_path)
    wavelengths = datacube.wavelengths

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
        mean_signal = np.mean(flat_image, axis=1)
        std_noise = np.std(flat_image, axis=1)
        snr = mean_signal / std_noise

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
    
    '''
    #Load user data
    datacube = load_envi(data)
    wavelengths = datacube.wavelengths

    # start dataframe
    df = pd.DataFrame(data=wavelengths, column='Wavelength')

    # convert datacube to numpy array
    array = None

    # block it 
    array = pad_image(array, block_size)
    blocked_image = view_as_blocks(array, (block_size, block_size, wavelengths))
   


    return


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
import numpy as np
import pandas as pd
import geopandas as gpd
from skimage.util import view_as_blocks
from sklearn.linear_model import LinearRegression
import rioxarray as rio

from utils import *


def homogeneous_area(data, geometry_path, geometry_attribute):

    """
    Signal-to-Noise Ratio (SNR) for each region of interest using basic homogeneous area.
    This is the simplest method and should be used with extreme caution.
    
    Args:
    data: Path to the hyperspectral envi data
    geometry_path: Path to the geometry (shp, json, etc).
    geometry_attribute: Attribute that uniquely identifies each region.
    
    Returns:
    A pandas DataFrame with SNR with respect to wavelengths
    """
    
    #Load user data
    src = rio.open_rasterio(data)
    wavelengths = read_center_wavelength(data)
    roi = gpd.read_file(geometry_path)
    roi = roi.to_crs(src.rio.crs) #ensure correct CRS

    # start dataframe
    df = pd.DataFrame(data=wavelengths, columns=['Wavelength'])

    # Loop through each ROI in the shapefile
    for _, r in roi.iterrows():
        region_id = r[geometry_attribute]

        # Mask raster with the current region's geometry
        out = src.rio.clip([r['geometry']]).values
        flat = out.reshape(out.shape[0], -1).T

        # Remove NoData values
        flat = flat[flat[:, -1] != -9999]

        # Calculate mean (signal) and std (noise) for each band
        mu = np.nanmean(flat, axis=0)
        sigma = np.nanstd(flat, axis=0)
        snr = mu / sigma

        # Add to dataframe
        df[f'{region_id}_Signal'] = mu
        df[f'{region_id}_Noise'] = sigma
        df[f'{region_id}_SNR'] = snr
    
    # set index
    df = df.set_index('Wavelength')

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
    array = rio.open_rasterio(data).values
    wavelengths = read_center_wavelength(data)

    # start dataframe
    df = pd.DataFrame(data=wavelengths, columns=['Wavelength'])

    # block it 
    array = pad_image(array, block_size)

    blocked_image = array.reshape((block_size,
                                  array.shape[1]// block_size,
                                  block_size,
                                  array.shape[2] // block_size,
                                  array.shape[0])).swapaxes(1, 2)

    # loop through each block and compute mu local and sigmal local for all wavelengths
    local_sigma_dict = {w: [] for w in wavelengths}
    local_mu_dict = {w: [] for w in wavelengths}
    for i in range(blocked_image.shape[2]):
        for j in range(blocked_image.shape[3]):
            ij = blocked_image[:,:,i,j,:]
            ij = ij.reshape(-1, ij.shape[2])
            print(ij.shape)
            ij = ij[ij[:, 0] != -9999]
            print(ij.shape)
            if ij.shape[0] != block_size**2: #edge NaN case
                pass
            else:
                mu_local = np.nanmean(ij, axis=0)
                print(mu_local)
                break
                sigma_local = np.nanstd(ij, axis=0)
                for k, w in enumerate(wavelengths):
                    local_sigma_dict[w].append(sigma_local[k])
                    local_mu_dict[w].append(mu_local[k])

    # bin 
    signal, noise = binning(local_mu_dict, local_mu_dict, wavelengths, nbins)

    # compute ratio
    snr = signal / noise
    df['SNR'] = snr
    
    return df






def rlsd(data, block_size, nbins=150):
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
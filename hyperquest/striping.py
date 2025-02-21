import numpy as np
from spectral import *

from .utils import *


def sigma_threshold(path_to_data, rotate, sigma_multiplier = 3):
    '''
    Uses a sigma threshold counting neighboring pixels to determine if striping is present, as presented in, 

    Yokoya 2010, Preprocessing of hyperspectral imagery with consideration of smile and keystone properties,

    NOTE: similar to Smile methods, this assumes you have the data (or know the rotation) so that cross-track corresponds correctly.

    Parameters
    ----------
    path_to_data : str
        Path to the .hdr or .nc
    rotate : int
        rotate counter clockwise, either 0, 90, 180, or 270.
    sigma_multiplier : int, float, optional 
        levels of sigma for threshold.

    Returns
    -------
    s : ndarray
        array of same shape of image where 0 is no stripe and 1 is classified as stripe.
    
    '''

    # Identify data type
    if path_to_data.lower().endswith('.nc'):
        image, fwhm, w, obs_time = retrieve_data_from_nc(path_to_data)
    else:
        # Load raster
        img_path = get_img_path_from_hdr(path_to_data)
        image = np.array(envi.open(path_to_data, img_path).load(), dtype=np.float64)

    # Ensure 2d
    if len(image.shape) != 2:
        raise Exception('Data needs to be a 2D array.')

    # ensure this is the raw data and has not been georeferenced.
    if (image[:,:]<0).sum() > 0:
        raise Exception('Please provide data that has not been geo-referenced.')
    
    # Perform rotation if needed
    if rotate != 0 and rotate != 90 and rotate != 180 and rotate != 270:
        raise ValueError('rotate must be 90, 180, or 270.')
    if rotate>=90:
        image = np.rot90(image, axes=(0,1), k=1)
    elif rotate>=180:
        image = np.rot90(image, axes=(0,1), k=2)
    elif rotate>=270:
        image = np.rot90(image, axes=(0,1), k=3)

    # create striping mask
    s = np.full_like(image, fill_value=0)

    # prep
    c_list = []
    # columns
    for j in range(image.shape[1]):

        # reset c on next column
        c = 0
        # rows
        for i in range(image.shape[0]):

            # skip first and last (missing ends)
            if j == 0 or j == image.shape[1] - 1:
                pass
            else:
                # extracting data from image
                x_ijk = image[i, j]
                x_left = image[i, j - 1]
                x_right = image[i, j + 1]

                # EQ 1 in Yokoya 2010
                if (x_ijk < x_left and x_ijk < x_right) or (x_ijk > x_left and x_ijk > x_right):
                    c += 1
        # store c at end of row
        c_list.append(c)

    # turn c  into array
    c_array = np.array(c_list, dtype=np.uint32)

    # compute std dev of c
    sigma = np.nanstd(c_array)
    mean = np.nanmean(c_array)
    threshold = mean + sigma*sigma_multiplier

    # for each i, assess if stripe
    for j in range(c_array.shape[0]):
        if np.abs(c_array[j]) > threshold:
            s[:, j] = 1

    return s

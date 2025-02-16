import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import re
from os.path import abspath, exists
from dateutil import parser
from datetime import timezone, datetime
import numpy as np
from spectral import *
import h5netcdf

def sigma_theshold(image, rotate, sigma_multiplier = 3, no_data_value=-9999):
    '''
    TODO
    Yokoya 2010, Preprocessing of hyperspectral imagery with consideration of smile and keystone properties

    Built out to also work with georectified imagery so we can assess quality of actual TOA radiance products.

    '''

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

    if len(image.shape) != 2:
        raise Exception('Data needs to be a 2D array.')

    # Mask no data values
    image[image <= no_data_value] = np.nan

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

                #plt.imshow(image)
                #plt.scatter(j - 1, i)
                #plt.scatter(j + 1 , i)
                #plt.scatter(j, i,  color='k')
                #plt.show()

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
    plt.hist(c_array)
    plt.show()

    # for each i, assess if stripe
    for j in range(c_array.shape[0]):
        if np.abs(c_array[j]) > threshold:
            s[:, j] = 1

    return s


def retrieve_data_from_nc(path_to_data):
    '''
    TODO:

    NOTE: keys are specific to sensor. right now, only EMIT uses NetCDF format that I know.. and the key is "radiance".
    '''

    # get absolute path 
    path_to_data = abspath(path_to_data)

    # read using h5netcdf
    ds = h5netcdf.File(path_to_data, mode='r')

    # get radiance and fhwm and wavelength
    array = np.array(ds['radiance'], dtype=np.float64)
    fwhm = np.array(ds['sensor_band_parameters']['fwhm'][:].data.tolist(), dtype=np.float64)
    wave = np.array(ds['sensor_band_parameters']['wavelengths'][:].data.tolist(), dtype=np.float64)

    obs_time = datetime.strptime(ds.attrs['time_coverage_start'], '%Y-%m-%dT%H:%M:%S+0000')

    return array, fwhm, wave, obs_time




from spectral import *

# Define path to envi image header file
path_to_data = './tests/data/EMIT_L1B_RAD_001_20220827T091626_2223906_009.nc'

array, fwhm, wave, obs_time = retrieve_data_from_nc(path_to_data)

array = array[:,:,15]

s = sigma_theshold(array, rotate=0, sigma_multiplier=3)

plt.imshow(s)
plt.show()

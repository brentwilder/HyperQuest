import numpy as np
from scipy.optimize import nnls
import re
from os.path import abspath, exists


def binning(local_mu, local_sigma, nbins):
    '''

    TODO

    computes signal and noise using histogram/binning method

    '''

    signal = np.full_like(local_mu[0,:], np.nan)
    noise = np.full_like(local_mu[0,:], np.nan)

    # Process each wavelength
    for idx in range(len(signal)):
        # Get LSD and mean values for this wavelength
        lsd_values = local_sigma[:, idx]
        lmu_values = local_mu[:, idx]

        # Create bins based on LSD values
        if np.all(np.isnan(lsd_values)):
            continue

        bin_min = np.nanmin(lsd_values)
        bin_max = np.nanmax(lsd_values)
        bin_edges = np.linspace(bin_min, bin_max, nbins)

        # Count blocks in each bin
        bin_counts, _ = np.histogram(lsd_values, bins=bin_edges)

        # Identify the bin with the highest count
        max_bin_idx = np.argmax(bin_counts)
        selected_bin_min = bin_edges[max_bin_idx]
        selected_bin_max = bin_edges[max_bin_idx + 1]

        # Filter LSD and mean values within the selected bin
        mask = (lsd_values >= selected_bin_min) & (lsd_values < selected_bin_max)
        selected_sd = lsd_values[mask]
        selected_mu = lmu_values[mask]

        # Compute noise (mean of selected standard deviations)
        noise[idx] = np.nanmean(selected_sd)

        # Compute signal (mean of selected mean values)
        signal[idx] = np.nanmean(selected_mu)

    return signal.astype(float), noise.astype(float)





def block_regression_spectral(block):
    '''
    TODO:
    for each NxN block, perform regression on k+1, k-1 spectral data

    '''
    # Assume no data value TODO
    block = block.astype(float)
    block[block <= -999] = np.nan

    # create empty arrays
    mu_local = np.full(block.shape[1], np.nan) 
    sigma_local = np.full(block.shape[1], np.nan)  

    # loop through all but first and last band
    for k in range(1, block.shape[1] - 1):
        X = np.vstack((block[:, k - 1], block[:, k + 1])).T 
        y = block[:, k] 

        # If y is valid, proceed
        if not np.any(np.isnan(y)):
            # Create a mask for valid (non-NaN) data points in X
            valid_mask_x = ~np.isnan(X[:, 0]) & ~np.isnan(X[:, 1])
            X_valid = X[valid_mask_x]
            y_valid = y[valid_mask_x]

            # regression on enough data
            if len(y_valid) > 3: 
                coef, _ = nnls(X_valid, y_valid)
                y_pred = X_valid @ coef

                # Calculate residuals and mean
                # wxh -3 because of dof, see the following,
                # "Residual-scaled local standard deviations method for estimating noise in hyperspectral images"
                sigma_local[k] = np.nanstd(y_valid - y_pred, ddof=3)
                mu_local[k] = np.mean(y_valid)

    return mu_local, sigma_local





def block_regression_spectral_spatial(block):
    '''
    TODO:
    for each NxN block, perform regression on k+1 (same pixel), k-1 (same pixel), and k from nearby pixel. 

    '''
 
    # Assume no data value TODO
    block = block.astype(float)
    block[block <= -999] = np.nan

    # create empty arrays
    mu_local = np.full(block.shape[1], np.nan) 
    sigma_local = np.full(block.shape[1], np.nan)  

    for k in range(1, block.shape[1] - 1):

        X = np.vstack((block[:, k - 1], block[:, k + 1])).T 
        neighbor_k = np.roll(block[:, k], shift=1)  # Shift 1 to find a neighbor pixel
        X = np.column_stack((X, neighbor_k))
        y = block[:, k] 

        # If y is valid, proceed
        if not np.any(np.isnan(y)):
            # Create a mask for valid (non-NaN) data points in X
            valid_mask_x = ~np.isnan(X[:, 0]) & ~np.isnan(X[:, 1])
            X_valid = X[valid_mask_x]
            y_valid = y[valid_mask_x]

            # regression on enough data
            if len(y_valid) > 3: 
                coef, _ = nnls(X_valid, y_valid)
                y_pred = X_valid @ coef

                # Calculate residuals and mean
                # wxh -4 
                sigma_local[k] = np.nanstd(y_valid - y_pred, ddof=4)
                mu_local[k] = np.mean(y_valid)

    return mu_local, sigma_local



def pad_image(image, block_size):
    '''
    TODO:
    pads image for NxN blocking to be allowed.

    '''
    bands, height, width = image.shape
    
    # Calculate padding for height and width
    pad_height = (block_size - (height % block_size)) % block_size
    pad_width = (block_size - (width % block_size)) % block_size

    padding = [(0, 0),  
                (0, pad_height), 
                (0, pad_width)] 


    # Apply padding 
    padded_image = np.pad(image, padding, 
                          mode='constant', 
                          constant_values=-9999)

    return padded_image


def get_blocks(array, block_size):
    '''
    TODO:
    provides the full array of blocks based on NxN size.

    '''

    # Reshape into blocks
    blocked_image = array.reshape(
        array.shape[0],  # Number of bands
        array.shape[1] // block_size, block_size,  # Rows into blocks
        array.shape[2] // block_size, block_size   # Columns into blocks
    ).swapaxes(1, 2)  # Swap to ensure consistent block ordering

    # Flatten into tasks for processing (each task represents one pixel block)
    blocks = blocked_image.reshape(
        blocked_image.shape[0],  # Number of bands
        -1,                      # Flatten spatial
        block_size * block_size  # Flatten block size
    ).transpose(1, 2, 0)         # (num_blocks, block_pixels, bands)

    return blocks


def read_center_wavelengths(hdr_path):
    '''
    TODO:
    Reads center wavelengths from the hdr file

    '''
    
    # get absolute path 
    hdr_path = abspath(hdr_path)

    # Raise exception if does not end in .hdr
    if not hdr_path.lower().endswith('.hdr'):
        raise ValueError(f'Invalid file format: {hdr_path}. Expected an .hdr file.')

    wavelength = None
    for line in open(hdr_path, 'r'):
        if 'wavelength =' in line or 'wavelength=' in line: 
            wavelength = re.findall(r"[+-]?\d+\.\d+", line)
            wavelength = ','.join(wavelength)
            wavelength = wavelength.split(',')
            wavelength = np.array(wavelength).astype(float)
    
    return wavelength





def get_img_path_from_hdr(hdr_path):
    '''
    TODO:
    quickly gets actual image path from relative position of .hdr file

    '''
    
    # Ensure the file ends in .hdr
    if not hdr_path.lower().endswith('.hdr'):
        raise ValueError(f'Invalid file format: {hdr_path}. Expected a .hdr file.')

    # If there, get the base path without .hdr
    base_path = hdr_path[:-4]  # Remove last 4 characters (".hdr")

    # get absolute path 
    base_path = abspath(base_path)

    # Possible raster file extensions to check
    raster_extensions = ['.raw', '.img', '.dat', '.bsq', '.bin', ''] 

    # Find which raster file exists
    img_path = None
    for ext in raster_extensions:
        possible_path = base_path + ext
        if exists(possible_path):
            img_path = possible_path
            break

    # if still None, image file was not found.
    if img_path is None:
        raise FileNotFoundError(f"No corresponding image file found for {hdr_path}")
    
    return img_path



def linear_to_db(snr_linear):
    '''
    TODO:
    Convert the SNR to units of dB.

    '''

    snr_db = 10 * np.log10(snr_linear)

    return snr_db
    

def mask_water_using_ndwi(array, img_path, ndwi_threshold=0.25):
    '''
    TODO:
    Returns array where NDWI greater than a threshold are set to -9999 (masked out).

    Reason behind this is that water typically has very very low signal, and therefore different SNR compared to the image.

    It may be a common thing to need to remove water here so this method is called in al of the SNR functions. 

    '''

    wavelengths = read_center_wavelengths(img_path)
    green_index = np.argmin(np.abs(wavelengths - 559))
    nir_index = np.argmin(np.abs(wavelengths - 864))
    green = array[green_index, :, :] 
    nir = array[nir_index, :, :] 
    ndwi = (green - nir) / (green + nir)

    array[:, ndwi > ndwi_threshold] = -9999

    return array


def mask_atmos_windows(value, wavelengths):
    '''
    TODO
    '''
    
    mask = ((wavelengths >= 1250) & (wavelengths <= 1450)) | ((wavelengths >= 1780) & (wavelengths <= 1950))

    value[mask] = np.nan
    
    return value
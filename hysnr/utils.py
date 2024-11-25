import numpy as np
from scipy.optimize import nnls
import re
import os


def binning(local_mu, local_sigma, nbins):
    """
    TODO

    Parameters:
    local_mu (ndarray): xx
    local_sigma (ndarray): xx
    nbins (int): Number of bins for aggregation.

    Returns:
    signal (ndarray): Array of signal values with respect to wavelength
    noise (ndarray): Array of noise values with respect to wavelength.
    """

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
        #import matplotlib.pyplot as plt
        #plt.title(f'Band Number: {idx}')
        #plt.hist(lmu_values[~np.isnan(lmu_values)], bins=nbins, color='red', alpha=0.7)
        #plt.hist(lsd_values[~np.isnan(lsd_values)], bins=nbins, color='k', alpha=0.4)
        #plt.show()

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
    """
    Perform regression for each band in the block. The regression involves
    calculating the relationship between the current band and its neighbors
    (k-1 and k+1).

    Parameters:
    block (ndarray): Block of data.

    Returns:
    tuple: Local mean and standard deviation for the block after regression.
    """
 
    block = block.astype(float)
    block[block == -9999] = np.nan

    mu_local = np.full(block.shape[1], np.nan) 
    sigma_local = np.full(block.shape[1], np.nan)  

    for k in range(1, block.shape[1] - 1):  # Skip the first and last bands
        X = np.vstack((block[:, k - 1], block[:, k + 1])).T  # Neighboring bands
        y = block[:, k]  # Current band

        # If y is valid, proceed
        if not np.any(np.isnan(y)):
            # Create a mask for valid (non-NaN) data points in X
            valid_mask_x = ~np.isnan(X[:, 0]) & ~np.isnan(X[:, 1])  # Check if neither X[k-1] nor X[k+1] is NaN

            # Apply the mask to X and y to remove rows with NaN values
            X_valid = X[valid_mask_x]
            y_valid = y[valid_mask_x]

            # Perform regression
            if len(y_valid) > 3:  # need enough data for regression, 
                coef, _ = nnls(X_valid, y_valid)
                y_pred = X_valid @ coef  # Predicted values

                # Calculate residuals and mean
                # wxh -3 because of dof, see the following,
                # "Residual-scaled local standard deviations method for estimating noise in hyperspectral images"
                sigma_local[k] = np.sqrt(((1/(block.shape[1]- 3)) * np.sum((y_valid - y_pred)**2)))
                mu_local[k] = np.mean(y_valid)

                #import matplotlib.pyplot as plt
                #plt.scatter(y, y_pred, alpha=0.6, color='blue', label=f'Band {k}')
                #plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Fit')
                #plt.xlabel('Observed (y)')
                #plt.ylabel('Predicted (y_pred)')
                #plt.title(f'Band {k}, mu:{np.nanmean(y_valid)}, sd:{np.nanstd(y_valid - y_pred)}')
                #plt.legend()
                #plt.grid(True)
                #plt.show()

    return mu_local, sigma_local





def pad_image(image, block_size):
    """
    Pads the image to ensure the dimensions are divisible by block_size, using np.nan for padding.
    """
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





def read_center_wavelengths(img_path):
    '''
    TODO
    '''

    # Remove any existing extension (e.g., .img, .bsq) 
    base_name, _ = os.path.splitext(img_path)
    
    # Add .hdr extension
    hdr_path = base_name + '.hdr'

    wavelength = None
    for line in open(hdr_path, 'r'):
        if 'wavelength =' in line or 'wavelength=' in line: 
            wavelength = re.findall(r"[+-]?\d+\.\d+", line)
            wavelength = ','.join(wavelength)
            wavelength = wavelength.split(',')
            wavelength = np.array(wavelength)
    
    return wavelength




def linear_to_db(snr_linear):
    '''
    TODO
    
    '''
    snr_db = 10 * np.log10(snr_linear)

    return snr_db
    
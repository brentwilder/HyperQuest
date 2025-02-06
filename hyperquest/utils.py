import numpy as np
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


def pad_image(image, block_size):
    '''
    TODO:
    pads image for NxN blocking to be allowed.

    '''
    rows, cols, bands = image.shape

    pad_rows = (block_size - (rows % block_size)) % block_size
    pad_cols = (block_size - (cols % block_size)) % block_size

    padded_image = np.full((rows + pad_rows, cols + pad_cols, bands), -9999, dtype=np.float64)
    padded_image[:rows, :cols, :] = image  

    return padded_image


def get_blocks(array, block_size):
    '''
    TODO:
    provides the full array of blocks based on NxN size.

    '''
    rows, cols, bands = array.shape

    # Reshape into blocks
    blocked_image = array.reshape(
        rows // block_size, block_size,
        cols // block_size, block_size,
        bands
        ).swapaxes(1, 2)

    # Flatten 
    blocks = blocked_image.reshape(-1, block_size * block_size, bands)

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
        if 'wavelength =' in str.lower(line) or 'wavelength=' in str.lower(line): 
            wavelength = re.findall(r"[+-]?\d+\.\d+", line)
            wavelength = ','.join(wavelength)
            wavelength = wavelength.split(',')
            wavelength = np.array(wavelength).astype(float)
    
    return wavelength


def read_fwhm(hdr_path):
    '''
    TODO:
    Reads FWHM from the hdr file

    '''
    
    # get absolute path 
    hdr_path = abspath(hdr_path)

    # Raise exception if does not end in .hdr
    if not hdr_path.lower().endswith('.hdr'):
        raise ValueError(f'Invalid file format: {hdr_path}. Expected an .hdr file.')

    fwhm = None
    for line in open(hdr_path, 'r'):
        if 'fwhm =' in str.lower(line) or 'fwhm=' in str.lower(line): 
            fwhm = re.findall(r"[+-]?\d+\.\d+", line)
            fwhm = ','.join(fwhm)
            fwhm = fwhm.split(',')
            fwhm = np.array(fwhm).astype(float)
    
    return fwhm


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
    green = array[:, :, green_index] 
    nir = array[:, :, nir_index] 
    ndwi = (green - nir) / (green + nir)

    array[(ndwi > ndwi_threshold)] = -9999

    return array


def mask_atmos_windows(value, wavelengths):
    '''
    TODO
    '''
    
    mask = ((wavelengths >= 1250) & (wavelengths <= 1450)) | ((wavelengths >= 1780) & (wavelengths <= 1950))

    value[mask] = np.nan
    
    return value
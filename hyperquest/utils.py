import numpy as np
import re
from os.path import abspath, exists
from dateutil import parser
from datetime import timezone
import numpy as np
from spectral import *
from skimage.draw import line

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




def read_hdr_metadata(hdr_path):
    '''
    TODO:

    '''

    # Get absolute path
    hdr_path = abspath(hdr_path)

    # Raise exception if file does not end in .hdr
    if not hdr_path.lower().endswith('.hdr'):
        raise ValueError(f'Invalid file format: {hdr_path}. Expected an .hdr file.')

    # Initialize variables
    wavelength = None
    fwhm = None
    start_time = None

    # Read the .hdr file and extract data
    for line in open(hdr_path, 'r'):
        line_lower = line.strip().lower()

        # wavelengths
        if 'wavelength' in line_lower and 'unit' not in line_lower:
            wavelength = re.findall(r"[+-]?\d+\.\d+", line)
            wavelength = ','.join(wavelength)
            wavelength = wavelength.split(',')
            wavelength = np.array(wavelength).astype(float)
            # Convert wavelengths from micrometers to nanometers if necessary
            if wavelength[0] < 300:
                wavelength = wavelength*1000

        # FWHM
        elif 'fwhm' in line_lower:
            fwhm = re.findall(r"[+-]?\d+\.\d+", line)
            fwhm = ','.join(fwhm)
            fwhm = fwhm.split(',')
            fwhm = np.array(fwhm, dtype=np.float64)    

        # Extract acquisition start time
        elif 'start' in line_lower and 'time' in line_lower:
            start_time = line.split('=')[-1].strip()
            obs_time = parser.parse(start_time).replace(tzinfo=timezone.utc)

    # ensure these are the same length
    if len(wavelength) != len(fwhm):
        raise ValueError('Wavelength and FWHM arrays have different lengths.')

    return wavelength, fwhm, obs_time




def get_img_path_from_hdr(hdr_path): 
    '''
    TODO:

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
    return 10 * np.log10(snr_linear)
    

def mask_water_using_ndwi(array, hdr_path, ndwi_threshold=0.25):
    '''
    Returns array where NDWI greater than a threshold are set to NaN.

    Parameters: 
        array (ndarray): 3d array of TOA radiance.
        hdr_path (str): Path to the .hdr file.
        ndwi_threshold (float): values above this value are masked.

    Returns:
        array: 3d array of TOA radiance (with water masked out).

    '''

    wavelengths,_,_ = read_hdr_metadata(hdr_path)
    green_index = np.argmin(np.abs(wavelengths - 559))
    nir_index = np.argmin(np.abs(wavelengths - 864))
    green = array[:, :, green_index] 
    nir = array[:, :, nir_index] 
    ndwi = (green - nir) / (green + nir)

    array[(ndwi > ndwi_threshold)] = np.nan

    return array


def mask_atmos_windows(spectra, wavelengths):
    '''
    Given spectra and wavelengths in nanometers, mask out noisy bands.

    Parameters: 
        spectra (ndarray): 1d array of TOA radiance data.
        wavelengths (ndarray): 1d array of sensor center wavelengths

    Returns:
        spectra: TOA radiance data but with noisy atmospheric bands masked out
    '''
    
    mask = ((wavelengths >= 1250) & (wavelengths <= 1450)) | ((wavelengths >= 1780) & (wavelengths <= 1950))

    spectra[mask] = np.nan
    
    return spectra


def cross_track_stats(image, no_data_value=-9999):
    '''
    Scans image line-by-line as the imager would and produces one row of average and std dev. To do this it assumes image is perfectly orthogonal.
    This function is helpful to reduce an image to 1 row for assessment but the image is rotated because of georectification.  

    Parameters: 
        image (ndarray): either a 2d or 3d array.
        no_data_value (int or float): Value used to describe no data regions.

    Returns:
        mean_along_line, std_along_line: mean and standard deviation of values for each band in a 1d numpy array. 
    '''

    # Mask no data values
    image[image <= no_data_value] = np.nan

    # Make a 3D array if not
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]

    if len(image.shape) != 3:
        raise Exception('Data needs to be a 3D array.')

    # Now i pad the array to ensure our line we make to grab cross track stats does not go out of bounds
    pad_value = np.nan 
    PAD = 100  
    image = np.pad(image,
                   ((PAD, PAD), 
                    (PAD, PAD), 
                    (0, 0)), 
                    mode='constant', constant_values=pad_value)
    r, c = image.shape[0], image.shape[1]

    # Get points of interest of valid data so that can manage the angles of the image
    # all of this trouble is because geo-referenced images are rotated to north
    # and so this is my sort of solution to this is to assume orthogonal image
    # which more or less works well for this case..
    top_left = next((x, y) for y in range(r) for x in range(c) if image[y, x, :].sum() > 0)
    bottom_left = next((x, y) for x in range(c) for y in range(r-1, -1, -1) if image[y, x, :].sum() > 0)
    top_right = next((x, y) for x in range(c-1, -1, -1) for y in range(0, r) if image[y, x, :].sum() > 0)

    # direction cross track
    dx, dy  = top_right[0] - top_left[0] , top_right[1] - top_left[1]
    dx, dy = dx / (dx**2 + dy**2)**0.5, dy / (dx**2 + dy**2)**0.5

    # prep for while
    mean_along_line = []
    std_along_line = []
    i = 0 
    while top_left[0] + i * dx <= top_right[0]:  

        # grab firt and last points
        x0, y0 = int(top_left[0] + i * dx), int(top_left[1] + i * dy)
        x1, y1 = int(bottom_left[0] + i * dx), int(bottom_left[1] + i * dy)

        # Create line using skiamge and then extract from image
        rr, cc = line(y0, x0, y1, x1)
        data = image[rr, cc, :]

        # Calculate mean and std 
        mean_data = np.nanmean(data, axis=0)
        std_data = np.nanstd(data, axis=0)

        # store and set next i
        mean_along_line.append(mean_data)
        std_along_line.append(std_data)
        i += 1  

    mean_along_line = np.array(mean_along_line)
    std_along_line = np.array(std_along_line)

    return mean_along_line, std_along_line

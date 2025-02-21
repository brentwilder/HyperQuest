import numpy as np
import re
from os.path import abspath, exists
from dateutil import parser
from datetime import timezone, datetime
import numpy as np
from spectral import *
import h5netcdf

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

    padded_image = np.full((rows + pad_rows, cols + pad_cols, bands), np.nan, dtype=np.float64)
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
    obs_time = None

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
    if 'rad' in path_to_data.lower():
        array = np.array(ds['radiance'], dtype=np.float64)
    else:
        array = np.array(ds['reflectance'], dtype=np.float64)

    fwhm = np.array(ds['sensor_band_parameters']['fwhm'][:].data.tolist(), dtype=np.float64)
    wave = np.array(ds['sensor_band_parameters']['wavelengths'][:].data.tolist(), dtype=np.float64)

    obs_time = datetime.strptime(ds.attrs['time_coverage_start'], '%Y-%m-%dT%H:%M:%S+0000')

    return array, fwhm, wave, obs_time


def linear_to_db(snr_linear):
    return 10 * np.log10(snr_linear)
    

def mask_water_using_ndwi(array, wavelengths, no_data_value = -9999, ndwi_threshold=0.25):
    '''
    Returns array where NDWI greater than a threshold are set to NaN.

    Parameters: 
        array (ndarray): 3d array of TOA radiance.
        hdr_path (str): Path to the .hdr file.
        ndwi_threshold (float): values above this value are masked.

    Returns:
        array: 3d array of TOA radiance (with water masked out).

    '''

    green_index = np.argmin(np.abs(wavelengths - 559))
    nir_index = np.argmin(np.abs(wavelengths - 864))
    green = array[:, :, green_index] 
    nir = array[:, :, nir_index] 
    ndwi = (green - nir) / (green + nir)

    array[(ndwi > ndwi_threshold)] = no_data_value

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
    
    mask = ((wavelengths >= 1250) & (wavelengths <= 1450)) | ((wavelengths >= 1750) & (wavelengths <= 1970))

    spectra[mask] = np.nan
    
    return spectra

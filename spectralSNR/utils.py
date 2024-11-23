import rasterio
import numpy as np
import re





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




def binning(local_sigma_dict, local_mu_dict, wavelengths, nbins):
    '''
    TODO
    '''
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

    return signal, noise



def read_center_wavelength(img_path):
    '''
    TODO
    '''
    wavelength = None
    for line in open(f'{img_path}.hdr', 'r'):
        if 'wavelength =' in line or 'wavelength=' in line: 
            wavelength = re.findall("[+-]?\d+\.\d+", line)
            wavelength = ','.join(wavelength)
            wavelength = wavelength.split(',')
            wavelength = np.array(wavelength)
    
    return wavelength

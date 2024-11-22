from spectral import io 
import numpy as np


def load_envi(data):
    """
    Helper function for loading ENVI data in the methods.
 
    Parameters:
    data (str or object): Path to an ENVI file or object
    
    Returns:
    datacube (object): The loaded ENVI data cube.
    """
    if isinstance(data, str):
        try:
            data = io.envi(data)
            return data
        except Exception as e:
            print(f"Error loading ENVI file: {e}")
            return None
    elif isinstance(data, object):  
        if hasattr(data, 'get_data') and hasattr(data, 'get_header'):
            return data 
        else:
            raise ValueError("Provided data is not a valid ENVI format object.")
    else:
        raise TypeError("Provided data is not a valid file path (string) or a valid ENVI data object.")




def pad_image(image, block_size):
    """
    Pads the image to ensure the dimensions are divisible by block_size, using np.nan for padding.
    """
    height, width, bands = image.shape
    
    # Calculate padding for height and width
    pad_height = (block_size - (height % block_size)) % block_size
    pad_width = (block_size - (width % block_size)) % block_size
    
    # Apply padding with np.nan
    padded_image = np.pad(image, 
                          ((0, pad_height), (0, pad_width), (0, 0)), 
                          mode='constant', 
                          constant_values=np.nan)
    
    return padded_image
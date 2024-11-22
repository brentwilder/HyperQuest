from spectral import open_image
import numpy as np


def load_data(data):
    """
    Helper function for loading hyperspectral data in the methods.
 
    Parameters:
    data (str or object): Path to a file (envi), or path to an object (envi)
    
    Returns:
    datacube (object): The loaded ENVI data cube.
    """
    if isinstance(data, str):
        try:
            data = open_image(data)
            return data
        except Exception as e:
            print(f"Error loading file with Spectral package: {e}")
            return None
    elif isinstance(data, object):  
        return data 
    else:
        raise TypeError("Provided data is not an object or path.")



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
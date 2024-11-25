from snr import *
from hysnr.utils import *

import matplotlib.pyplot as plt
import pandas as pd 
import geopandas as gpd

# TESTING
path = 'C:/Users/kenne/OneDrive/Documents/snr/tests/data/ca-test'
roi = 'C:/Users/kenne/OneDrive/Documents/snr/tests/data/roi.shp'
geometry = gpd.read_file(roi)['geometry']
ncpus = 3   # for working in parallel????
block_size = 16


wavelengths = read_center_wavelengths(path)

## RLSD TEST
signal, noise, snr = rlsd(path, block_size=16, nbins=150, ncpus=ncpus, snr_in_db=False, output_all=True)
plt.plot(wavelengths, noise, color='red')
plt.show()

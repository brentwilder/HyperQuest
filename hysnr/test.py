from snr import *
from common import *

import matplotlib.pyplot as plt
import pandas as pd 
import geopandas as gpd

# TESTING
path = 'C:/Users/kenne/OneDrive/Documents/snr/tests/data/smaller_ca'
roi = 'C:/Users/kenne/OneDrive/Documents/snr/tests/data/roi.shp'
geometry = gpd.read_file(roi)['geometry']
ncpus = 2 # for working in parallel????
block_size = 6


wavelengths = read_center_wavelengths(path)

## LMLSD TEST
#snr = lmlsd(path, block_size=block_size, nbins=150, ncpus=ncpus, snr_in_db=False)
#plt.plot(wavelengths, snr, color='blue')
## RLSD TEST
snr = rlsd(path, block_size=block_size, nbins=3, ncpus=ncpus, snr_in_db=False)
plt.plot(wavelengths, snr, color='red')
## HOMOGENOUS TEST
signal, noise, snr = homogeneous_area(path, geometry, snr_in_db=False, output_all=True)
plt.plot(wavelengths, snr, color='black')

plt.show()

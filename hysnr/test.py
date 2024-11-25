from snr import *
from utils import *

import matplotlib.pyplot as plt
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
plt.scatter(wavelengths, snr, color='black')


snr = hrdsdc(path)

plt.scatter(wavelengths, snr, color='orange')

snr = ssdc(path, block_size=16, nbins=150, ncpus=ncpus)
plt.scatter(wavelengths, snr, color='magenta')

plt.show()
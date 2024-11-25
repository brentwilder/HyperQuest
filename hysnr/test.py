from snr import *
from common import *

import matplotlib.pyplot as plt
import pandas as pd 
import geopandas as gpd

# TESTING
path = 'C:/Users/kenne/OneDrive/Documents/snr/tests/data/smaller_ca'
roi = 'C:/Users/kenne/OneDrive/Documents/snr/tests/data/roi.shp'
geometry = gpd.read_file(roi)['geometry']
ncpus = 4 # for working in parallel????
block_size = 2


wavelengths = read_center_wavelengths(path)

#df = homogeneous_area(path,roi,"region")

#df_l = lmlsd(path, block_size=block_size, nbins=150, ncpus=ncpus)
#signal, noise, snr = homogeneous_area(path, geometry, snr_in_db=True, output_all=True)

snr = lmlsd(path, block_size=block_size, nbins=150, ncpus=ncpus, snr_in_db=True)
#df_r = rlsd(path, block_size=block_size, nbins=150, ncpus=ncpus)

#plt.plot(df_l.index, df_l['SNR'], color='k')
plt.plot(wavelengths, snr, color='green')
#plt.plot(df_r.index, df_r['SNR'], color='blue')

plt.show()

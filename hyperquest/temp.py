
import hyperquest

import numpy as np
from spectral import *
from joblib import Parallel, delayed
from skimage.segmentation import slic
from sklearn.decomposition import PCA

# Define path to envi image header file
hdr_path = '/Users/brent/Code/HyperQuest/tests/data/SISTER_EMIT_L1B_RDN_20220827T091626_000.hdr'
img_path = '/Users/brent/Code/HyperQuest/tests/data/SISTER_EMIT_L1B_RDN_20220827T091626_000.bin'

# get wavelengths
wavelengths = hyperquest.read_center_wavelengths(hdr_path)

snr = hyperquest.ssdc(hdr_path, block_size=10, nbins=150, ncpus=8)

import matplotlib.pyplot as plt

plt.figure(figsize= (12,8))
plt.rcParams.update({'font.size': 12})

plt.scatter(wavelengths, snr, color='black', s=75, alpha=0.4)


plt.xlabel('Wavelength [nm]')
plt.ylabel('SNR')

plt.show()
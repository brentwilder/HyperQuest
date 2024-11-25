import hysnr
import matplotlib.pyplot as plt

# inputs
path = './tests/data/ca-test'
ncpus = 3  
block_size = 16
nbins = 150
n_segments = 200
compactness = 0.1
n_pca = 3

# get wavelengths
wavelengths = hysnr.read_center_wavelengths(path)

## RLSD TEST
signal, noise, snr = hysnr.rlsd(path, block_size=block_size, 
                                nbins=nbins, ncpus=ncpus, snr_in_db=False, output_all=True)
plt.scatter(wavelengths, snr, color='black')

## HRDSDC TEST
snr = hysnr.hrdsdc(path, n_segments=n_segments, 
                   compactness=compactness, n_pca=n_pca, ncpus=ncpus)
plt.scatter(wavelengths, snr, color='orange')


## SSDC TEST
snr = hysnr.ssdc(path, block_size=block_size, nbins=nbins, ncpus=ncpus)
plt.scatter(wavelengths, snr, color='magenta')


# Display
plt.show()
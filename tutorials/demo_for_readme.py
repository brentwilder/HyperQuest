import hyperquest
import matplotlib.pyplot as plt

# set path
envi_img_path = './tests/data/ca-test'

# get wavelengths
wavelengths = hyperquest.read_center_wavelengths(envi_img_path)

# compute using HRDSDC method
snr = hyperquest.hrdsdc(envi_img_path, n_segments=1000, 
                        compactness=0.1, n_pca=3, ncpus=3)

# save plot
plt.scatter(wavelengths, snr, color='black', s=100, alpha=0.7)
plt.xlabel('Wavelength (micrometer)')
plt.ylabel('SNR')
plt.savefig("./tests/plots/demo_snr.png", dpi=300, bbox_inches="tight")
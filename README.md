# HyperQuest

[![Build Status](https://github.com/brentwilder/hyperquest/actions/workflows/pytest.yml/badge.svg)](https://github.com/brentwilder/hyperquest/actions/workflows/pytest.yml)
![PyPI](https://img.shields.io/pypi/v/hyperquest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hyperquest)
[![Downloads](https://pepy.tech/badge/hyperquest)](https://pepy.tech/project/hyperquest)


`hyperquest`: A Python package for estimating image-wide noise across wavelengths in hyperspectral imaging (imaging spectroscopy) with an emphasis on repeatability and speed. Computations are sped up and scale with number of cpus.

Plenty of methods for denoising imagery already exist, however, sometimes there are applications where knowing/comparing SNR from image conditions is of interest. It is also important to point out this is __not__ instrument noise, which is measured in laboratory. These methods here provide an estimate of "actual noise in real conditions" (Curran & Dungan, 1989; Cogliati et al., 2021).

It's my hope `hyperquest` may be a useful tool and/or learning resource for computing SNR from imaging spectroscopy data. Comments and suggestions are welcome! 


## Installation Instructions

The latest release can be installed via pip:

```bash
pip install hyperquest
```

## Methods currently available
- __(HRDSDC)__ Homogeneous regions division and spectral de-correlation (Gao et al., 2008)

- __(SSDC)__ Spectral and spatial de-correlation (Roger & Arnold, 1996)

- __(RLSD)__ Residual-scaled local standard deviation (Gao et al., 2007)


## Usage example
```python
import hyperquest
import matplotlib.pyplot as plt


# Define path to envi image header file
envi_hdr_path = '/path/my_spectral_image.hdr'

# get wavelengths
wavelengths = hyperquest.read_center_wavelengths(envi_hdr_path)

# compute using HRDSDC method
snr = hyperquest.hrdsdc(envi_hdr_path, n_segments=10000, 
                        compactness=0.1, n_pca=3, ncpus=3)

plt.scatter(wavelengths, snr, color='black', s=100, alpha=0.7)
```
![SNR Plot](tests/plots/demo_snr.png)




## TODO:

- brainstorm: other quality metrics outside of SNR in this package?

- including other segmentation methods? Currently is all built around SLIC (via scikitlearn).

- provide Cogliati et al. (2021) method: extract edges and also include neighbor pixel.



## References:

- Cogliati, S., Sarti, F., Chiarantini, L., Cosi, M., Lorusso, R., Lopinto, E., ... & Colombo, R. (2021). The PRISMA imaging spectroscopy mission: overview and first performance analysis. Remote sensing of environment, 262, 112499.

- Curran, P. J., & Dungan, J. L. (1989). Estimation of signal-to-noise: a new procedure applied to AVIRIS data. IEEE Transactions on Geoscience and Remote sensing, 27(5), 620-628.

- Gao, L., Wen, J., & Ran, Q. (2007, November). Residual-scaled local standard deviations method for estimating noise in hyperspectral images. In Mippr 2007: Multispectral Image Processing (Vol. 6787, pp. 290-298). SPIE.

- Gao, L. R., Zhang, B., Zhang, X., Zhang, W. J., & Tong, Q. X. (2008). A new operational method for estimating noise in hyperspectral images. IEEE Geoscience and remote sensing letters, 5(1), 83-87.

- Roger, R. E., & Arnold, J. F. (1996). Reliably estimating the noise in AVIRIS hyperspectral images. International Journal of Remote Sensing, 17(10), 1951-1962.

- Tian, W., Zhao, Q., Kan, Z., Long, X., Liu, H., & Cheng, J. (2022). A new method for estimating signal-to-noise ratio in UAV hyperspectral images based on pure pixel extraction. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 16, 399-408.

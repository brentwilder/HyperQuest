# HyperQuest

[![Build Status](https://github.com/brentwilder/hyperquest/actions/workflows/pytest.yml/badge.svg)](https://github.com/brentwilder/hyperquest/actions/workflows/pytest.yml)
![PyPI](https://img.shields.io/pypi/v/hyperquest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hyperquest)
[![Downloads](https://pepy.tech/badge/hyperquest)](https://pepy.tech/project/hyperquest)


`hyperquest`: A Python package for estimating image-wide quality estimation metrics of hyperspectral imaging (imaging spectroscopy). Computations are sped up and scale with number of cpus.

Important: this package assumes your hyperspectral data is in ENVI format with a .HDR file.



## Installation Instructions

The latest release can be installed via pip:

```bash
pip install hyperquest
```


## All Methods

| **Category**             | **Method**                 | **Description**                                                                                                                               |
|--------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| **SNR**                  | `hrdsdc()`                 | Homogeneous regions division and spectral de-correlation (Gao et al., 2008)                                                                   |
|                          | `rlsd()`                   | Residual-scaled local standard deviation (Gao et al., 2007)                                                                                   |
|                          | `ssdc()`                   | Spectral and spatial de-correlation (Roger & Arnold, 1996)                                                                                    |
| **Co-Registration**      | `sub_pixel_shift()`        | Computes sub pixel co-registration between the VNIR & VSWIR imagers using skimage phase_cross_correlation                                     |
| **Smile**                | `smile_metric()`           | Similar to MATLAB "smileMetric". Computes derivatives of O2 and CO2 absorption features across-track (Dadon et al., 2010).                    |
|                          | `nodd_o2a()`               | Similar to method in Felde et al. (2003) to solve for nm shift at O2-A across-track. Requires radiative transfer model run.                   |
| **Radiative Transfer**   | `run_libradtran()`         | Runs libRadtran based on user input geometry and atmosphere at 1.0 cm-1. Saves to a .csv file for use in methods requiring radiative transfer.|




## Usage example

- see [SNR example](tutorials/example_using_EMIT.ipynb) where different SNR methods are computed over Libya-4.
- see [Smile example ](tutorials/testing_smile_methods.ipynb) where different smile methods are computed over Libya-4.




## References:

- Cogliati, S., Sarti, F., Chiarantini, L., Cosi, M., Lorusso, R., Lopinto, E., ... & Colombo, R. (2021). The PRISMA imaging spectroscopy mission: overview and first performance analysis. Remote sensing of environment, 262, 112499.
- Curran, P. J., & Dungan, J. L. (1989). Estimation of signal-to-noise: a new procedure applied to AVIRIS data. IEEE Transactions on Geoscience and Remote sensing, 27(5), 620-628.
- Dadon, A., Ben-Dor, E., & Karnieli, A. (2010). Use of derivative calculations and minimum noise fraction transform for detecting and correcting the spectral curvature effect (smile) in Hyperion images. IEEE Transactions on Geoscience and Remote Sensing, 48(6), 2603-2612.
- Felde, G. W., Anderson, G. P., Cooley, T. W., Matthew, M. W., Berk, A., & Lee, J. (2003, July). Analysis of Hyperion data with the FLAASH atmospheric correction algorithm. In IGARSS 2003. 2003 IEEE International Geoscience and Remote Sensing Symposium. Proceedings (IEEE Cat. No. 03CH37477) (Vol. 1, pp. 90-92). IEEE.
- Gao, L., Wen, J., & Ran, Q. (2007, November). Residual-scaled local standard deviations method for estimating noise in hyperspectral images. In Mippr 2007: Multispectral Image Processing (Vol. 6787, pp. 290-298). SPIE.
- Gao, L. R., Zhang, B., Zhang, X., Zhang, W. J., & Tong, Q. X. (2008). A new operational method for estimating noise in hyperspectral images. IEEE Geoscience and remote sensing letters, 5(1), 83-87.
- Roger, R. E., & Arnold, J. F. (1996). Reliably estimating the noise in AVIRIS hyperspectral images. International Journal of Remote Sensing, 17(10), 1951-1962.
- Scheffler, D., Hollstein, A., Diedrich, H., Segl, K., & Hostert, P. (2017). AROSICS: An automated and robust open-source image co-registration software for multi-sensor satellite data. Remote sensing, 9(7), 676.
- Thompson, D. R., Green, R. O., Bradley, C., Brodrick, P. G., Mahowald, N., Dor, E. B., ... & Zandbergen, S. (2024). On-orbit calibration and performance of the EMIT imaging spectrometer. Remote Sensing of Environment, 303, 113986.
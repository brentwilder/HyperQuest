# HyperQuest

[![Build Status](https://github.com/brentwilder/hyperquest/actions/workflows/pytest.yml/badge.svg)](https://github.com/brentwilder/hyperquest/actions/workflows/pytest.yml)
![PyPI](https://img.shields.io/pypi/v/hyperquest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hyperquest)
[![Downloads](https://pepy.tech/badge/hyperquest)](https://pepy.tech/project/hyperquest)


`hyperquest`: A Python package for estimating image-wide quality estimation metrics of hyperspectral imaging (imaging spectroscopy). Computations are sped up and scale with number of cpus.

__Important: this package assumes the following about input hyperspectral data:__ 
- Data must be in NetCDF (.nc) or ENVI (.hdr)
- Data not georeferenced (typically referred to as L1B before ortho)
- Data in radiance (assumed microW/cm2/nm/sr (for now))
- Pushbroom imaging spectrometer, such as, but not limited to:
    - AVIRIS-NG, AVIRIS-3, DESIS, EnMAP, EMIT, GaoFen-5, HISUI, Hyperion EO-1, HySIS, PRISMA, Tanager-1

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
| **Intrinsic Dimensionality (ID)**| `random_matrix_theory()` | Determining the Intrinsic Dimension (ID) of a Hyperspectral Image Using Random Matrix Theory (Cawse-Nicholson et al., 2012, Cawse-Nicholson et al., 2022) |
| **Co-Registration**      | `sub_pixel_shift()`        | Computes sub pixel co-registration between the VNIR & VSWIR imagers using skimage phase_cross_correlation                                     |
| **Striping (not destriping)**| `sigma_theshold()`     | As presented in Yokoya 2010, Preprocessing of hyperspectral imagery with consideration of smile and keystone properties.                      |
| **Smile**                | `smile_metric()`           | Similar to MATLAB "smileMetric". Computes derivatives of O2 and CO2 absorption features across-track (Dadon et al., 2010).                    |
|                          | `nodd_o2a()`               | Similar to method in Felde et al. (2003) to solve for nm shift at O2-A across-track. Requires radiative transfer model run.                   |
| **Radiative Transfer**   | `run_libradtran()`         | Runs libRadtran based on user input geometry and atmosphere at 0.1 nm spectral resolution. Saves to a .csv file for use in methods requiring radiative transfer.|


## Usage example

- see [EMIT example](tutorials/example_using_EMIT.ipynb) which has different methods computed over Libya-4.

## libRadtran install instructions
- Can be installed on Unix type system using the following link:
    - http://www.libradtran.org/doku.php?id=download


## Citation
Brent Wilder. (2025). brentwilder/HyperQuest: v0.XXX (vXXX). Zenodo. https://doi.org/10.5281/zenodo.14890171


## References:
- Cawse-Nicholson, K., Damelin, S. B., Robin, A., & Sears, M. (2012). Determining the intrinsic dimension of a hyperspectral image using random matrix theory. IEEE Transactions on Image Processing, 22(4), 1301-1310.
- Cawse‐Nicholson, K., Raiho, A. M., Thompson, D. R., Hulley, G. C., Miller, C. E., Miner, K. R., ... & Zareh, S. K. (2022). Intrinsic dimensionality as a metric for the impact of mission design parameters. Journal of Geophysical Research: Biogeosciences, 127(8), e2022JG006876.
- Cogliati, S., Sarti, F., Chiarantini, L., Cosi, M., Lorusso, R., Lopinto, E., ... & Colombo, R. (2021). The PRISMA imaging spectroscopy mission: overview and first performance analysis. Remote sensing of environment, 262, 112499.
- Curran, P. J., & Dungan, J. L. (1989). Estimation of signal-to-noise: a new procedure applied to AVIRIS data. IEEE Transactions on Geoscience and Remote sensing, 27(5), 620-628.
- Dadon, A., Ben-Dor, E., & Karnieli, A. (2010). Use of derivative calculations and minimum noise fraction transform for detecting and correcting the spectral curvature effect (smile) in Hyperion images. IEEE Transactions on Geoscience and Remote Sensing, 48(6), 2603-2612.
- Felde, G. W., Anderson, G. P., Cooley, T. W., Matthew, M. W., Berk, A., & Lee, J. (2003, July). Analysis of Hyperion data with the FLAASH atmospheric correction algorithm. In IGARSS 2003. 2003 IEEE International Geoscience and Remote Sensing Symposium. Proceedings (IEEE Cat. No. 03CH37477) (Vol. 1, pp. 90-92). IEEE.
- Gao, L., Wen, J., & Ran, Q. (2007, November). Residual-scaled local standard deviations method for estimating noise in hyperspectral images. In Mippr 2007: Multispectral Image Processing (Vol. 6787, pp. 290-298). SPIE.
- Gao, L. R., Zhang, B., Zhang, X., Zhang, W. J., & Tong, Q. X. (2008). A new operational method for estimating noise in hyperspectral images. IEEE Geoscience and remote sensing letters, 5(1), 83-87.
- Mayer, B., & Kylling, A. (2005). The libRadtran software package for radiative transfer calculations-description and examples of use. Atmospheric Chemistry and Physics, 5(7), 1855-1877.
- Roger, R. E., & Arnold, J. F. (1996). Reliably estimating the noise in AVIRIS hyperspectral images. International Journal of Remote Sensing, 17(10), 1951-1962.
- Scheffler, D., Hollstein, A., Diedrich, H., Segl, K., & Hostert, P. (2017). AROSICS: An automated and robust open-source image co-registration software for multi-sensor satellite data. Remote sensing, 9(7), 676.
- Thompson, D. R., Green, R. O., Bradley, C., Brodrick, P. G., Mahowald, N., Dor, E. B., ... & Zandbergen, S. (2024). On-orbit calibration and performance of the EMIT imaging spectrometer. Remote Sensing of Environment, 303, 113986.

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

Available methods and summaries can be found in [documentation](https://hyperquest.readthedocs.io).

## Usage example
- see [EMIT example](tutorials/example_using_EMIT.ipynb) which has different methods computed over Libya-4.

## libRadtran install instructions
- Can be installed on Unix type system using the following link:
    - http://www.libradtran.org/doku.php?id=download

## Citation
Brent Wilder. (2025). brentwilder/HyperQuest: v0.XXX (vXXX). Zenodo. https://doi.org/10.5281/zenodo.14890171
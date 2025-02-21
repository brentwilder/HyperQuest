# HyperQuest

[![Build Status](https://github.com/brentwilder/hyperquest/actions/workflows/pytest.yml/badge.svg)](https://github.com/brentwilder/hyperquest/actions/workflows/pytest.yml)
![PyPI](https://img.shields.io/pypi/v/hyperquest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hyperquest)
[![Downloads](https://pepy.tech/badge/hyperquest)](https://pepy.tech/project/hyperquest)


`hyperquest`: A Python package for estimating image-wide quality estimation metrics of hyperspectral imaging (imaging spectroscopy). Computations are sped up and scale with number of cpus. Available methods and summaries can be found in [documentation](https://hyperquest.readthedocs.io).

### Important: this package assumes the following about input hyperspectral data:
- Data must be in NetCDF (.nc) or ENVI (.hdr)
- Currently data is expected in Radiance.
  - For smile & striping methods, data must not be georeferenced (typically referred to as L1B before ortho)
- Pushbroom imaging spectrometer, such as, but not limited to:
    - AVIRIS-NG, AVIRIS-3, DESIS, EnMAP, EMIT, GaoFen-5, HISUI, Hyperion EO-1, HySIS, PRISMA, Tanager-1

NOTE: this is under active development. It is important to note that noise methods shown here do not account for spectrally correlated noise. This is a work in progress as I digest literature and translate into python.

## Installation Instructions
The latest release can be installed via pip:

```bash
pip install hyperquest
```

If using Windows PC, [you must have "Build Tools"](https://wiki.python.org/moin/WindowsCompilers) installed to compile cython code,
- Testing on my beat-up Windows PC (Windows11), I did the following to get it to work
  - Installed Visual Studio Build Tools 2022
  - making sure to check the box next to "Desktop development with C++"
  - and then, pip install hyperquest


## Usage example
- see [EMIT example](tutorials/example_using_EMIT.ipynb) which has different methods computed over Libya-4.

## libRadtran install instructions
- Can be installed on Unix type system using the following link:
    - http://www.libradtran.org/doku.php?id=download

## Citation
Brent Wilder. (2025). brentwilder/HyperQuest: v0.XXX (vXXX). Zenodo. https://doi.org/10.5281/zenodo.14890171
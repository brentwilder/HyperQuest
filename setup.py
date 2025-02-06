from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

with open("README.md", "r") as fh:
    long_description = fh.read()


ext_modules = [
    Extension(
        'hyperquest.mlr',
        sources=['hyperquest/mlr.pyx'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="hyperquest", 
    version="0.1.4", 
    author="Brent Wilder", 
    author_email="brentwilder@u.boisestate.edu",
    description=" A Python package for Hyperspectral quality estimation in hyperspectral imaging (imaging spectroscopy)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brentwilder/hyperquest", 
    ext_modules=ext_modules,
    packages=find_packages(), 
    install_requires=[
        "numpy>=1.4.0", 
        "pandas>=1.2.0",
        "scikit-image>=0.18.0",
        "scikit-learn>=0.24.0",
        "joblib>=1.0.0", 
        "cython>=3.0.11",
        "spectral>=0.23.0"
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7"
)

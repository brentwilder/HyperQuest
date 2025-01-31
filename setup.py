from setuptools import setup, find_packages

# Read the README file for the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hyperquest", 
    version="0.1.1", 
    author="Brent Wilder", 
    author_email="brentwilder@u.boisestate.edu",
    description=" A Python package for Hyperspectral Quality Estimation and computing image-wide noise in hyperspectral imaging (imaging spectroscopy)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brentwilder/hyperquest", 
    packages=find_packages(), 
    install_requires=[
        "numpy>=1.20.0", 
        "pandas>=1.2.0",
        "geopandas>=0.9.0",
        "rasterio>=1.2.0",
        "scikit-image>=0.18.0",
        "scikit-learn>=0.24.0",
        "joblib>=1.0.0", 
        "scipy>=1.6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7", 
)

Usage
=========================

.. _installation:

Installation
------------

To use hyperquest, first install it using pip:

.. code-block:: console

   $ pip install hyperquest

Table of Methods
----------------

.. list-table::
   :header-rows: 1

   * - **Category**
     - **Method**
     - **Description**
   * - **SNR**
     - `hrdsdc()`
     - Homogeneous regions division and spectral de-correlation (Gao et al., 2008)
   * - 
     - `rlsd()`
     - Residual-scaled local standard deviation (Gao et al., 2007)
   * - 
     - `ssdc()`
     - Spectral and spatial de-correlation (Roger & Arnold, 1996)
   * - **Intrinsic Dimensionality (ID)**
     - `random_matrix_theory()`
     - Determining the Intrinsic Dimension (ID) of a Hyperspectral Image Using Random Matrix Theory (Cawse-Nicholson et al., 2012, 2022)
   * - **Co-Registration**
     - `sub_pixel_shift()`
     - Computes sub pixel co-registration between the VNIR & VSWIR imagers using skimage phase_cross_correlation
   * - **Striping (not destriping)**
     - `sigma_theshold()`
     - As presented in Yokoya 2010, Preprocessing of hyperspectral imagery with consideration of smile and keystone properties.
   * - **Smile**
     - `smile_metric()`
     - Similar to MATLAB "smileMetric". Computes derivatives of O2 and CO2 absorption features across-track (Dadon et al., 2010).
   * - 
     - `nodd_o2a()`
     - Similar to method in Felde et al. (2003) to solve for nm shift at O2-A across-track. Requires radiative transfer model run.
   * - **Radiative Transfer**
     - `run_libradtran()`
     - Runs libRadtran based on user input geometry and atmosphere at 0.1 nm spectral resolution. Saves to a .csv file for use in methods requiring radiative transfer.


Example
----------------
see the following for an example using an EMIT image,
- https://github.com/brentwilder/HyperQuest/blob/main/tutorials/example_using_EMIT.ipynb
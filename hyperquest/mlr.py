import numpy as np
from scipy.optimize import nnls


def mlr_spectral(block):
    '''
    TODO
    
    '''

    # remove data that is NaN (keep only positive values)
    block = block[block[:,0] > -99]

    # Make a blank set of mu and sigma for filling in (value for each wavelength)
    mu_block = np.full(block.shape[1], np.nan)
    sigma_block = np.full(block.shape[1], np.nan)
    
    # for k in range of wavelengths (except first and last)
    for k in range(1, block.shape[1] - 1):

        # create the X and y for MLR
        X = np.vstack((block[:, k - 1], block[:, k + 1])).T
        y = block[:, k]

        if len(y) > 50: # from Gao, at least 50 pixels
            coef, _ = nnls(X, y)
            y_pred = X @ coef

            # 3 DOF because of MLR
            sigma_block[k] = np.nanstd(y - y_pred, ddof=3)
            mu_block[k] = np.nanmean(y)

    return mu_block, sigma_block


def mlr_spectral_spatial(block):
    '''
    TODO
    
    '''

    # remove data that is NaN (keep only positive values)
    block = block[block[:,0] > -99]

    # Make a blank set of mu and sigma for filling in (value for each wavelength)
    mu_block = np.full(block.shape[1], np.nan)
    sigma_block = np.full(block.shape[1], np.nan)
    
    # for k in range of wavelengths (except first and last)
    for k in range(1, block.shape[1] - 1):

        # create the X and y for MLR
        X = np.vstack((block[:, k - 1], block[:, k + 1])).T
        neighbor_k = np.roll(block[:, k], shift=1)  # Shift 1 to find a neighbor pixel
        X = np.column_stack((X, neighbor_k))
        y = block[:, k]

        if len(y) > 50: # from Gao, at least 50 pixels
            coef, _ = nnls(X, y)
            y_pred = X @ coef

            # 3 DOF because of MLR
            sigma_block[k] = np.nanstd(y - y_pred, ddof=4)
            mu_block[k] = np.nanmean(y)

    return mu_block, sigma_block
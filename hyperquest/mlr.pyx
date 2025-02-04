# mlr_functions.pyx
import cython
import numpy as np
cimport numpy as np
import ctypes

np.import_array()


def mlr_spectral(np.ndarray[np.float64_t, ndim=2] block):
    '''
    TODO
    '''

    # remove data that is NaN (keep only positive values)
    block = block[block[:, 0] > -99]

    cdef int rows = block.shape[0]
    cdef int cols = block.shape[1]

    # Create the arrays with NaN
    cdef np.ndarray[np.float64_t, ndim=1] mu_block = np.full(cols, np.float64(np.nan))
    cdef np.ndarray[np.float64_t, ndim=1] sigma_block = np.full(cols, np.float64(np.nan))

    cdef int k
    cdef np.ndarray[np.float64_t, ndim=2] X
    cdef np.ndarray[np.float64_t, ndim=1] y
    cdef np.ndarray[np.float64_t, ndim=1] coef
    cdef np.ndarray[np.float64_t, ndim=1] y_pred

    # for k in range of wavelengths (except first and last)
    for k in range(1, cols - 1):
        # create the X and y for MLR
        X = np.vstack([block[:, k - 1], block[:, k + 1]]).T
        y = block[:, k]

        if len(y) > 50: # from Gao, at least 50 pixels
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ coef

            # 3 DOF because of MLR
            sigma_block[k] = np.std(y - y_pred, ddof=3)
            mu_block[k] = np.mean(y)

    return mu_block, sigma_block


def mlr_spectral_spatial(np.ndarray[np.float64_t, ndim=2] block):
    '''
    TODO
    '''

    # remove data that is NaN (keep only positive values)
    block = block[block[:, 0] > -99]

    cdef int rows = block.shape[0]
    cdef int cols = block.shape[1]
    
    # Create the arrays with NaN
    cdef np.ndarray[np.float64_t, ndim=1] mu_block = np.full(cols, np.float64(np.nan))
    cdef np.ndarray[np.float64_t, ndim=1] sigma_block = np.full(cols, np.float64(np.nan))

    cdef int k
    cdef np.ndarray[np.float64_t, ndim=2] X
    cdef np.ndarray[np.float64_t, ndim=1] y
    cdef np.ndarray[np.float64_t, ndim=1] coef
    cdef np.ndarray[np.float64_t, ndim=1] y_pred
    cdef np.ndarray[np.float64_t, ndim=1] neighbor_k

    # for k in range of wavelengths (except first and last)
    for k in range(1, cols - 1):
        # create the X and y for MLR
        X = np.vstack([block[:, k - 1], block[:, k + 1]]).T
        neighbor_k = np.roll(block[:, k], shift=1)  # Shift 1 to find a neighbor pixel
        X = np.column_stack([X, neighbor_k])
        y = block[:, k]

        if len(y) > 50: # from Gao, at least 50 pixels
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ coef

            # 3 DOF because of MLR
            sigma_block[k] = np.std(y - y_pred, ddof=4)
            mu_block[k] = np.mean(y)

    return mu_block, sigma_block

import numpy as np

from .utils import *



def random_matrix_theory(path_to_data, noise_variance, mask_waterbodies=True, alpha=0.5, no_data_value=-9999):
    '''
    A line-by-line python adaption of K. Cawse-Nicholson algorithm presented in,

    Cawse-Nicholson, K., Raiho, A. M., Thompson, D. R., Hulley, 
    G. C., Miller, C. E., Miner, K. R., ... & Zareh, S. K. (2022). 
    Intrinsic dimensionality as a metric for the impact of mission design parameters. 
    Journal of Geophysical Research: Biogeosciences, 127(8), e2022JG006876.

    Parameters: 
        path_to_data (str): Path to the .hdr or .nc
        noise_variance (ndarray): noise variance for each band. Used to compute N (noise covariance matrix of size bands x bands).
        mask_waterbodies (bool, optional): Whether to mask water bodies based on NDWI threshold of 0.25. Default is True.
        alpha (float): Significance level. 0.5 was found to be optimal in a study using hyperspectral data (Cawse-Nicholson et al., 2012).
        no_data_value (int or float): Value used to describe no data regions.

    Returns:
        K_est (int): Intrinsic Dimensionality (ID) of image.

    '''

    # Identify data type
    if path_to_data.lower().endswith('.nc'):
        img, _, w, _ = retrieve_data_from_nc(path_to_data)
    else:
        # Load raster
        img_path = get_img_path_from_hdr(path_to_data)
        img = np.array(envi.open(path_to_data, img_path).load(), dtype=np.float64)
        # get wavelengths
        w, _, _ = read_hdr_metadata(path_to_data)

    # ensure data is 3d
    if len(img.shape) != 3:
        raise Exception('Data needs to be a 3D array.')

    # mask waterbodies
    if mask_waterbodies is True:
        img = mask_water_using_ndwi(img, w)

    # Mask no data values
    img[img <= no_data_value] = np.nan

    # reshape R based on n,bands
    rows, cols, bands = img.shape
    n = rows * cols
    R = np.reshape(img, (n,bands))

    # mean pixel value, size [1,bands]
    m = np.nanmean(R, axis=0)

    # create a matrix of mean values the same size as R
    M = np.tile(m, (n, 1))

    # set nan to zero
    R[np.isnan(R)] = 0

    # image covariance matrix
    S = (R-M).T @ (R-M)  / n
    S[np.isnan(S)] = 0

    # compute sigma, used for threshold at end
    a = 0.5
    b = 0.5
    mu = 1/n * (np.sqrt(n-a) + np.sqrt(bands-b))**2
    sig = 1/n * (np.sqrt(n-a) + np.sqrt(bands-b))*(1/np.sqrt(n-a)+1/np.sqrt(bands-b))**(1/3)
    s = (-3/2 * np.log(4*np.sqrt(np.pi) * alpha/100))**(2/3)
    sigma = mu + s*sig

    # Get the noise covariance matrix of size bands x bands
    noise_variance[np.isnan(noise_variance)] = 0
    noise_variance  = np.expand_dims(noise_variance, axis=1)
    N = np.diag(np.diag(noise_variance.T * noise_variance / bands)) # similar to https://github.com/bearshng/LRTFL0/blob/master/estNoise.m
    N[np.isnan(N)] = 0
    
    # Eigenvectors and eigenvalues of S
    # Note these are opposite of MATLAB [evec_S,eval_S]
    eval_S, evec_S = np.linalg.eig(S)
    sortind = np.argsort(eval_S)[::-1]
    eval_S = eval_S[sortind]
    evec_S = evec_S[:,sortind]

    # Eigenvectors and eigenvalues of Pi = S-N
    eval_Pi, evec_Pi = np.linalg.eig(S-N) 
    sortind2 = np.argsort(eval_Pi)[::-1]
    eval_Pi = eval_Pi[sortind2]
    evec_Pi = evec_Pi[:,sortind2]

    # there is a different threshold for each band to represent noise conditions
    X = []
    for i in range(bands):
        X.append((evec_Pi[:,i].T @ N @ evec_S[:,i]) / (evec_Pi[:,i].T @ evec_S[:,i])) 
    X = np.array(X)

    # The ID is the number of eigenvalues exceeding the threshold X.T*sigma
    thresholded = np.maximum(0, eval_S - X * sigma)
    intersect_i = np.argwhere(thresholded > 0)
    K_est = len(intersect_i)


    return K_est

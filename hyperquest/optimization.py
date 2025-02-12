import numpy as np
from scipy import optimize


def nodd_sse_min(x0, rtm_radiance, rtm_wave,w_sensor,fwhm, l_toa_observed):
    '''
    Define the objective function.
    
    '''
    # get updated step
    dlambda = x0[0]
    new_cwl = w_sensor + dlambda
    dfwhm = x0[1:]
    new_fhwm = fwhm + dfwhm

    # Apply new model data with SRF
    sigma = new_fhwm / (2* np.sqrt(2*np.log(2)))
    srf = np.exp(-1 * ( ((rtm_wave - new_cwl[:, None])**2) / (2 * sigma[:, None]**2)))
    l_toa_model = np.trapz(rtm_radiance * srf, dx=1) / np.trapz(srf, dx=1) 

    # NODD method below
    dtau_obs = np.gradient(-np.log(l_toa_observed), w_sensor)
    dtau_model = np.gradient(-np.log(l_toa_model), w_sensor)
    #offset mean
    dtau_obs -= np.mean(dtau_obs)
    dtau_model -= np.mean(dtau_model)
    # normalize by RMS
    nodd_obs = dtau_obs / np.sqrt(np.mean(dtau_obs**2))
    nodd_model = dtau_model / np.sqrt(np.mean(dtau_model**2))

    # compute residual and SSE
    residual =  nodd_model - nodd_obs
    sse = np.sum(residual**2)

    return sse


def invert_cwl_and_fwhm(x0, l_toa_observed, rtm_radiance, rtm_wave, w_sensor, fwhm):
    '''
    invert for CWL and FWHM
    '''

    if np.isnan(l_toa_observed).all():
        return (np.nan, np.nan)

    # run nonlinear optimizer (constrained)
    opt_result = optimize.minimize(nodd_sse_min, x0,
                                   args=(rtm_radiance,rtm_wave, w_sensor,fwhm,l_toa_observed),
                                   method='Nelder-Mead'
                                   )
    
    # Save output
    xfinal = opt_result.x
    dlambda = xfinal[0]
    dfwhm = xfinal[1:]

    # reduce output to only be the center-band nearest to 760-nm
    o2_a =  np.argmin(np.abs(w_sensor-760))
    cwl_opt = w_sensor[o2_a] + dlambda
    fwhm_opt = fwhm[o2_a] + dfwhm[o2_a]

    return (cwl_opt, fwhm_opt)

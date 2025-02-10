# Import libraries
import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt


def fun_min(x0, rtm_radiance, rtm_wave, l_toa_observed):
    '''
    Define the objective function.
    
    '''
    # Get x ready
    cwl = x0[0]
    fwhm = x0[1]

    # compute new band radiance based on CWL and FWHM
    sigma = fwhm / (2* np.sqrt(2*np.log(2)))

    # range of vales (wavelengths from rtm)
    srf = np.exp(-1 * ( (rtm_wave - cwl)**2 / (2*sigma**2) ) )

    # band radiance 
    l_toa_sensor_model = np.trapz(rtm_radiance * srf, dx=1) / np.trapz(srf, dx=1) 

    # Input the values into the system of equations
    residual =  l_toa_sensor_model - l_toa_observed
    print(l_toa_observed, l_toa_sensor_model)

    # Calculate SSE
    sse = np.sum(residual**2)
    print(cwl, fwhm)

    plt.scatter(rtm_wave, rtm_radiance, color='red')
    plt.plot(rtm_wave, srf, color='k')
    plt.scatter(cwl, l_toa_sensor_model)
    plt.show()

    return sse


def invert_cwl_and_fwhm(x0, l_toa_observed, rtm_radiance, rtm_wave):
    '''
    invert for CWL and FWHM


    '''

    lb = [x0[0]-50, 0]
    ub = [x0[0]+50, 30]
    bounds =  optimize.Bounds(lb=lb, ub=ub, keep_feasible=True)

    # run nonlinear optimizer (unconstrained)
    opt_result = optimize.minimize(fun_min, x0,
                            args=(rtm_radiance, 
                                  rtm_wave, 
                                  l_toa_observed),
                                  method='SLSQP'
                                 )
    
    # Save output
    xfinal = opt_result.x
    cwl_final = xfinal[0]
    fwhm_final = xfinal[1]
   

    # compute new band radiance based on CWL and FWHM
    sigma = fwhm_final / (2* np.sqrt(2*np.log(2)))

    # range of vales (wavelengths from rtm)
    srf = np.exp(-1 * ( (rtm_wave - cwl_final)**2 / (2*sigma**2) ) )

    # band radiance 
    l_toa_sensor_model = np.trapz(rtm_radiance * srf, dx=1) / np.trapz(srf, dx=1) 

    plt.scatter(rtm_wave, rtm_radiance, color='red')
    plt.plot(rtm_wave, srf, color='k')
    plt.show()


    return (cwl_final, fwhm_final)
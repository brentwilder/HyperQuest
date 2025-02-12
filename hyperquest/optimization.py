

# Import libraries
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def fun_min(x0, rtm_radiance, rtm_wave,w_sensor, l_toa_observed):
    '''
    Define the objective function.
    
    '''

    dlambda = x0[0]
    cwl = w_sensor + dlambda
    fwhm = x0[1:]

    sigma = fwhm / (2* np.sqrt(2*np.log(2)))

    srf = np.exp(-1 * ((rtm_wave - cwl[:, None])**2) / (2 * sigma[:, None]**2))

    l_toa_sensor_model = np.trapz(rtm_radiance * srf, dx=1) / np.trapz(srf, dx=1) 

    residual =  l_toa_sensor_model - l_toa_observed

    sse = np.sum(residual**2)

    # Plot for debugging
    print(dlambda)
    plt.scatter(w_sensor, l_toa_observed, color='red', label='Observed Radiance')
    plt.scatter(cwl, l_toa_sensor_model, color='blue', label='Modeled Radiance')
    plt.legend()
    plt.show()



    return sse


def invert_cwl_and_fwhm(x0, l_toa_observed, rtm_radiance, rtm_wave, w_sensor):
    '''
    invert for CWL and FWHM
    '''

    # CWL: data should not go beyond this range because it has been cropped
    # FWHD: should not be negative
    #lb = [700, 1e-16]
    #ub = [800, 50]
    #bounds =  optimize.Bounds(lb=lb, ub=ub, keep_feasible=True)

    # run nonlinear optimizer (constrained)
    opt_result = optimize.minimize(fun_min, x0,
                            args=(rtm_radiance, 
                                  rtm_wave, w_sensor,
                                  l_toa_observed),
                                  method='Nelder-Mead'
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





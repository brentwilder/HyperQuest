# Import libraries
import os
import subprocess
import numpy as np
import pandas as pd
from spectral import *
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator


def lrt_prepper(elevation, utc_time, lat, lon, vza):

    '''
    TODO

    So in summary users need
    phi = sensor azimuth , and vza = viewer zenith angle.

    

    '''


    # use pysolar compute sza from utc_time TODO
    sza = ''
    phi0 = 'sun azimuth'
    
    # convert to km
    elevation = elevation / 1000 

    # Check to use subarctic or midlat summer atmosphere
    if abs(lat) >= 60:
        atmos = 'ss'
    else:
        atmos = 'ms'

    # Assign N / S / E / W
    if lat >= 0:
        lat_inp = str(f'N {abs(lat)}')
    else:
        lat_inp = str(f'S {abs(lat)}')

    if lon >= 0:
        lon_inp = str(f'E {abs(lon)}')
    else:
        lon_inp = str(f'W {abs(lon)}')

    # cos vza
    umu = np.cos(np.radians(vza))


    return vza, umu, sza, phi0, lat_inp, lon_inp, elevation, atmos





def write_lrt_inp(o3, h , aod, a, out_str, umu, phi0, phi, sza, lat_inp, lon_inp, doy, altitude_km,
              atmos, path_to_libradtran_bin, lrt_dir, path_to_libradtran_base):
    '''

    adapted from: https://github.com/MarcYin/libradtran


    '''
    foutstr = out_str[0] + out_str[1]
    fname = f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alb_{a}_alt_{round(altitude_km*1000)}_{foutstr}'
    with open(f'{fname}.INP', 'w') as f:
        f.write('source solar\n')  # extraterrestrial spectrum
        f.write('wavelength 340 2510\n')  # set range for lambda
        f.write(f'atmosphere_file {path_to_libradtran_base}/data/atmmod/afgl{atmos}.dat\n')
        f.write(f'albedo {a}\n') 
        f.write(f'umu {umu}\n') # Cosine of the view zenith angle
        f.write(f'phi0 {phi0}\n') # SAA
        f.write(f'phi {phi}\n') # VAA
        f.write(f'sza {sza}\n')  # solar zenith angle
        f.write('rte_solver disort\n')
        f.write('pseudospherical\n')
        f.write(f'latitude {lat_inp}\n')
        f.write(f'longitude {lon_inp}\n')
        f.write(f'day_of_year {doy}\n') 
        f.write(f'mol_modify O3 {o3} DU\n')   
        f.write(f'mol_abs_param reptran fine\n')   # Fine cm-1
        f.write(f'mol_modify H2O {h} MM\n')   
        f.write(f'crs_model rayleigh bodhaine \n')  
        f.write(f'zout {out_str[0]}\n')  
        f.write(f'altitude {altitude_km}\n')    
        f.write(f'aerosol_default\n')  
        f.write(f'aerosol_species_file continental_average\n') 
        f.write(f'aerosol_set_tau_at_wvl 550 {aod}\n')  
        f.write(f'output_quantity transmittance\n')  
        f.write(f'output_user lambda {out_str[1]}\n')  
        f.write('quiet')
    cmd = f'{path_to_libradtran_bin}/uvspec < {fname}.INP > {fname}.out'
    return cmd




def write_lrt_inp_irrad(o3, h , aod, a, out_str, umu, phi0, phi, sza, lat_inp, lon_inp, doy, altitude_km,
                        atmos, path_to_libradtran_bin, lrt_dir, path_to_libradtran_base):
    # Run here manually for irrad
    fname = f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alt_{round(altitude_km*1000)}_IRRAD'
    with open(f'{fname}.INP', 'w') as f:
        f.write('source solar\n') 
        f.write('wavelength 340 2510\n')  
        f.write(f'atmosphere_file {path_to_libradtran_base}/data/atmmod/afgl{atmos}.dat\n')
        f.write(f'albedo {a}\n')  
        f.write(f'sza {sza}\n')  
        f.write('rte_solver disort\n')  
        f.write('pseudospherical\n')
        f.write(f'latitude {lat_inp}\n')
        f.write(f'longitude {lon_inp}\n')
        f.write(f'day_of_year {doy}\n')  
        f.write(f'zout {altitude_km}\n')  
        f.write(f'aerosol_default\n')  
        f.write(f'aerosol_species_file continental_average\n')  
        f.write(f'aerosol_set_tau_at_wvl 550 {aod}\n')    
        f.write(f'mol_modify O3 {o3} DU\n')  
        f.write(f'mol_abs_param reptran fine\n')   # Fine cm-1
        f.write(f'mol_modify H2O {h} MM\n')    
        f.write(f'crs_model rayleigh bodhaine \n')  
        f.write(f'output_user lambda edir edn \n') 
        f.write('quiet')
    cmd = f'{path_to_libradtran_bin}/uvspec < {fname}.INP > {fname}.out'
    return cmd



def lrt_create_args_for_pool(h20_range,
                             aod,
                             altitude_km,
                             umu, phi0, 
                             phi,vza,
                             sza, lat_inp,
                             lon_inp, doy, atmos, 
                             o3, albedo,
                             lrt_dir, 
                             path_to_libradtran_bin):
    '''
    Takes in the range of water column vapor, a550, and altitude
    and creates input commands for libRadtran, that will be ran in multipool.

    '''
    # Run the LRT LUT pipeline
    path_to_libradtran_base = os.path.dirname(path_to_libradtran_bin)

    lrt_inp = []
    lrt_inp_irrad = []
    for h in h20_range:   
        # path radiance run
        cmd = write_lrt_inp(o3, h,aod,0, ['toa','uu'], umu, phi0, phi, sza, 
                        lat_inp, lon_inp, doy, altitude_km, atmos, path_to_libradtran_bin, 
                        lrt_dir, path_to_libradtran_base)
        lrt_inp.append([cmd,path_to_libradtran_bin])
        
        # upward transmittance run
        cmd = write_lrt_inp(o3, h,aod,0, ['sur','eglo'], umu, phi0, phi, vza, 
                        lat_inp, lon_inp, doy, altitude_km, atmos, path_to_libradtran_bin, 
                        lrt_dir, path_to_libradtran_base)
        lrt_inp.append([cmd,path_to_libradtran_bin])

        # spherical albedo run 1
        cmd = write_lrt_inp(o3, h,aod,0.15, ['sur','eglo'], umu, phi0, phi, sza, 
                        lat_inp, lon_inp, doy, altitude_km, atmos, path_to_libradtran_bin, 
                        lrt_dir, path_to_libradtran_base)
        lrt_inp.append([cmd,path_to_libradtran_bin])
        
        # spherical albedo run 2
        cmd = write_lrt_inp(o3, h,aod,0.5, ['sur','eglo'], umu, phi0, phi, sza, 
                        lat_inp, lon_inp, doy, altitude_km, atmos, path_to_libradtran_bin, 
                        lrt_dir, path_to_libradtran_base)   
        lrt_inp.append([cmd,path_to_libradtran_bin])

        # incoming solar irradiance run
        cmd = write_lrt_inp_irrad(o3, h,aod, albedo, ['toa','uu'], umu, phi0, phi, sza, 
                        lat_inp, lon_inp, doy, altitude_km, atmos, path_to_libradtran_bin, 
                        lrt_dir, path_to_libradtran_base)
        lrt_inp_irrad.append([cmd,path_to_libradtran_bin])

    return lrt_inp_irrad, lrt_inp








def lut_grid(h20_range,aod, altitude_km, path_to_img_base, sensor_wavelengths):
    '''

    Grid the LUT so they are continuous variables for numerical optimization.

    '''
    # LRT dir
    lrt_dir = f'{path_to_img_base}_albedo/libradtran'

    # Create empty grids
    l0_arr = np.empty(shape=(len(h20_range), len(sensor_wavelengths)))
    t_up_arr = np.empty(shape=(len(h20_range),len(sensor_wavelengths)))
    s_arr  = np.empty(shape=(len(h20_range), len(sensor_wavelengths)))
    edir_arr = np.empty(shape=(len(h20_range),len(sensor_wavelengths)))
    edn_arr = np.empty(shape=(len(h20_range), len(sensor_wavelengths)))

    for i in range(0, len(h20_range)):

        h = h20_range[i]

        # Now load in each of them into pandas to perform math.
        df_r = pd.read_csv(f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alb_0_alt_{round(altitude_km*1000)}_toauu.out', delim_whitespace=True, header=None)
        df_r.columns = ['Wavelength','uu']

        df_t = pd.read_csv(f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alb_0_alt_{round(altitude_km*1000)}_sureglo.out', delim_whitespace=True, header=None)
        df_t.columns = ['Wavelength', 'eglo']

        df_s1 = pd.read_csv(f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alb_0.15_alt_{round(altitude_km*1000)}_sureglo.out', delim_whitespace=True, header=None)
        df_s1.columns = ['Wavelength', 'eglo']

        df_s2 = pd.read_csv(f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alb_0.5_alt_{round(altitude_km*1000)}_sureglo.out', delim_whitespace=True, header=None)
        df_s2.columns = ['Wavelength', 'eglo']

        df_irr = pd.read_csv(f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alt_{round(altitude_km*1000)}_IRRAD.out', delim_whitespace=True, header=None)
        df_irr.columns = ['Wavelength', 'edir', 'edn']

        # Fit spline to match  for L_0 (path radiance)
        fun_r = interpolate.interp1d(df_r['Wavelength'], df_r['uu'], kind='slinear')
        l0 = fun_r(sensor_wavelengths)

        # Compute t_up (upward transmittance)
        fun_t = interpolate.interp1d(df_t['Wavelength'], df_t['eglo'], kind='slinear')
        t_up = fun_t(sensor_wavelengths)   

        # Compute S (atmos sphere albedo)
        df_s2['sph_alb'] = (df_s2['eglo'] - df_s1['eglo']) / (0.5 * df_s2['eglo'] -  0.15 * df_s1['eglo'])
        fun_s = interpolate.interp1d(df_s2['Wavelength'], df_s2['sph_alb'], kind='slinear')
        s = fun_s(sensor_wavelengths)

        # Fit spline to match edir and edn
        f_dir = interpolate.interp1d(df_irr['Wavelength'], df_irr['edir'], kind='slinear')
        edir = f_dir(sensor_wavelengths)

        f_edn = interpolate.interp1d(df_irr['Wavelength'], df_irr['edn'], kind='slinear')
        edn = f_edn(sensor_wavelengths)

        # append the results
        l0_arr[i,:] = l0
        t_up_arr[i,:] = t_up
        s_arr[i,:] = s
        edir_arr[i,:] = edir
        edn_arr[i,:] = edn
    
    # Now prep for new grid
    W = np.copy(sensor_wavelengths)
    H = np.array(h20_range)

    # Create grid functions
    g_l0 = RegularGridInterpolator((H,W), l0_arr, method='linear')
    g_tup = RegularGridInterpolator((H, W), t_up_arr, method='linear')
    g_s = RegularGridInterpolator((H, W), s_arr, method='linear')
    g_edir = RegularGridInterpolator((H, W), edir_arr, method='linear')
    g_edn = RegularGridInterpolator((H, W), edn_arr, method='linear')


    return  g_l0, g_tup, g_s, g_edir, g_edn





def lrt_reader(h, aod, alt, sza, rho_surface,
               g_l0, g_tup, g_s, g_edir, g_edn,
               sensor_wavelengths, 
               cosi=1, shadow=1, svf=1, slope=0):
    
    '''
    TODO

    '''
    # Ensure optimization stays in bounds
    if h <= 1:
        h=1

    # Setup arrays
    W = np.copy(sensor_wavelengths)
    H = np.array(h)

    # GRID INTERPS
    l0 = g_l0((H,W))
    t_up = g_tup((H,W))
    s = g_s((H,W))
    edir0 = g_edir((H,W))
    edn0 = g_edn((H,W))

    # Correct to local conditions
    #############################  
    t_up = t_up / np.cos(np.radians(sza)) 

    # Adjust local Fdir and Fdiff
    edir =  edir0 * cosi  * shadow #shadow: 0=shadow, 1=sun
    edn =  edn0  * svf
    
    # Combine diffuse and direct into S_total
    s_total = edir + edn

    # Add in adjacent pixels estimate (terrain influence)
    # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JD034294
    # This only impacts diffuse component.
    ct = max(0,((1 + np.cos(np.radians(slope))) / 2 ) - svf)
    s_total = s_total + (edn0*rho_surface * ct)

    # Correct units to be microW/cm2/nm/sr
    s_total = s_total / 10
    l0 = l0 / 10


    return l0, t_up, s, s_total
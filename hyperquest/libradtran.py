# Import libraries
import os
import numpy as np
import pandas as pd
from spectral import *



def get_libradtran_install_path(user_input_path):
    '''
    TODO
    '''
    user_input_path = os.path.abspath(user_input_path)

    # Check if they already gave the bin directory
    if os.path.basename(user_input_path) == 'bin' and os.path.isdir(user_input_path):
        return user_input_path

    # Check if 'bin' exists inside the given directory
    bin_path = os.path.join(user_input_path, 'bin')
    if os.path.isdir(bin_path):
        return bin_path
    else:
        raise FileNotFoundError('Could not determine location of libRadtran installation.')


    return



def get_libradtran_output_dir(hdr_path):
    '''
    TODO
    '''

    filename = os.path.splitext(os.path.basename(hdr_path))[0]
    
    # directory where hdr_path is located
    parent_dir = os.path.dirname(os.path.abspath(hdr_path))
    
    # Define the rtm output directory
    lrt_out_dir = os.path.join(parent_dir, f'rtm-{filename}')

    # Create the directory if it does not exist
    os.makedirs(lrt_out_dir, exist_ok=True)

    return lrt_out_dir




def write_lrt_inp(o3, h , aod, a, out_str, umu, phi0, phi, sza, lat_inp, lon_inp, doy, altitude_km,
              atmos, path_to_libradtran_bin, lrt_dir, path_to_libradtran_base):
    '''

    adapted from: https://github.com/MarcYin/libradtran

        Cm-1 resolution of libRadtran ... which is output from lrt as 0.1 nm spectral resolution.


    '''
    foutstr = out_str[0] + out_str[1]
    fname = f'{lrt_dir}/lrt_h20_{h}_aot_{aod}_alb_{a}_alt_{round(altitude_km*1000)}_{foutstr}'
    with open(f'{fname}.INP', 'w') as f:
        f.write(f'source solar {path_to_libradtran_base}/data/solar_flux/kurudz_0.1nm.dat\n') 
        f.write('wavelength 549 860\n')  
        f.write(f'atmosphere_file {path_to_libradtran_base}/data/atmmod/afgl{atmos}.dat\n')
        f.write(f'albedo {a}\n') 
        f.write(f'umu {umu}\n') # Cosine of the view zenith angle
        f.write(f'phi0 {phi0}\n') # SAA
        f.write(f'phi {phi}\n') # VAA
        f.write(f'sza {sza}\n')  # solar zenith angle
        f.write('rte_solver disort\n')
        f.write(f'number_of_streams 8\n')
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
        f.write(f'source solar {path_to_libradtran_base}/data/solar_flux/kurudz_0.1nm.dat\n') 
        f.write('wavelength 549 860\n')  
        f.write(f'atmosphere_file {path_to_libradtran_base}/data/atmmod/afgl{atmos}.dat\n')
        f.write(f'albedo {a}\n')  
        f.write(f'sza {sza}\n')  
        f.write('rte_solver disort\n')  
        f.write(f'number_of_streams 8\n')
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



def lrt_create_args_for_pool(h,
                             aod,
                             altitude_km,
                             umu, phi0, 
                             phi,vza,
                             sza, lat_inp,
                             lon_inp, doy, atmos, 
                             o3, albedo,
                             lrt_dir, path_to_libradtran_bin):
    '''
    TODO
    '''
    # Run the LRT LUT pipeline
    path_to_libradtran_base = os.path.dirname(path_to_libradtran_bin)

    lrt_inp = []
    lrt_inp_irrad = []

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





def lrt_to_pandas_dataframe(h,aod, altitude_km, sza, lrt_out_dir):
    '''

    Save the data to panadas

    '''

    # Now load in each of them into pandas to perform math.
    df_r = pd.read_csv(f'{lrt_out_dir}/lrt_h20_{h}_aot_{aod}_alb_0_alt_{round(altitude_km*1000)}_toauu.out', sep='\s+', header=None)
    df_r.columns = ['Wavelength','uu']

    df_t = pd.read_csv(f'{lrt_out_dir}/lrt_h20_{h}_aot_{aod}_alb_0_alt_{round(altitude_km*1000)}_sureglo.out', sep='\s+', header=None)
    df_t.columns = ['Wavelength', 'eglo']

    df_s1 = pd.read_csv(f'{lrt_out_dir}/lrt_h20_{h}_aot_{aod}_alb_0.15_alt_{round(altitude_km*1000)}_sureglo.out', sep='\s+', header=None)
    df_s1.columns = ['Wavelength', 'eglo']

    df_s2 = pd.read_csv(f'{lrt_out_dir}/lrt_h20_{h}_aot_{aod}_alb_0.5_alt_{round(altitude_km*1000)}_sureglo.out', sep='\s+', header=None)
    df_s2.columns = ['Wavelength', 'eglo']

    df_irr = pd.read_csv(f'{lrt_out_dir}/lrt_h20_{h}_aot_{aod}_alt_{round(altitude_km*1000)}_IRRAD.out', sep='\s+', header=None)
    df_irr.columns = ['Wavelength', 'edir', 'edn']

    # Compute S (atmos sphere albedo)
    df_s2['sph_alb'] = (df_s2['eglo'] - df_s1['eglo']) / (0.5 * df_s2['eglo'] -  0.15 * df_s1['eglo'])

    # to one pandas dataframe
    df = pd.DataFrame(data=df_irr['Wavelength'], columns=['Wavelength'])
    df['l0'] = df_r['uu']
    df['t_up'] = df_t['eglo'] / np.cos(np.radians(sza)) 
    df['s'] = df_s2['sph_alb']
    df['e_dir'] = df_irr['edir']
    df['e_diff'] = df_irr['edn']

    # Set units to be microW/cm2/nm/sr (common for EMIT & PRISMA)
    df['e_dir'] = df['e_dir'] / 10
    df['e_diff'] = df['e_diff'] / 10
    df['l0'] = df['l0'] / 10

    
    return  df

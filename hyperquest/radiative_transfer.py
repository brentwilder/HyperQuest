from pysolar import solar
from joblib import Parallel, delayed
from os.path import abspath
import subprocess

from .libradtran import *
from .utils import *

def run_libradtran(h2o_mm, aod_at_550nm, sensor_zenith_angle, sensor_azimith_angle,
                   path_to_data, average_elevation_meters, lat, lon, libradtran_path, ncpus=1, o3_DU=300, albedo=0.15):
    '''
    TODO
    '''

    # Get absolute path
    path_to_data = abspath(path_to_data)
    libradtran_path = abspath(libradtran_path)

    # path_to_libradtran_install
    # get abs, get bin directory... throw error if not found
    path_to_libradtran_bin = get_libradtran_install_path(libradtran_path)

    # Identify data type
    if path_to_data.lower().endswith('.nc'):
        _, _, _, obs_time = retrieve_data_from_nc(path_to_data)
    else:
        # get wavelengths
        _, _, obs_time = read_hdr_metadata(path_to_data)

    # convert to doy
    doy = obs_time.timetuple().tm_yday

    # path to where runs are saved
    lrt_out_dir = get_libradtran_output_dir(path_to_data)
    
    # average altitude in km
    altitude_km = average_elevation_meters / 1000

    # use pysolar compute saa and sza
    phi0 = solar.get_azimuth(lat,lon, obs_time)
    sza = 90 - solar.get_altitude(lat,lon, obs_time)

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
    umu = np.cos(np.radians(sensor_zenith_angle))

    # get commands for running  libradtran
    lrt_inp_irrad, lrt_inp = lrt_create_args_for_pool(h2o_mm, aod_at_550nm, altitude_km, umu, phi0, 
                                                      sensor_azimith_angle,sensor_zenith_angle, 
                                                      sza, lat_inp, lon_inp,
                                                      doy, atmos, o3_DU, albedo, 
                                                      lrt_out_dir, path_to_libradtran_bin)
    
    # set max workers to 2 for now - RAM dominant
    ncpus = (min(ncpus, 2))

    # Go trhough runs in parallel
    Parallel(n_jobs=ncpus)(delayed(subprocess.run)(cmd, shell=True, cwd=path_to_libradtran_bin) 
                           for cmd in (lrt_inp + lrt_inp_irrad)
                           )

    # Create pandas datatable after runs
    df = lrt_to_pandas_dataframe(h2o_mm, aod_at_550nm, altitude_km, sza, lrt_out_dir)
    
    df['h_mm'] = h2o_mm
    df['aod_550'] = aod_at_550nm

    # Save to csv file
    csv_path = f'{lrt_out_dir}/radiative_transfer_output.csv'
    df.to_csv(csv_path)


    return df
from dem_stitcher import stitch_dem
import rasterio as rio
from pysolar import solar
from rasterio.warp import transform_bounds
from joblib import Parallel, delayed
from os.path import abspath
import subprocess

from .libradtran import *
from .utils import *

def run_libradtran(h2o_mm, aod_at_550nm, sensor_zenith_angle, sensor_azimith_angle,
                   hdr_path, libradtran_path, ncpus=1, o3_DU=300, albedo=0.15):
    '''
    TODO
    '''

    # Get absolute path
    hdr_path = abspath(hdr_path)
    libradtran_path = abspath(libradtran_path)

    # path_to_libradtran_install
    # get abs, get bin directory... throw error if not found
    path_to_libradtran_bin = get_libradtran_install_path(libradtran_path)

    # Get data from hdr
    wavelength, fwhm, obs_time = read_hdr_metadata(hdr_path)
    doy = obs_time.timetuple().tm_yday

    # path to where runs are saved
    lrt_out_dir = get_libradtran_output_dir(hdr_path)

    # Get bounds
    img_path = get_img_path_from_hdr(hdr_path)
    with rio.open(img_path) as dataset:
        bounds_utm = dataset.bounds
        # to EPSG:4326 , as xmin, ymin, xmax, ymax in epsg:4326
        bounding_box = transform_bounds(dataset.crs, 'EPSG:4326', *bounds_utm)
    
    # Get Copernicus DEM data based on bounding box in hdr 
    # (assume mus = mu0 ; flat assumption) ...  X is an mxn numpy array
    X, _ = stitch_dem(bounding_box, dem_name='glo_90')
    X = X[:, :].flatten()
    X = X[X < 8848] #mt everest
    X = X[X > -430] #deadsea
    X = X[~np.isnan(X)]
    altitude_km = np.nanmean(X) / 1000

    # Get average lat and lon
    lon = np.mean([bounding_box[0], bounding_box[2]])
    lat = np.mean([bounding_box[1], bounding_box[3]])

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
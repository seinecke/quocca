# quocca: QUick Observation of Cloud Coverage using All sky images
#  
# Utilities for calibration, adding cameras, fitting camera parameters.
#  
# Authors: S. Einecke <sabrina.einecke@adelaide.edu.au>
#          T. Hoinka <tobias.hoinka@icecube.wisc.edu>
#          H. Nawrath <helena.nawrath@tu-dortmund.de>
import numpy as np

from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator

from ..catalog import Catalog
from ..image import Image

from astropy.coordinates import Angle
import astropy.units as u
from astropy.time import Time

from skimage.filters import gaussian

from ruamel.yaml import YAML

from pkg_resources import resource_filename


def update_camera(name, **kwargs):
    with open(resource_filename('quocca', 'resources/cameras.yaml')) as file:
        yaml = YAML(typ='safe', pure=True)
        __config__ = yaml.load(file)
        __supported_cameras__ = list(__config__.keys())
    if name not in __supported_cameras__:
        raise NameError('Camera {} does not exist.'
                        .format(name))
    __config__[name].update(kwargs)
    res_fn = resource_filename('quocca', 'resources/cameras.yaml')
    yaml = YAML(typ='safe', pure=True)
    yaml.default_flow_style = False
    yaml.dump(__config__, open(res_fn, 'w'))


def add_camera(name,
               location,
               mapping,
               resolution,
               size,
               zenith,
               radius,
               az_offset,
               timestamps,
               max_val,
               force=False,
               **kwargs):
    """Adds a new camera to the ensemble.

    Parameters
    ----------
    name : str
        Name of the camera. If name already exists, abort, unless force is
        True, then camera will be overwritten.
    location : dict
        Dict with keys lat, lon, height containing the location on earth.
    mapping : str
        Identifier for the mapping function, see documentation of the camera
        module for details.
    resolution : dict
        Dict with keys x, y containing the resolution of the camera in pixels.
    size : dict
        Dict with keys x, y, containing the physical size of the sensor in mm.
    radius : float
        Radius in pixels
    az_offset : float
        Azimuth offset in degress
    timestamps : list
        List of timestamps in header
    max_val : int
        Maximum value in image data (usually 2 ** n)
    force : bool
        Whether or not to force overwriting
    kwargs : keywords
        Additional parameters
    """
    with open(resource_filename('quocca', 'resources/cameras.yaml')) as file:
        yaml = YAML(typ='safe', pure=True)
        __config__ = yaml.safe_load(file)
        __supported_catalogs__ = list(__config__.keys())
    if name in __supported_catalogs__ and not force:
        raise NameError('Camera {} already exists. Use keyword force to overwrite.'
                        .format(name))
    d = {name: {
        'location': location,
        'mapping': mapping,
        'resolution': resolution,
        'size': size,
        'zenith': zenith,
        'radius': radius,
        'az_offset': az_offset,
        'timestamps': timestamps,
        'max_val': max_val,
        **kwargs
    }}
    __config__.update(d)
    res_fn = resource_filename('quocca', 'resources/cameras.yaml')
    yaml = YAML(typ='safe', pure=True)
    yaml.default_flow_style = False
    yaml.dump(__config__, open(res_fn, 'w'))


def fit_camera_params(img_path, cam,
                      kwargs_catalog={'catalog':'hipparcos', 'max_mag': 6, 
                     'min_dist': 12.0, 'max_var': 2, 'min_alt': 30},
                      x0=None,
                      init_sigma=10.0,
                      stepsize=1.2,
                      update=False,
                      verbose=False):
    """Procedure to fit camera parameters using a clear sky image.

    Parameters
    ----------
    img_path : str
        Path to a clear sky image.
    cam: quocca.camera.Camera object
        Camera.
    max_mag : float
        Maximum magnitude to be considered during the fit.
    x0 : numpy.array, shape=(4,)
        Initial guess for camera parameters. If None is passed (default), then
        some initial guess is generated from other camera parameters.
    init_sigma : float
        Initial sigma passed to gaussian filtering to smear out stars. If fit
        results are very obviously wrong, try increasing this value.
    stepsize : float
        Factor that is multiplied to sigma after each step. If fit results are
        diverging, try brining this closer to 1.

    Returns
    -------
    fit_results : dict
        Dictionary containing the results of the fit.
    """
    if stepsize <= 1.0:
        raise ValueError('Stepsize needs to be > 1.0.')
    if init_sigma < 0.1:
        raise ValueError('init_sigma needs to be > 0.1.')

    img = cam.read(img_path)
    img.add_catalog(**kwargs_catalog)

    def fitness(img, ip, r, zx, zy, ao):
        pos = cam.__calib_project__(img.stars.az.values,
                                    img.stars.alt.values,
                                    ao, zx, zy, r)
        return -np.sum(ip(np.array(pos)) ** 2)
    
    w, h = cam.resolution['y'], cam.resolution['x']
    if x0 is None:
        x0 = np.array([w * 0.5,
                       h * 0.5,
                       (w + h) * 0.25,
                       90.0])
    s = init_sigma
    ip0 = RegularGridInterpolator((np.arange(w), np.arange(h)),
                                   img.image, method='linear',
                                   bounds_error=False,
                                   fill_value=0)
    if verbose: print('Starting minimization procedure ...')
    i = 1
    while True:
        if s <= 1e-1:
            break
        ip = RegularGridInterpolator((np.arange(w), np.arange(h)),
                                      gaussian(img.image, s),
                                      method='linear',
                                      bounds_error=False, fill_value=0)
        res = minimize(lambda p: fitness(img, ip, *p), x0=x0, method='powell')
        s /= stepsize
        x0 = res.x
        if verbose:
            print("Step {}: {:.3f}, {}, {}"
                  .format(i, fitness(img, ip0, *res.x), res.success, res.x))
        i += 1
    if verbose: print("Final result:\n  zenith: ({}, {})\n  radius: {}\n  azimuth offset: {}"
                      .format(*x0))
    if update:
        update_camera(cam.name, **{'zenith': {'x': float(x0[0]),
                                              'y': float(x0[1])}, 
                                   'radius': float(x0[2]),
                                   'az_offset': float(x0[3])})
    return {'zenith': {'x': x0[0],
                       'y': x0[1]}, 
            'radius': x0[2],
            'az_offset': x0[3]}


def fit_camera_params_llh(img_path, cam,
                      kwargs_catalog={'catalog':'hipparcos', 'max_mag': 6, 
                     'min_dist': 12.0, 'max_var': 2, 'min_alt': 30},
                      kwargs_method={'sigma': 0.6, 'fit_size': 20, 
                      'presmoothing': 0.0, 'remove_detected_stars': True, 
                      'fit_aim': 'calib', 'verbose': True, 'tol': 1e-15},
                      x0=None,
                      update=False,
                      verbose=False):
    """Procedure to fit camera parameters using a clear sky image.

    Parameters
    ----------
    img_path : str
        Path to a clear sky image.
    cam: quocca.camera.Camera object
        Camera.
    max_mag : float
        Maximum magnitude to be considered during the fit.
    x0 : numpy.array, shape=(4,)
        Initial guess for camera parameters. If None is passed (default), then
        some initial guess is generated from other camera parameters.

    Returns
    -------
    fit_results : dict
        Dictionary containing the results of the fit.
    """

    img = cam.read(img_path)
    img.add_catalog(**kwargs_catalog)
    det = img.detect(method='llh', **kwargs_method)

    def fitness(img, zx, zy, r, ao):
        pos = cam.__calib_project__(img.stars.az.values,
                                    img.stars.alt.values,
                                    ao, zx, zy, r)
        print(np.sum(np.sqrt(
              (det.x_fit - pos[:,0])**2 
            + (det.y_fit - pos[:,1])**2
            ) / len(img.stars)) )
        return np.sum(np.sqrt(
              (det.x_fit - pos[:,0])**2 
            + (det.y_fit - pos[:,1])**2
            ) )
    
    w, h = cam.resolution['y'], cam.resolution['x']
    if x0 is None:
        x0 = np.array([w * 0.5,
                       h * 0.5,
                       (w + h) * 0.25,
                       90.0])


    res = minimize(lambda p: fitness(img, *p), x0=x0, method='powell')
    x0 = res.x
        
    if verbose: print("Final result:\n  zenith: ({}, {})\n  radius: {}\n  azimuth offset: {}"
                      .format(*x0))
    if update:
        update_camera(cam.name, **{'zenith': {'x': float(x0[0]),
                                              'y': float(x0[1])}, 
                                   'radius': float(x0[2]),
                                   'az_offset': float(x0[3])})
    return {'zenith': {'x': x0[0],
                       'y': x0[1]}, 
            'radius': x0[2],
            'az_offset': x0[3]}


def calibrate_method(img_path, cam, method, time=0,
                     kwargs_catalog={'catalog':'hipparcos', 'max_mag': 6, 
                     'min_dist': 12.0, 'max_var': 2, 'min_alt': 30}, 
                     kwargs_method={'sigma':1.6, 'fit_size': 4}, 
                     update=False):
    """Calibrates a method for a camera, i.e. fits the response of the camera
    to a certain star detection method using a very clear night sky image.

    Parameters
    ----------
    img_path : str
        Path to clear sky image for calibration
    cam : quocca.camera.Camera object
        Camera
    method : quocca.detection object
        Method to calibrate
    verbose : bool
        Whether or not to show status messages
    update : bool
        Whether or not to update Settings automatically
    time : astropy.time.Time object or 0
        Start of calibration period. If 0 then the calibration is used as default.

    Returns
    -------
    dict
    """
    method_name = method + '_star_detection'
    img = cam.read(img_path)
    img.add_catalog(**kwargs_catalog)
    result = img.detect(method, **kwargs_method)
    result = result.merge(img.stars, on='id')
    calibration = 1.0 / np.median(result.M_fit / np.exp(-result.mag))
    with open(resource_filename('quocca', 'resources/cameras.yaml')) as file:
        yaml = YAML(typ='safe', pure=True)
        __config__ = yaml.load(file)
        __supported_cameras__ = list(__config__.keys())
    if cam.name not in __supported_cameras__:
        raise NameError('Camera {} does not exist.'
                        .format(name))
    try:
        params = __config__[cam.name][method_name]
    except KeyError:
        params = {}
    if time == 0:
        time = '1970-01-01T00:00:01.000'
    else:
        time = str(time)
    params.update({time: float(calibration)})
    if update:
        update_camera(cam.name, **{method_name: params})
    return params
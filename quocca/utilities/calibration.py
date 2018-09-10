"""quocca: All Sky Camera Analysis Tools

Calibration.

2018"""
import numpy as np

from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator

from ..catalog import Catalog
from ..image import Image

from astropy.coordinates import Angle
import astropy.units as u
from astropy.time import Time

from skimage.filters import gaussian

from ruamel import yaml

from pkg_resources import resource_filename


def project_stars(altaz, radius, zx, zy, ao):
    phi, theta = altaz.az, altaz.alt
    theta = Angle('90d') - theta
    r = np.sqrt(2.0) * radius * np.sin(theta.to(u.rad) / 2.0)
    ao = ao * u.deg
    row = -r * np.sin(phi + ao) + zx
    col = r * np.cos(phi + ao) + zy
    return row, col


def update_camera(name, **kwargs):
    with open(resource_filename('quocca', 'resources/cameras.yaml')) as file:
        __config__ = yaml.safe_load(file)
        __supported_cameras__ = list(__config__.keys())
    if name not in __supported_cameras__:
        raise NameError('Camera {} does not exist.'
                        .format(name))
    __config__[name].update(kwargs)
    res_fn = resource_filename('quocca', 'resources/cameras.yaml')
    yaml.safe_dump(__config__, open(res_fn, 'w'), default_flow_style=False)


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
    yaml.safe_dump(__config__, open(res_fn, 'w'), default_flow_style=False)


def fit_camera_params(img_path,
                      cam,
                      max_mag=3.0,
                      x0=None,
                      init_sigma=10.0,
                      stepsize=1.2,
                      update=False,
                      verbose=True):
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

    if verbose: print('Reading in catalog ...')
    cat = Catalog('hipparcos')
    if verbose: print('Reading in image ...')
    img = Image(img_path, cam, cat)
    if verbose: print('Transforming coordinates ...')
    altaz = cat.get_horizontal(cam, img.time)
    altaz = altaz[img.mask][img.star_mag < max_mag]

    def fitness(ip, r, zx, zy, ao):
        return -np.sum(ip(project_stars(altaz, r, zx, zy, ao)) ** 2)
    w, h = cam.resolution['x'], cam.resolution['y']
    if x0 is None:
        x0 = np.array([w * 0.5,
                       h * 0.5,
                       (w + h) * 0.25,
                       90.0])
    s = init_sigma
    ip0 = RegularGridInterpolator((np.arange(w), np.arange(h)),
                                   img.image, method='linear',
                                   bounds_error=False)
    if verbose: print('Starting minimization procedure ...')
    i = 1
    while True:
        if s <= 1e-1:
            break
        ip = RegularGridInterpolator((np.arange(w), np.arange(h)),
                                      gaussian(img.image, s),
                                      method='linear',
                                      bounds_error=False)
        res = minimize(lambda p: fitness(ip, *p), x0=x0, method='powell')
        s /= stepsize
        x0 = res.x
        if verbose:
            print("Step {}: {:.3f}, {}, {}"
                  .format(i, fitness(ip0, *res.x), res.success, res.x))
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


def calibrate_method(img_path, cam, method, time=0,
                     kwargs_method={}, kwargs_catalog={}, update=True):
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
    calibration = 1.0 / np.median(result.M_fit / np.exp(-result.v_mag))
    with open(resource_filename('quocca', 'resources/cameras.yaml')) as file:
        __config__ = yaml.safe_load(file)
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
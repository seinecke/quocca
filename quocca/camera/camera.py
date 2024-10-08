# quocca: QUick Observation of Cloud Coverage using All sky images
#  
# Camera classes and methods. The Camera class defines all parameters necessary
# for a sky cam analysis.
#  
# Authors: S. Einecke <sabrina.einecke@adelaide.edu.au>
#          T. Hoinka <tobias.hoinka@icecube.wisc.edu>
#          H. Nawrath <helena.nawrath@tu-dortmund.de>

import numpy as np

from ruamel.yaml import YAML
from pkg_resources import resource_filename
from imageio import imread
from scipy.interpolate import RegularGridInterpolator

from astropy.coordinates import EarthLocation

from ..image import Image
from ..utilities import calibrate_method


def r2theta_lin(r, radius):
    return r * np.pi / (2.0 * radius)


def r2theta_nonlin(r, radius):
    return 2.0 * np.arcsin(np.sqrt(2.0) * r / radius)


def theta2r_lin(theta, radius):
    return 2.0 * np.deg2rad(theta) * radius / np.pi
    #return np.deg2rad(theta) * radius


def theta2r_nonlin(theta, radius):
    return np.sqrt(2.0) * radius * np.sin(np.deg2rad(theta) / 2.0)


def rphi2pxl(r, phi, az_offset, x0, y0):
    row = -r * np.sin(np.deg2rad(phi + az_offset)) + y0
    col = r * np.cos(np.deg2rad(phi + az_offset)) + x0
    return np.column_stack((row, col))


class Camera:
    """Camera class, contains parameters specific to a camera.
    
    Attributes
    ----------
    az_offset : float
        Offset angle against north in degrees.
    location : astropy.coordinates.EarthLocation object
        Location of the camera.
    mask : scipy.interpolate.RegularGridInterpolator
        Camera-specific mask of permanently obscured objects in field of view.
    name : str
        Name of the camera.
    
    Notes
    -----
    Supported cameras are gathered from a config file located in
    `resources/cameras.yaml`. Adding, removing or altering the parameters
    of cameras requires editing that config manually.
    """

    # Loading config and listing supported camera types.
    with open(resource_filename('quocca', 'resources/cameras.yaml')) as file:
        yaml = YAML(typ='safe', pure=True)
        __config__ = yaml.load(file)
        __supported_cameras__ = list(__config__.keys())
    
    # List of attributes required for a Camera to be viable.
    __required_attributes__ = [
        'az_offset',
        'location',
        'mapping',
        'max_val',
        'radius',
        'resolution',
        'timestamps',
        'zenith'
    ]
    
    def __init__(self, name):
        """Constructor. Camera-specific settings are specified.
        If necessary, also a mask is applied, indicating regions in the field of 
        view, where buildings, nature etc obstruct the observation of the sky.
        
        Parameters
        ----------
        name : str
            Name of the camera.
        """
        if name not in self.__supported_cameras__:
            raise NotImplementedError('Unsupported Camera {}'.format(name))
        else:
            self.name = name
            # Class attributes are updated with all keys available in the
            # Config file.
            self.__dict__.update(**self.__config__[name])

            # Location and offset are converted to a astropy quantity.
            self.location = EarthLocation(**self.location)
            try:
                mask_path = resource_filename('quocca', self.mask)
                mask = np.array(imread(mask_path))[:,:,0] != 0
            except:
                mask = np.ones((self.resolution['y'], self.resolution['x']))
            tx = np.arange(mask.shape[0])
            ty = np.arange(mask.shape[1])
            self.mask = RegularGridInterpolator((tx, ty), mask,
                                                bounds_error=False,
                                                method='nearest')

            # Checking for any missing obligatory attributes.
            for attribute in self.__required_attributes__:
                if attribute not in list(self.__dict__.keys()):
                    raise KeyError('{} attribute missing in configuration.'
                                   .format(attribute))

            if self.mapping == 'lin':
                self.theta2r_fun = theta2r_lin
                self.r2theta_fun = r2theta_lin
            elif self.mapping == 'nonlin':
                self.theta2r_fun = theta2r_nonlin
                self.r2theta_fun = r2theta_nonlin
            else:
                raise NotImplementedError('Unsupported Mapping {}'
                                          .format(self.mapping))
    
    def __str__(self):
        return 'Camera {}'.format(self.name)

    def __repr__(self):
        s = "All Sky Cam '{}':\n".format(self.name)
        s += "\tLocation: ({:.2f},{:.2f})\n".format(self.location.lat,
                                            self.location.lon)
        s += "\tAzmimuth Offset: {:.2f}\n".format(self.az_offset)
        s += "\tMapping: {}\n".format(self.mapping)
        s += "\tMaximum Pixel Value: {}\n".format(self.max_val)
        s += "\tRadius: {:.2f}\n".format(self.radius)
        s += "\tResolution: {}x{}px".format(self.resolution['x'],
                                            self.resolution['y'])
        s += "\tZenith Position: ({:.2f},{:.2f})px".format(self.zenith['x'],
                                                           self.zenith['y'])
        return s

    def read(self, filename, key='pic1'):
        """Reads an image corresponding to camera defined in the object.

        Parameters
        ----------
        filename : str
            Path to image file.

        Returns
        -------
        img : quocca.image.Image object
            Image.
        """
        return Image(filename, self, key=key)

    
    def theta2r(self, theta):
        """Converts an altitude `theta` into a apparent radius in pixels. Used
        function depends on attribute `mapping`. Currently, only
        `lin` and `nonlin` are supported.
        
        Parameters
        ----------
        theta : float or numpy.array
            Altitude angle in degrees.
        
        Returns
        -------
        r : numpy.array
            Calculated radius in pixels.
        """
        return self.theta2r_fun(theta, self.radius)

        
    def r2theta(self, r):
        """Converts a radius in pixels `r` into an altitude. Used function
        depends on attribute `mapping`. Currently, only `lin` and
        `nonlin` are supported.
        
        Parameters
        ----------
        r : numpy.array
            Radius in pixels.
        
        Returns
        -------
        theta : numpy.array
            Calculated altitude in degrees.
        """
        return self.r2theta_fun(theta, self.radius)
        
    def check_mask(self, x, y):
        """Checks whether or not if a point is within the bounds of the mask provided. True
        means the checked point should not be obstructed by a permanently present
        object.
        
        Parameters
        ----------
        x : numpy.array
            x coordinate in pixels.
        y : numpy.array
            y coordinate in pixels.
        
        Returns
        -------
        numpy.array, dtype=bool
            Whether or not the points are obstructed.
        """
        return self.mask((x, y)) == 1
        
    def __project_stars__(self, phi, theta):
        r = self.theta2r(90.0 - theta)
        row = -r * np.sin(phi + self.az_offset) + self.zenith['x']
        col = r * np.cos(phi + self.az_offset) + self.zenith['y']
        return row, col

    def __calib_project__(self, phi, theta, az_offset,
                          zenith_x, zenith_y, radius):
        r = self.theta2r_fun(90.0 - theta, radius)
        pxl = rphi2pxl(r, phi, az_offset, zenith_x,
                       zenith_y)
        return pxl

    def project_stars(self, catalog, time):
        """Projects stars listed in `catalog` onto the camera at `time`.
        
        Parameters
        ----------
        catalog : quocca.catalog.Catalog object
            Catalog containing the stars to be projected.
        time : astropy.time.Time object
            Timestamp.
        
        Returns
        -------
        pxl : numpy.array, shape=(n_stars, 2)
            Calculated pixel coordinates for each star.
        altaz : numpy.array, shape=(n_stars, 2)
            Alt/Az-Coordinates of each star.
        """
        altaz = catalog.get_horizontal(self, time)
        phi, theta = altaz.az, altaz.alt
        r = self.theta2r(90.0 - theta)
        pxl = rphi2pxl(r, phi, self.az_offset, self.zenith['x'],
                       self.zenith['y'])
        return pxl, altaz

    def calibrate(self,
                  img,
                  method,
                  time=0,
                  update=True,
                  kwargs_catalog={},
                  kwargs_method={}):
        """Calibrate a method for this camera.

        Parameters
        ----------
        img : str
            Path to clear calibration image.
        method : str or quocca.detection.StarDetection object
            Method to calibrate for
        time : 0 or astropy.time.Time object
            Time of the begin of the calibration period.
        kwargs_method : dict
            Keyword arguments for method initialisation.
        kwargs_catalog : dict
            Keyword arguments for catalog initialisation.
        """
        return calibrate_method(img, self, method, time, kwargs_catalog,
                                kwargs_method, update=update)

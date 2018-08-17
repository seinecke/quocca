"""quocca: All Sky Camera Analysis Tools

Cameras.

2018"""

import numpy as np

from ruamel import yaml
from pkg_resources import resource_filename
from imageio import imread
from scipy.interpolate import RegularGridInterpolator

from astropy import units as u
from astropy.coordinates import Angle, EarthLocation


class Camera:
    """Camera class, contains parameters specific to a camera.
    
    Attributes
    ----------
    az_offset : astropy.coordinates.Angle object
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

    with open(resource_filename('quocca', 'resources/cameras.yaml')) as file:
        __config__ = yaml.safe_load(file)
        __supported_cameras__ = list(__config__.keys())
    
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
            self.__dict__.update(**self.__config__[name])
            self.location = EarthLocation(**self.location)
            self.az_offset = Angle(self.az_offset * u.deg)
            mask_path = resource_filename('quocca', self.mask)
            mask = np.array(imread(mask_path)) != 0
            tx = np.arange(mask.shape[0])
            ty = np.arange(mask.shape[1])
            self.mask = RegularGridInterpolator((tx, ty), mask,
                                                bounds_error=False,
                                                method='nearest')
    
    def __str__(self):
        return 'Camera {}'.format(self.name)
    
    def theta2r(self, theta):
        """Converts an altitude `theta` into a apparent radius in pixels. Used
        function depends on attribute `mapping`. Currently, only
        `lin` and `nonlin` are supported.
        
        Parameters
        ----------
        theta : astropy.coordinates.Angle
            Altitude angle in degrees.
        
        Returns
        -------
        r : numpy.array
            Calculated radius in pixels.
        """
        if self.mapping == 'lin':
            return self.radius / (np.pi * u.rad / 2.0) * theta.to(u.rad)
        elif self.mapping == 'nonlin':
            return np.sqrt(2.0) * self.radius * np.sin(theta.to(u.rad) / 2.0)
        else:
            raise NotImplementedError('Unsupported Mapping {}'
                                      .format(self.mapping))
        
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
        if self.mapping == 'lin':
            return r * (np.pi * u.rad) / (2.0 * self.radius)
        elif self.mapping == 'nonlin':
            return 2.0 * np.arcsin(np.sqrt(2.0) * r / self.radius)
        
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
        pixel_coordinates : numpy.array, shape=(n_stars, 2)
            Calculated pixel coordinates for each star.
        magnitude : numpy.array
            Magnitude of each star.
        """
        altaz = catalog.get_horizontal(self, time)
        phi, theta = altaz.az, altaz.alt
        r = self.theta2r(Angle('90d') - theta)
        row = -r * np.sin(phi + self.az_offset) + self.zenith['x']
        col = r * np.cos(phi + self.az_offset) + self.zenith['y']
        return np.column_stack((row, col)), catalog['v_mag']
        
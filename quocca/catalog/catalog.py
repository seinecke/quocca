"""quocca: All Sky Camera Analysis Tools

Catalogs.

2018"""

import numpy as np
from ruamel import yaml
from pkg_resources import resource_filename

from astropy.table import Table
from astropy.constants import atm
from astropy.coordinates import SkyCoord, AltAz


class Catalog(Table):
    with open(resource_filename('quocca', 'resources/catalogs.yaml')) as file:
        __config__ = yaml.safe_load(file)
        __supported_catalogs__ = list(__config__.keys())
        
    def __init__(self, name):
        if name not in self.__supported_catalogs__:
            raise NotImplementedError('Unsupported Catalog {}'.format(name))
        super(Catalog, self).__init__(Table.read(resource_filename('quocca', self.__config__[name]['file'])))
        self.remove_rows(np.isnan(self['ra']) | np.isnan(self['dec']))
        self.id = self[self.__config__[name]['id']]
        self.var = self[self.__config__[name]['var']]
        self.ra = self[self.__config__[name]['ra']]
        self.dec = self[self.__config__[name]['dec']]
        self.mag = self[self.__config__[name]['mag']]
    
    def get_horizontal(self, camera, time):
        """Transforms ra/dec coordinates from catalog into alt-az coordinates.

        Parameters
        ----------
        camera : quocca.camera.Camera object
            Camera for which to transform coordinates for.
        time : astropy.time.Time object
            Timestamp.

        Returns:
        --------
        pos_altaz : astropy.coordinates.sky_coordinate.SkyCoord object
            Positions in altitude and azimuth.
        """
        pos = SkyCoord(ra=self['ra'], dec=self['dec'],
                       frame='icrs', unit='deg')
        pos_altaz = pos.transform_to(AltAz(obstime=time,
                                           location=camera.location,
                                           pressure=atm))
        return pos_altaz
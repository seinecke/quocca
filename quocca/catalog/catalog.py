"""quocca: All Sky Camera Analysis Tools

Catalogs.

2018"""

import numpy as np
import pandas as pd
from ruamel import yaml
from pkg_resources import resource_filename

from astropy.table import Table
from astropy.constants import atm
from astropy.coordinates import SkyCoord, AltAz


class Catalog(pd.DataFrame):
    with open(resource_filename('quocca', 'resources/catalogs.yaml')) as file:
        __config__ = yaml.safe_load(file)
        __supported_catalogs__ = list(__config__.keys())

    def __init__(self, name):
        if name not in self.__supported_catalogs__:
            raise NotImplementedError('Unsupported Catalog {}'.format(name))
        table = Table.read(resource_filename('quocca',
                                             self.__config__[name]['file']))
        table.remove_rows(np.isnan(table[self.__config__[name]['ra']])
                        | np.isnan(table[self.__config__[name]['dec']]))
        table = table.to_pandas()
        attributes = {
            'id': np.array(table[self.__config__[name]['id']]).astype(int),
            'variability': np.array(table[self.__config__[name]['var']]),
            'ra': np.array(table[self.__config__[name]['ra']]),
            'dec': np.array(table[self.__config__[name]['dec']]),
            'mag': np.array(table[self.__config__[name]['mag']])
        }
        super(Catalog, self).__init__(attributes,
                                      index=attributes['id'],
                                      copy=True)

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
        pos = SkyCoord(ra=self.ra.values, dec=self.dec.values,
                       frame='icrs', unit='deg')
        altaz = pos.transform_to(AltAz(obstime=time,
                                       location=camera.location,
                                       pressure=atm))
        return pd.DataFrame({'alt': np.array(altaz.alt),
                             'az': np.array(altaz.az)},
                             index=self.id.values)
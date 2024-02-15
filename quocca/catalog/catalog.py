# quocca: QUick Observation of Cloud Coverage using All sky images
#  
# Defines the Catalog class, which is pretty much just a pandas DataFrame with
# a few additional tweaks to facilitate handling.  
#  
# Authors: S. Einecke <sabrina.einecke@adelaide.edu.au>
#          T. Hoinka <tobias.hoinka@icecube.wisc.edu>
#          H. Nawrath <helena.nawrath@tu-dortmund.de>

import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from pkg_resources import resource_filename

from astropy.table import Table
from astropy.constants import atm
from astropy.coordinates import SkyCoord, AltAz


class Catalog(pd.DataFrame):
    """Catalog class. Basically a pandas dataframe with few additional features
    to facilitate the handling of star catalogs.

    Attributes
    ----------
    id : pandas.DataFrame column
        Some id for each star, e.g. the HIP
    variability : pandas.DataFrame column
        Some qualifier for the amount of variability of a star.
    ra, dec : pandas.DataFrame column
        Ra/Dec coordinates in deg.
    mag : pandas.DataFrame column
        The magnitude of each star.
    """
    with open(resource_filename('quocca', 'resources/catalogs.yaml')) as file:
        yaml = YAML(typ='safe', pure=True)
        __config__ = yaml.load(file)
        __supported_catalogs__ = list(__config__.keys())

    def __init__(self, name):
        """Inilialize a catalog.

        Parameters
        ----------
        name : str
            Name of the catalog. See Catalog.__supported_catalogs__ for which
            catalogs are supported.
        """
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
        pos_altaz : pandas.DataFrame
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
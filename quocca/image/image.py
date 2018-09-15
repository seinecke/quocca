"""quocca: All Sky Camera Analysis Tools

Images.

2018"""

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.time import Time
from astropy.io import fits
from astropy.coordinates import get_body, AltAz

import scipy.io as sio
from scipy.spatial import distance_matrix, cKDTree

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm

from skimage.filters import gaussian

from ..plotting import show_img
from ..catalog import Catalog
from ..detection import StarDetectionLLH, StarDetectionFilter


def nearby_stars(x, y, mag, radius):
    """Masks all stars that have a brigther star closer
    than `radius`.

    Parameters
    ----------
    x, y : numpy.array
        Positions
    mag : numpy.array
        Magnitudes
    radius : float
        Radius

    Returns
    -------
    mask : numpy.array, dtype=bool
        Boolean mask that `True` whenever a brighter star is NOT nearby.
    """
    magnitude_sorted = np.argsort(-mag)
    pos = np.column_stack((x, y))
    kdtree = cKDTree(pos)
    closest = kdtree.query_ball_point(pos, radius, 1)
    mask = np.ones(len(mag), dtype=bool)
    checked = np.zeros(len(mag), dtype=bool)
    for i in magnitude_sorted:
        if mask[i] and len(closest[i]) > 1:
            for j in range(1, len(closest[i])):
                if checked[closest[i][j]]:
                    mask[closest[i][j]] = False
        checked[i] = True
    return mask


class Image:
    """Class containing image and auxilliary information.

    Attributes
    ----------
    image : numpy.array, shape=(width, height)
        Image, normalized to [0,1].
    time : astropy.time.Time object
        Timestamp.
    camera : quocca.camera.Camera
        Camera the image is associated with.
    star_pos, star_mag : numpy.array
        Position in pixels and magnitude of the stars potentially visible.
    
    Notes
    -----
    Currently only `mat`, `fits.gz` and `fits` files are supported.
    """
    __supported_formats__ = ['mat', 'gz', 'fits']

    def __init__(self, path, camera):
        """Constructor of Image.
        
        Parameters
        ----------
        path : str
            Path to image file.
        camera : quocca.camera.Camera object
            Camera associated with the image.
        """
        suffix = path.split(".")[-1]
        if suffix not in self.__supported_formats__:
            raise NotImplementedError("Unsupported Filetype {}".format(suffix))

        if suffix == 'fits' or suffix == 'gz':
            with fits.open(path) as f:
                image = f[0].data
                for timestamp in camera.timestamps:
                    try:
                        timestamp = f[0].header[timestamp]
                        time = Time(timestamp, scale='utc')
                    except KeyError:
                        continue

        if suffix == 'mat':
            matfile = sio.loadmat(path)
            image = matfile['pic1']
            for timestamp in camera.timestamps:
                try:
                    timestamp = matfile[timestamp]
                    date, time = timestamp[0].split(' ')
                    date = date.replace('/', '-')
                    time = Time(date + str('T') + time, scale='utc')
                except KeyError:
                    continue

        image[np.isnan(image)] = 0.0
        self.image = np.clip(image / camera.max_val, 0.0, 1.0)
        self.time = time
        self.camera = camera
        self.stars = None

    def __repr__(self):
        self.show()
        plt.show()
        return "All Sky Camera Image by '{}' on {}.".format(self.camera.name,
                                                            self.time)

    def add_catalog(self, catalog='hipparcos', max_mag=5.5, min_dist=10.0):
        """Adds a catalog to the image.

        Parameters
        ----------
        catalog : str or quocca.catalog.Catalog object
            Catalog name or Catalog object.
        max_mag : float
            Maximum magnitude.
        min_dist : float
            Minimal distance between one star to the next.
        """
        if type(catalog) == str:
            catalog = Catalog(catalog)
        star_pos, star_altaz = self.camera.project_stars(catalog, self.time)
        mask_nearby = nearby_stars(star_pos[:,0],
                                   star_pos[:,1],
                                   catalog.mag, min_dist or 0.0)

        mask_mag = catalog.mag < (max_mag or np.inf)
        mask_obscur = self.camera.check_mask(*star_pos.T) == 1
        mask = mask_obscur & mask_mag & mask_nearby 
        self.stars = pd.DataFrame({'id': catalog.id[mask].astype(int),
                                   'x': star_pos[mask, 0],
                                   'y': star_pos[mask, 1],
                                   'alt': np.array(star_altaz.alt)[mask],
                                   'az': np.array(star_altaz.az)[mask],
                                   'mag': catalog.mag[mask],
                                   'var': catalog.var[mask]},
                                   index=catalog.id[mask].astype(int))

    def rm_celestial_bodies(self, radius=10.0, bodies=['moon', 'venus', 'mars',
                                                       'mercury', 'jupiter',
                                                       'saturn']):
        """Removes stars in the vicinity of bright celestial bodies from the
        image.

        Parameters
        ----------
        radius : float
            Radius in pixels
        bodies : list(str)
            List of names of celestial bodies. For allowed values see
            astropy.coordinates.get_body.
        """
        altaz_cs = AltAz(obstime=self.time, location=self.camera.location)
        objects = [
            get_body(p, self.time, self.camera.location).transform_to(altaz_cs)
            for p in bodies
        ]
        px_pos = np.array([
            np.array(self.camera.__project_stars__(p.az, p.alt))
            for p in objects
        ])
        D = distance_matrix(self.star_pos, px_pos)
        choice = np.all(D > radius, axis=1)
        self.star_pos = self.star_pos[choice, :]
        self.star_mag = self.star_mag[choice]


    def detect(self, method='llh', **kwargs):
        """Detects stars in the image.

        Parameters
        ----------
        method : str or quocca.detect.StarDetection object
            The method to use. The following methods are available:
                * `llh`: Likelihood fit of all found stars.
                * `filter`: LoG filter

        kwargs : keywords
            Initialisation keywords.

        Returns
        -------
        results : pd.DataFrame
            Results in a DataFrame. 
        """
        if self.stars is None:
            raise AttributeError('No catalog added to image. Use add_catalog.')
        if type(method) == str:
            if method == 'llh':
                det = StarDetectionLLH(self.camera, **kwargs)
            elif method == 'filter':
                det = StarDetectionFilter(self.camera, **kwargs)
            else:
                raise ValueError('Unsupported method {}'.format(method))
        results = det.detect(self)
        return results

    def show(self, **kwargs):
        """Convenience wrapper for quocca.plotting.show_img.
        """
        ax = show_img(self, **kwargs)
        return ax
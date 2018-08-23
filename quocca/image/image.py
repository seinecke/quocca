"""quocca: All Sky Camera Analysis Tools

Images.

2018"""

import numpy as np

from astropy import units as u
from astropy.time import Time
from astropy.io import fits

import scipy.io as sio

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm

from skimage.filters import gaussian

from ..plotting import show_img


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

    def __init__(self, path, camera, catalog):
        """Constructor of Image.
        
        Parameters
        ----------
        path : str
            Path to image file.
        camera : quocca.camera.Camera object
            Camera associated with the image.
        catalog : quocca.catalog.Catalog object
            Star catalog.
        """
        suffix = path.split(".")[-1]
        if suffix not in self.__supported_formats__:
            raise NotImplementedError("Unsupported Filetype {}".format(suffix))

        if suffix == 'fits' or suffix == 'gz':
            with fits.open(path) as f:
                image = f[0].dataa
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
        self.star_pos, self.star_mag = self.camera.project_stars(catalog,
                                                                 self.time)
        self.mask = camera.check_mask(*self.star_pos.T) == 1
        self.star_pos = np.array(self.star_pos[self.mask, :])
        self.star_mag = np.array(self.star_mag[self.mask])

    def __repr__(self):
        self.show()
        return "All Sky Camera Image by '{}' on {}.".format(self.camera.name,
                                                            self.time)

    def show(self, **kwargs):
        """Convenience wrapper for quocca.plotting.show_img.
        """
        ax = show_img(self, **kwargs)
        return ax
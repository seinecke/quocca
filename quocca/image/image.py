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

    def show(self, upper=90.0, show_stars=5.0):
        """Shows the image.
        
        Parameters
        ----------
        upper : float, default=90.0
            Quantile at which to clip the image at the upper end. Increase
            when displayed image appears to be overexposed and vice-versa.
        show_stars : float, default=5.0
            Magnitude limit of stars. Set to some super low value to show no
            stars at all.
        """
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(self.image, vmin=0.0, vmax=np.percentile(self.image, upper),
                  cmap='gray')
        display = self.star_mag < show_stars
        ax.scatter(self.star_pos[display, 1], self.star_pos[display, 0], s=100,
                   marker='o', facecolor='', edgecolor='w')
        angles = [15, 30, 45, 60, 75, 90]
        circles = [Circle((self.camera.zenith['x'],
                          self.camera.zenith['y']),
                         self.camera.theta2r(angle * u.deg),
                         facecolor='none', ec='w', linestyle=':')
                   for angle in angles]
        circles[-1].set_linestyle('-')
        for angle in angles:
            plt.text(self.camera.zenith['x'],
                     self.camera.zenith['y']\
                     + self.camera.theta2r(angle * u.deg) - 10,
                     "{}".format(angle), color='w')
        for c in circles:
            ax.add_patch(c)
            plt.axis('off')
        return fig, ax
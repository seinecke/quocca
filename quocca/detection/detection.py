"""quocca: All Sky Camera Analysis Tools

Detection.

2018"""

import numpy as np
import pandas as pd

from progressbar import progressbar
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.ndimage import convolve
from skimage.filters import gaussian
from ruamel import yaml
from pkg_resources import resource_filename
import warnings
from astropy.time import Time


def get_slice(pos, size, shape):
    pos = list(np.round(pos).astype(int))
    a_min = np.clip(pos[1] - size[1], 0, shape[0] - 1)
    a_max = np.clip(pos[1] + size[1] + 1, 0, shape[0] - 1)
    b_min = np.clip(pos[0] - size[0], 0, shape[1] - 1)
    b_max = np.clip(pos[0] + size[0] + 1, 0, shape[1] - 1)
    return (slice(a_min, a_max, None),
            slice(b_min, b_max, None))


def laplacian_gaussian_filter(img, sigma, prec=1e-16):
    """Laplacian of the Gaussian filter.

    Parameters
    ----------
    img : numpy.array
        Image.
    sigma : float
        Spread parameter of the filter.
    prec : float, default=1e-16
        Values of the kernel lower than this value are cut away.

    Returns
    -------
    filtered_img : numpy.array
        Filtered image.
    """
    tx = np.arange(img.shape[0]) - img.shape[0] * 0.5
    ty = np.arange(img.shape[0]) - img.shape[0] * 0.5
    mx, my = np.meshgrid(tx, ty)
    r2 = (mx ** 2 + my ** 2) / (2.0 * sigma ** 2)
    kernel = 1.0 / (np.pi * sigma ** 4) * (1.0 - r2) * np.exp(-r2)
    kernel /= np.max(kernel)
    kernelsum = np.max(np.abs(kernel), axis=0)
    below_prec = np.where(kernelsum > prec)[0]
    lower = below_prec[0]
    upper = below_prec[-1]
    kernel = kernel[lower:upper, lower:upper]
    return convolve(img, kernel)


def mean_cov(mx, my, M):
    """Calculates mean and covariance of a matrix `M`.

    Parameters
    ----------
    mx, my : numpy.array
        Meshgrid matrices
    M : numpy.array
        Matrix

    Returns
    -------
    mean : numpy.array
        Mean vector
    cov : numpy.array
        Covariance matrix
    """
    norm = np.sum(M)
    mean_x = np.sum(mx * M) / norm
    mean_y = np.sum(my * M) / norm
    mean_xx = np.sum(mx ** 2 * M) / norm
    mean_yy = np.sum(my ** 2 * M) / norm
    mean_xy = np.sum(mx * my * M) / norm
    mean = np.array([mean_x, mean_y])
    cov = np.array([[mean_xx - mean_x ** 2, mean_xy - mean_x * mean_y],
                    [mean_xy - mean_x * mean_y, mean_yy - mean_y ** 2]])
    return mean, cov


def get_calibration(cam_name, meth_name, time):
    """Gathers a suitable calibration for a camera and method at a given time.

    Parameters
    ----------
    cam_name : str
        Camera name
    meth_name : str
        Method name
    time : astropy.time.Time object
        Time

    Returns
    -------
    calibration : float
        Calibration
    """
    with open(resource_filename('quocca', 'resources/cameras.yaml')) as file:
        config = yaml.safe_load(file)
        try:
            calibration = config[cam_name][meth_name]
            calib_keys = list(calibration.keys())
            times = {c: Time(c) for c in calib_keys}
            available = False
            chosen_key = None
            for c, t in times.items():
                if time > t:
                    if chosen_key is None:
                        chosen_key = c
                    elif time - t < time - times[chosen_key]:
                        chosen_key = c
            try:
                return calibration[chosen_key]
            except:
                warnings.warn('No calibration setting found.')
                return 1.0


        except KeyError:
            warnings.warn('No calibration setting found.')
            return 1.0


class StarDetectionBase:
    """StarDetection base class. Do not use directly, only use inherited
    classes."""
    name = 'base_star_detection'

    def __init__(self, camera, **kwargs):
        self.camera = camera

    def detect(self, image, **kwargs):
        self.calibration = get_calibration(self.camera.name,
                                           self.name,
                                           image.time)

class StarDetectionLLH(StarDetectionBase):
    """Likelihood approach to star detection. Fits every star individually
    using a simple model.
    
    Attributes
    ----------
    fit_pos : bool
        Whether or not to fit the position of the star
        [WARNING: Not implemented!]
    fit_sigma : bool
        Whether or not to fit the shape of the star
        [WARNING: Not implemented!]
    name : str
        Name of the method
    presmoothing : float
        Sigma parameter of the Gaussian filter applied beforehand.
    sigma : float
        Sigma parameter of the model used during fit.
    size : tuple(int)
        Size of the cropped out parts of the image for fitting.
    """
    name = 'llh_star_detection'

    def __init__(self, camera, sigma=1.7, fit_size=8, fit_pos=True,
                 fit_sigma=False, presmoothing=1.5,
                 remove_detected_stars=False, verbose=True):
        super(StarDetectionLLH, self).__init__(camera)
        self.sigma = sigma
        self.size = (fit_size, fit_size)
        self.fit_pos = fit_pos
        self.fit_sigma = fit_sigma
        self.presmoothing = presmoothing
        self.verbose = verbose
        self.remove_detected_stars = remove_detected_stars

    def get_slice(self, pos, shape):
        pos = list(np.round(pos).astype(int))
        a_min = np.clip(pos[1] - self.size[1], 0, shape[0] - 1)
        a_max = np.clip(pos[1] + self.size[1] + 1, 0, shape[0] - 1)
        b_min = np.clip(pos[0] - self.size[0], 0, shape[1] - 1)
        b_max = np.clip(pos[0] + self.size[0] + 1, 0, shape[1] - 1)
        return (slice(a_min, a_max, None),
                slice(b_min, b_max, None))
    
    def blob_func(self, x, y, x0, y0, mag, sigma, bkg):
        arg = -((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2)
        return np.abs(mag) * np.exp(arg) + np.abs(bkg)

    """def blob_func(self, x, y, x0, y0, mag, sx, sy, rho, bkg):
        x_ = x - x0
        y_ = y - y0
        upper = sy ** 2 * x_ ** 2 - 2.0 * rho * sx * sy * x_ * y_ + sx ** 2\
                * y_ ** 2
        det2 = (rho **2 - 1) * sx ** 2 * sy ** 2
        arg = upper / det2
        return np.clip(np.abs(mag) * np.exp(arg) + np.abs(bkg), 0.0, 1.0)"""
    
    def detect(self, image):
        """Detect method.
        
        Parameters
        ----------
        image : quocca.image.Image
            Image.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing all the results, namely
                * `M_fit`: Fitted linear magnitude.
                * `b_fit`: Fitted background level.
                * `x_fit`: Fitted x-position.
                * `y_fit`: Fitted y-position.
                * `v_mag`: Expected (logarithmic) magnitude.
                * `x`: Expected x-position.
                * `y`: Expected y-position.
                * `visibility`: Calculated visibility factor.
        """
        super(StarDetectionLLH, self).detect(image)
        img = gaussian(image.image, self.presmoothing)
        tx = np.arange(img.shape[0])
        ty = np.arange(img.shape[1])
        mx, my = np.meshgrid(ty, tx)
        n_stars = len(image.stars)
        pos = np.column_stack((image.stars.x.values,
                               image.stars.y.values))
        keys = [
            'id',
            'M_fit',
            'b_fit',
            'x_fit',
            'y_fit',
            'visibility'
        ]
        results = {key: np.zeros(n_stars) for key in keys}

        # Sort by magnitude to process stars ordered by magnitude.
        mag_sort_idx = np.argsort(image.stars.mag.values)
        if self.verbose:
            iterator = progressbar(mag_sort_idx, total=len(mag_sort_idx))
        else:
            iterator = mag_sort_idx
        for idx in iterator:
            sel = get_slice((pos[idx,1], pos[idx,0]), self.size, img.shape)
            def fit_function(p):
                return np.sum((
                    self.blob_func(mx[sel], my[sel], p[2], p[3], p[0],
                                   self.sigma, p[1]) - img[sel]) ** 2
                )
            r = minimize(fit_function,
                         x0=[np.max(img[sel]) - np.mean(img[sel]), 
                             np.mean(img[sel]),
                             pos[idx,1], pos[idx,0]],
                         method='Powell')
            if self.remove_detected_stars:
                img[sel] -= self.blob_func(mx[sel], my[sel], r.x[2], r.x[3],
                                           r.x[0], self.sigma, 0.0)
            visibility = np.abs(r.x[0]) / np.exp(-image.stars.mag.iloc[idx])
            results['id'][idx] = image.stars.id.iloc[idx]
            results['M_fit'][idx] = np.abs(r.x[0])
            results['visibility'][idx] = visibility * self.calibration
            results['b_fit'][idx] = np.abs(r.x[1])
            results['y_fit'][idx] = r.x[2]
            results['x_fit'][idx] = r.x[3]
        return pd.DataFrame(results, index=results['id'].astype(int))


class StarDetectionFilter(StarDetectionBase):
    name = 'filter_star_detection'

    def __init__(self, camera, sigma=1.0, fit_size=5, quantile=100.0,
                 verbose=True):
        super(StarDetectionFilter, self).__init__(camera)
        self.sigma = sigma
        self.size = (fit_size, fit_size)
        self.quantile = quantile
        self.verbose = verbose
    
    def detect(self, image):
        super(StarDetectionFilter, self).detect(image)
        from matplotlib import pyplot as plt
        img = laplacian_gaussian_filter(image.image, self.sigma)

        tx = np.arange(img.shape[0])
        ty = np.arange(img.shape[1])
        mx, my = np.meshgrid(tx, ty)

        n_stars = len(image.stars)
        pos = np.column_stack((image.stars.x.values,
                               image.stars.y.values))
        results = {
            key: np.zeros(n_stars)
            for key in ['id', 'M_fit', 'visibility']
        }
        iterator = progressbar(range(n_stars)) if self.verbose else range(n_stars)
        for idx in iterator:
            sel = get_slice((pos[idx,1], pos[idx,0]), self.size, img.shape)
            M = np.percentile(img[sel], self.quantile)

            results['id'][idx] = image.stars.id.iloc[idx]
            results['M_fit'][idx] = M
            visibility = M / np.exp(-image.stars.mag.iloc[idx])
            results['visibility'][idx] = visibility * self.calibration
        return pd.DataFrame(results, index=results['id'].astype(int))

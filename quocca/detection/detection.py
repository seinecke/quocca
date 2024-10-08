# quocca: QUick Observation of Cloud Coverage using All sky images
#  
# Star Detection classes and methods. So far two detection methods are
# implemented:
#  
# 1. StarDetectionLLH [T. Hoinka]
#    Fits a star model to every expected star position in the image.
# 2. StarDetectionFilter [J. Adam]
#    Filters the image using a LoG Filter and then checks the brightness at
#    the position of expected stars.
#  
# Authors: S. Einecke <sabrina.einecke@adelaide.edu.au>
#          T. Hoinka <tobias.hoinka@icecube.wisc.edu>
#          H. Nawrath <helena.nawrath@tu-dortmund.de>

import numpy as np
import pandas as pd

from progressbar import progressbar
from tqdm import tqdm

from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.ndimage import convolve
from scipy.special import expit

from skimage.filters import gaussian
from skimage.feature import blob_log

from ruamel.yaml import YAML
from pkg_resources import resource_filename
import warnings
from astropy.time import Time


def laplacian_gaussian_filter(img, sigma, prec=1e-16):
     """Laplacian of the Gaussian filter using analytical description.
     Note: scipy.ndimage.gaussian_laplacian is similar, but works way worse.

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


def crop(pos, size, shape):
    """Crops out a snippet from an image surrounding a position `pos`.

    Parameters
    ----------
    pos : numpy.array, shape=(n,)
        Position
    size : tuple
        Size of the cropped image.
    shape : tuple
        Shape of the image.

    Returns
    -------
    slice : tuple(slice, slice)
        Tuple of slices to crop the image.
    """
    pos = list(np.round(pos).astype(int))
    a_min = np.clip(pos[1] - size[1], 0, shape[0] - 1)
    a_max = np.clip(pos[1] + size[1] + 1, 0, shape[0] - 1)
    b_min = np.clip(pos[0] - size[0], 0, shape[1] - 1)
    b_max = np.clip(pos[0] + size[0] + 1, 0, shape[1] - 1)
    return (slice(a_min, a_max, None),
            slice(b_min, b_max, None))


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
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(file)
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


def blob_func(x, y, x0, y0, mag, sigma, bkg):
    """Model function for the star.

    Parameters
    ----------
    x, y : numpy.array
        Coordinates for evaluation
    x0, y0 : float
        Location of the star.
    mag : float
        Magnitude of the star.
    sigma : float
        Spread of the star.
    bkg : float
        Background level.

    Returns
    -------
    model : np.array
        Model.
    """
    arg = -((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2)
    return np.clip(np.abs(mag) * np.exp(arg) + np.abs(bkg), 0, 1)


def blob_bivar_func(x, y, x0, y0, mag, sigx, sigy, cor, bkg):
    """Bivariate Gaussian model for the star.

    Parameters
    ----------
    x, y : numpy.array
        Coordinates for evaluation
    x0, y0 : float
        Location of the star.
    mag : float
        Magnitude of the star.
    sigma : float
        Spread of the star.
    bkg : float
        Background level.

    Returns
    -------
    model : np.array
        Model.    
    """

    arg = -( (1 / (2 * (1 - cor**2)) ) * \
        ( ( (x - x0) / sigx)**2 
            - 2 * cor * ( ( (x - x0) / sigx) * ( (y - y0) / sigy))
            +( (y - y0) / sigy)**2) )
    return np.clip(np.abs(mag) * np.exp(arg) + np.abs(bkg), 0, 1)


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
    name : str
        Name of the method
    presmoothing : float
        Sigma parameter of the Gaussian filter applied beforehand.
    sigma : float
        Sigma parameter of the model used during fit.
    size : tuple(int)
        Size of the cropped out parts of the image for fitting.
    verbose : bool
        Verbosity flag.
    remove_detected_stars : bool
        Whether or not to successively remove fitted stars as they are
        detected.
    """
    name = 'llh_star_detection'

    def __init__(self, camera, sigma=1.6, fit_size=4, presmoothing=0.0,
                 remove_detected_stars=True, fit_aim='detect', 
                 verbose=True, tol=1e-15):
        super(StarDetectionLLH, self).__init__(camera)
        self.sigma = sigma
        self.size = (fit_size, fit_size)
        self.presmoothing = presmoothing
        self.verbose = verbose
        self.remove_detected_stars = remove_detected_stars
        self.tol = tol
        self.fit_aim = fit_aim
    
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
                * `id`: Id of each star.
                * `M_fit`: Fitted linear magnitude.
                * `b_fit`: Fitted background level.
                * `x_fit`: Fitted x-position.
                * `y_fit`: Fitted y-position.
                * `visibility`: Calculated visibility factor.
        """
        super(StarDetectionLLH, self).detect(image)

        # Apply presmoothing.
        img = gaussian(image.image, self.presmoothing)
        
        # Generate meshgrid for pixel coordinate grid.
        tx = np.arange(img.shape[0])
        ty = np.arange(img.shape[1])
        mx, my = np.meshgrid(ty, tx)
        n_stars = len(image.stars)
        pos = np.column_stack((image.stars.x.values,
                               image.stars.y.values))

        # Prepare keys to write during the detection.
        keys = [
            'id',
            'M_fit',
            'b_fit',
            'x_fit',
            'y_fit',
            #'sig_fit',
            #'sigx_fit',
            #'sigy_fit',
            #'cor_fit',
            'success_fit',
            'llh_fit',
            'iter_fit',
            'saturated',
            'visibility'
        ]
        results = {key: np.zeros(n_stars) for key in keys}

        # Sort by magnitude to process stars ordered by magnitude.
        mag_sort_idx = np.argsort(image.stars.mag.values)
        if self.verbose:
            iterator = tqdm(mag_sort_idx, total=len(mag_sort_idx))
        else:
            iterator = mag_sort_idx
        for idx in iterator:
            sel = crop((pos[idx,1], pos[idx,0]), self.size, img.shape)
            sel_max = np.max(img[sel])
            sel_mean = np.mean(img[sel])
            xy = np.unravel_index(np.argmax(img[sel]), img[sel].shape)

            if self.fit_aim == 'detect':
                def fit_function(p):
                    return np.sum((
                        blob_func(mx[sel], my[sel], p[0], p[1], p[2],
                                  self.sigma, p[3]) - img[sel]) ** 2
                        / (img[sel] + 1e-8)
                    )
                x0 = [pos[idx,1], pos[idx,0], 
                      sel_max - sel_mean, sel_mean]
                bounds = (
                    (pos[idx,1]-self.size[1], pos[idx,1]+self.size[1]),
                    (pos[idx,0]-self.size[0], pos[idx,0]+self.size[0]),
                    (0.0, 2*sel_max),
                    (0.0, sel_max),
                    )
            elif self.fit_aim == 'calib':
                def fit_function(p):
                    return np.sum((
                        blob_func(mx[sel], my[sel], p[0], p[1], p[2],
                                  self.sigma, p[3]) - img[sel]) ** 2
                        / (img[sel] + 1e-8)
                    )
                x0 = [mx[sel][xy], my[sel][xy],
                      sel_max - sel_mean, sel_mean]
                bounds = (
                    (pos[idx,1]-self.size[1], pos[idx,1]+self.size[1]),
                    (pos[idx,0]-self.size[0], pos[idx,0]+self.size[0]),
                    (0.0, 2*sel_max),
                    (0.0, sel_max),
                    )
            elif self.fit_aim == 'psf':
                def fit_function(p):
                    return np.sum((
                        blob_func(mx[sel], my[sel], p[0], p[1], p[2],
                                  p[4], p[3]) - img[sel]) ** 2
                        / (img[sel] + 1e-8)
                    )
                x0 = [mx[sel][xy], my[sel][xy],
                      sel_max - sel_mean, sel_mean,
                      self.sigma]
                bounds = (
                    (pos[idx,1]-self.size[1], pos[idx,1]+self.size[1]),
                    (pos[idx,0]-self.size[0], pos[idx,0]+self.size[0]),
                    (0.0, 2*sel_max),
                    (0.0, sel_max),
                    (0.1*self.sigma, 2*self.sigma)
                    )
            elif self.fit_aim == 'asym':
                def fit_function(p):
                    return np.sum((
                        blob_bivar_func(mx[sel], my[sel], p[0], p[1], p[2],
                                  p[4], p[5], p[6], p[3]) - img[sel]) ** 2
                        / (img[sel] + 1e-8)
                    )
                x0 = [mx[sel][xy], my[sel][xy],
                      sel_max - sel_mean, sel_mean,
                      self.sigma, self.sigma, 0.0]
                bounds = (
                    (pos[idx,1]-self.size[1], pos[idx,1]+self.size[1]),
                    (pos[idx,0]-self.size[0], pos[idx,0]+self.size[0]),
                    (0.0, 2*sel_max),
                    (0.0, sel_max),
                    (0.1*self.sigma, 2*self.sigma),
                    (0.1*self.sigma, 2*self.sigma),
                    (-1.0, 1.0)
                    )

            # Optimization details:
            # 1. max - mean is a good starting value for M
            # 2. mean is a good starting value for b
            # 3. The true x and y coordinates in the
            #    cropped image are good starting values for x0 and y0.
            # 4. SLSQP is by far the fastest minimzation method for this task
            #    plus it's more accurate, plus it can even handle bounds.
            r = minimize(
                fit_function,
                x0=x0,
                method='SLSQP',
                options={'maxiter': 500},
                tol=self.tol,
                bounds=bounds
            )

            if (self.fit_aim == 'detect') | (self.fit_aim == 'calib'):
                if self.remove_detected_stars:
                    img[sel] = img[sel] - blob_func(mx[sel], my[sel], r.x[0], r.x[1],
                                          r.x[2], self.sigma, r.x[3]) + r.x[3]
            elif self.fit_aim == 'psf':
                if self.remove_detected_stars:
                    img[sel] = img[sel] - blob_func(mx[sel], my[sel], r.x[0], r.x[1],
                                          r.x[2], r.x[4], r.x[3]) + r.x[3]
                    if 'sig_fit' not in results.keys():
                        results['sig_fit'] = np.zeros(n_stars)
                    results['sig_fit'][idx] = r.x[4]
            elif self.fit_aim == 'asym':
                if self.remove_detected_stars:
                    img[sel] = img[sel] - blob_bivar_func(mx[sel], my[sel], 
                        r.x[0], r.x[1], r.x[2], r.x[4], r.x[5], r.x[6], r.x[3]
                        ) + r.x[3]
                    if 'sigx_fit' not in results.keys():
                        results['sigx_fit'] = np.zeros(n_stars)
                        results['sigy_fit'] = np.zeros(n_stars)
                        results['cor_fit'] = np.zeros(n_stars)
                    results['sigx_fit'][idx] = r.x[4]
                    results['sigy_fit'][idx] = r.x[5]
                    results['cor_fit'][idx] = r.x[6]

            
            visibility = np.abs(r.x[0]) / np.exp(-image.stars.mag.iloc[idx])
            results['id'][idx] = image.stars.id.iloc[idx]
            results['M_fit'][idx] = np.abs(r.x[2])
            results['visibility'][idx] = visibility * self.calibration
            results['b_fit'][idx] = np.abs(r.x[3])
            results['y_fit'][idx] = r.x[0]
            results['x_fit'][idx] = r.x[1]
            results['llh_fit'][idx] = r.fun
            results['success_fit'][idx] = r.success
            results['iter_fit'][idx] = r.nit
            if sel_max == 1.0:
                results['saturated'][idx] = True
            else:
                results['saturated'][idx] = False
        results = pd.DataFrame(results, index=results['id'].astype(int))
        results.id = results.id.astype(int)
        return results


class StarDetectionLLH2(StarDetectionBase):
    """Likelihood approach to star detection. Fits every star individually
    using a more complex model than 'StarDetectionLLH'.
    
    Attributes
    ----------
    name : str
        Name of the method
    presmoothing : float
        Sigma parameter of the Gaussian filter applied beforehand.
    sigma : float
        Sigma parameter of the model used during fit.
    size : tuple(int)
        Size of the cropped out parts of the image for fitting.
    verbose : bool
        Verbosity flag.
    remove_detected_stars : bool
        Whether or not to successively remove fitted stars as they are
        detected.
    """
    name = 'llh2_star_detection'

    def __init__(self, camera, sigma=1.6, fit_size=4, presmoothing=0.0,
                 remove_detected_stars=True, verbose=True, tol=1e-15):
        super(StarDetectionLLH2, self).__init__(camera)
        self.sigma = sigma
        self.size = (fit_size, fit_size)
        self.presmoothing = presmoothing
        self.verbose = verbose
        self.remove_detected_stars = remove_detected_stars
        self.tol = tol
    
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
                * `id`: Id of each star.
                * `M_fit`: Fitted linear magnitude.
                * `b_fit`: Fitted background level.
                * `x_fit`: Fitted x-position.
                * `y_fit`: Fitted y-position.
                * `visibility`: Calculated visibility factor.
        """
        super(StarDetectionLLH2, self).detect(image)

        # Apply presmoothing.
        img = gaussian(image.image, self.presmoothing)
        
        # Generate meshgrid for pixel coordinate grid.
        tx = np.arange(img.shape[0])
        ty = np.arange(img.shape[1])
        mx, my = np.meshgrid(ty, tx)
        n_stars = len(image.stars)
        pos = np.column_stack((image.stars.x.values,
                               image.stars.y.values))

        # Prepare keys to write during the detection.
        keys = [
            'id',
            'M_fit',
            'b_fit',
            'x_fit',
            'y_fit',
            'sigx_fit',
            'sigy_fit',
            'cor_fit',
            'success_fit',
            'llh_fit',
            'visibility',
        ]
        results = {key: np.zeros(n_stars) for key in keys}

        # Sort by magnitude to process stars ordered by magnitude.
        mag_sort_idx = np.argsort(image.stars.mag.values)
        if self.verbose:
            iterator = tqdm(mag_sort_idx, total=len(mag_sort_idx))
        else:
            iterator = mag_sort_idx
        for idx in iterator:
            sel = crop((pos[idx,1], pos[idx,0]), self.size, img.shape)
            def fit_function(p):
                return np.sum((
                    blob_bivar_func(mx[sel], my[sel], 
                        p[0], p[1], p[2], p[3], p[4], p[5], p[6]
                        ) - img[sel]) ** 2
                    / (img[sel] + 1e-8)
                )
            # Optimization details:
            # 1. max - mean is a good starting value for M
            # 2. mean is a good starting value for b
            # 3. The true x and y coordinates in the
            #    cropped image are good starting values for x0 and y0.
            # 4. SLSQP is by far the fastest minimzation method for this task
            #    plus it's more accurate, plus it can even handle bounds.
            sel_max = np.max(img[sel])
            sel_min = np.min(img[sel])
            sel_mean = np.mean(img[sel])
            r = minimize(
                fit_function,
                x0=[pos[idx,1], pos[idx,0], sel_max - sel_mean, 
                    self.sigma, self.sigma, 0, sel_mean,
                    ],
                method='SLSQP',
                tol=self.tol,
                bounds=(
                    (pos[idx,1]-self.size[1], pos[idx,1]+self.size[1]),
                    (pos[idx,0]-self.size[0], pos[idx,0]+self.size[0]),
                    (0.0, sel_max),
                    (0.1*self.sigma, 2*self.sigma),
                    (0.1*self.sigma, 2*self.sigma),
                    (-1, 1),
                    (0.0, sel_max),  
                )
            )
            if self.remove_detected_stars:
                img[sel] -= blob_bivar_func(mx[sel], my[sel], 
                            r.x[0], r.x[1], r.x[2], r.x[3],
                            r.x[4], r.x[5], r.x[6],)
            visibility = np.abs(r.x[2]) / np.exp(-image.stars.mag.iloc[idx])
            results['id'][idx] = image.stars.id.iloc[idx]
            results['M_fit'][idx] = np.abs(r.x[2])
            results['visibility'][idx] = visibility * self.calibration
            results['b_fit'][idx] = np.abs(r.x[6])
            results['y_fit'][idx] = r.x[0]
            results['x_fit'][idx] = r.x[1]
            results['sigy_fit'][idx] = r.x[3]
            results['sigx_fit'][idx] = r.x[4]
            results['cor_fit'][idx] = r.x[5]
            results['success_fit'][idx] = r.success
            results['llh_fit'][idx] = r.fun
        results = pd.DataFrame(results, index=results['id'].astype(int))
        results.id = results.id.astype(int)
        return results


class StarDetectionFilter(StarDetectionBase):
    """Likelihood approach to star detection. Fits every star individually
    using a simple model.
    
    Attributes
    ----------
    name : str
        Name of the method
    sigma : float
        Sigma parameter of the model used for the LoG-filter
    size : tuple(int)
        Size of the cropped out parts of the image.
    verbose : bool
        Verbosity flag.
    quantile : float
        Quantile to use for calculating the M-value.
    """
    name = 'filter_star_detection'

    def __init__(self, camera, sigma=1.0, fit_size=4, quantile=100.0,
                 verbose=True):
        super(StarDetectionFilter, self).__init__(camera)
        self.sigma = sigma
        self.size = (fit_size, fit_size)
        self.quantile = quantile
        self.verbose = verbose
    
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
                * `id`: Id of each star.
                * `M_fit`: Fitted linear magnitude.
                * `visibility`: Calculated visibility factor.
        """
        super(StarDetectionFilter, self).detect(image)
        img = laplacian_gaussian_filter(image.image, self.sigma)

        tx = np.arange(img.shape[1])
        ty = np.arange(img.shape[0])
        mx, my = np.meshgrid(tx, ty)

        n_stars = len(image.stars)
        pos = np.column_stack((image.stars.x.values,
                               image.stars.y.values))

        # Prepare keys to write during the detection.
        keys = [
            'id',
            'M_fit',
            'b_fit',
            'x_fit',
            'y_fit',
            'visibility'
        ]
        results = {key: np.zeros(n_stars) for key in keys}
        iterator = progressbar(range(n_stars)) if self.verbose else range(n_stars)
        for idx in iterator:
            sel = crop((pos[idx,1], pos[idx,0]), self.size, img.shape)
            M = np.percentile(img[sel], self.quantile)

            mean, _ = mean_cov(mx[sel], my[sel],
                               (img[sel] - np.min(img[sel])) ** 2)

            results['id'][idx] = image.stars.id.iloc[idx]
            results['M_fit'][idx] = M
            results['b_fit'][idx] = np.mean(image.image[sel])
            results['x_fit'][idx] = mean[1]
            results['y_fit'][idx] = mean[0]
            visibility = M / np.exp(-image.stars.mag.iloc[idx])
            results['visibility'][idx] = visibility * self.calibration
        return pd.DataFrame(results, index=results['id'].astype(int))


class StarDetectionBlob(StarDetectionBase):
    name = 'blob_star_detection'

    def __init__(self, camera, sigma=1.6, threshold=0.001, radius=5):
        super(StarDetectionBlob, self).__init__(camera)
        self.sigma = sigma
        self.threshold = threshold
        self.radius = radius

    def detect(self, image):
        super(StarDetectionBlob, self).detect(image)
        blobs = blob_log(image.image, min_sigma=self.sigma,
                         max_sigma=self.sigma, num_sigma=1,
                         threshold=self.threshold)
        mask = self.camera.check_mask(blobs[:,0], blobs[:,1])
        blobs = blobs[mask,:]
        kdtree = cKDTree(blobs[:,:-1])
        star_pos = np.column_stack((image.stars.x, image.stars.y))
        matches = kdtree.query_ball_point(star_pos, self.radius)
        visibility = np.array([len(m) > 0 for m in matches])
        star_pos[visibility, :] = np.array([blobs[m[0],:-1]
                                            for m in matches if len(m) > 0])
        return pd.DataFrame({'id': image.stars.id,
                             'visibility': visibility.astype(float),
                            'x_fit': star_pos[:,0],
                            'y_fit': star_pos[:,1]},
                            index=image.stars.id.values)
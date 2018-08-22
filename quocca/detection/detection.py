"""quocca: All Sky Camera Analysis Tools

Detection.

2018"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.ndimage import convolve
from skimage.filters import gaussian
from ruamel import yaml
from pkg_resources import resource_filename
import warnings
from astropy.time import Time


class CalibrationWarning(Warning):
    pass


def nearby_stars(x, y, mag, radius):
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


def laplacian_gaussian_filter(img, sigma, prec=1e-16):
    tx = np.arange(img.shape[0]) - img.shape[0] * 0.5
    ty = np.arange(img.shape[1]) - img.shape[1] * 0.5
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


class StarDetectionLLH:
    name = 'llh_star_detection'

    def __init__(self, camera, sigma=1.7, fit_size=8, fit_pos=True,
                 fit_sigma=False, presmoothing=1.5):
        self.sigma = sigma
        self.size = (fit_size, fit_size)
        self.fit_pos = fit_pos
        self.fit_sigma = fit_sigma
        self.presmoothing = presmoothing
        self.camera = camera

    def get_slice(self, pos, shape):
        pos = list(np.round(pos).astype(int))
        a_min = np.clip(pos[1] - self.size[1], 0, shape[1] - 1)
        a_max = np.clip(pos[1] + self.size[1] + 1, 0, shape[1] - 1)
        b_min = np.clip(pos[0] - self.size[0], 0, shape[0] - 1)
        b_max = np.clip(pos[0] + self.size[0] + 1, 0, shape[0] - 1)
        return (slice(a_min, a_max, None),
                slice(b_min, b_max, None))
    
    def blob_func(self, x, y, x0, y0, mag, sx, sy, rho, bkg):
        x_ = x - x0
        y_ = y - y0
        upper = sy ** 2 * x_ ** 2 - 2.0 * rho * sx * sy * x_ * y_ + sx ** 2\
                * y_ ** 2
        det2 = (rho **2 - 1) * sx ** 2 * sy ** 2
        arg = upper / det2
        return np.clip(np.abs(mag) * np.exp(arg) + np.abs(bkg), 0.0, 1.0)
    
    def detect(self, image, max_mag=7.0, min_dist=16.0, verbose=True, remove_detected_stars=False):
        self.calibration = get_calibration(self.camera.name, self.name, image.time)
        img = gaussian(image.image, self.presmoothing)
        tx = np.arange(img.shape[0])
        ty = np.arange(img.shape[1])
        mx, my = np.meshgrid(tx, ty)
        mask_nearby = nearby_stars(image.star_pos[:,0],
                                   image.star_pos[:,1],
                                   image.star_mag, min_dist)
        if max_mag is None:
            mask_mag = np.ones(len(mask_nearby), dtype=bool)
        else:
            mask_mag = image.star_mag < max_mag
        mask = mask_mag & mask_nearby 
        n_stars = len(image.star_pos[mask])
        pos = image.star_pos[mask]
        results = {
            key: np.zeros(n_stars)
            for key in ['M_fit', 'b_fit', 'x_fit', 'y_fit', 'v_mag', 'x', 'y',
                        'visibility']
        }
        mag_sort_idx = np.argsort(image.star_mag[mask])
        if verbose:
            iterator = tqdm(mag_sort_idx, total=len(mag_sort_idx))
        else:
            iterator = mag_sort_idx
        for idx in iterator:
            sel = self.get_slice((pos[idx,1],
                                  pos[idx,0]),
                                 img.shape)
            def fit_function(p):
                return np.sum((self.blob_func(mx[sel], my[sel], p[2], p[3],
                                              p[0], self.sigma, self.sigma,
                                              0.0, p[1]) - img[sel]) ** 2)
            mean, cov = mean_cov(mx[sel], my[sel], (img[sel] - np.min(img[sel])) ** 3)
            r = minimize(fit_function,
                         #x0=[0.0, np.mean(img[sel]),
                         x0=[np.max(img[sel]) - np.mean(img[sel]), 
                             #np.min(img[sel]),
                             np.mean(img[sel]),
                             pos[idx,1], pos[idx,0]],
                             #mean[0], mean[1]],
                         method='Powell')
            if remove_detected_stars:
                img[sel] -= self.blob_func(mx[sel], my[sel], r.x[2], r.x[3],
                                              r.x[0], self.sigma, self.sigma,
                                              0.0, 0.0)
            results['M_fit'][idx] = np.abs(r.x[0])
            visibility = np.abs(r.x[0]) / np.exp(-image.star_mag[mask][idx])
            #results['visibility'][idx] = np.clip(visibility * self.calibration,
            #                                     0.0, 1.0)
            results['visibility'][idx] = visibility * self.calibration
            results['b_fit'][idx] = np.abs(r.x[1])
            results['y_fit'][idx] = r.x[2]
            results['x_fit'][idx] = r.x[3]
            results['v_mag'][idx] = image.star_mag[mask][idx]
            results['x'][idx] = image.star_pos[mask, 0][idx]
            results['y'][idx] = image.star_pos[mask, 1][idx]
        return pd.DataFrame(results)


class StarDetectionFilter:
    name = 'filter_star_detection'
    def __init__(self, camera, sigma, fit_size, quantile=100.0):
        self.sigma = sigma
        self.size = (fit_size, fit_size)
        self.quantile = quantile
        self.camera = camera
        with open(resource_filename('quocca', 'resources/cameras.yaml')) as file:
            config = yaml.safe_load(file)
            try:
                self.calibration = config[self.camera.name][self.name]
            except KeyError:
                warnings.warn('No calibration setting found.')
                self.calibration = None
    
    def get_slice(self, pos, shape):
        pos = list(np.round(pos).astype(int))
        a_min = np.clip(pos[1] - self.size[1], 0, shape[1] - 1)
        a_max = np.clip(pos[1] + self.size[1] + 1, 0, shape[1] - 1)
        b_min = np.clip(pos[0] - self.size[0], 0, shape[0] - 1)
        b_max = np.clip(pos[0] + self.size[0] + 1, 0, shape[0] - 1)
        return (slice(a_min, a_max, None),
                slice(b_min, b_max, None))

    def detect(self, image, max_mag=5.5, min_dist=6.0, verbose=True):
        if self.calibration is None:
            warnings.warn('Method {} for camera {} is not calibrated yet.'
                          .format(self.name, self.camera.name))
            self.calibration = 1.0
        img = laplacian_gaussian_filter(image.image, self.sigma)
        tx = np.arange(img.shape[0])
        ty = np.arange(img.shape[1])
        mx, my = np.meshgrid(tx, ty)
        mask_nearby = nearby_stars(image.star_pos[:,0],
                                   image.star_pos[:,1],
                                   image.star_mag, min_dist)
        mask_mag = image.star_mag < max_mag
        mask = mask_mag & mask_nearby 
        n_stars = len(image.star_pos[mask])
        pos = image.star_pos[mask]
        results = {
            key: np.zeros(n_stars)
            for key in ['M_fit', 'v_mag', 'x', 'y', 'visibility']
        }
        if verbose:
            iterator = tqdm(range(n_stars))
        else:
            iterator = range(n_stars)
        for idx in iterator:
            sel = self.get_slice((pos[idx,1],
                                  pos[idx,0]),
                                 img.shape)
            
            M = np.percentile(img[sel], self.quantile)

            results['M_fit'][idx] = M
            visibility = M / np.exp(-image.star_mag[mask][idx])
            #results['visibility'][idx] = np.clip(visibility * self.calibration,
            #                                     0.0, 1.0)
            results['visibility'][idx] = visibility * self.calibration
            results['v_mag'][idx] = image.star_mag[mask][idx]
            results['x'][idx] = image.star_pos[mask, 0][idx]
            results['y'][idx] = image.star_pos[mask, 1][idx]
        return pd.DataFrame(results)

"""quocca: All Sky Camera Analysis Tools

Cloudiness.

2018"""

from sklearn.neighbors.regression import KNeighborsRegressor, check_array, _get_weights
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import ruamel.yaml as yaml
from skimage.filters import gaussian


class CloudinessCalculator:
    """Cloudiness base class
    """
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, x, y, fit_results):
        raise NotImplementedError('Fit procedure not implemented.')

    def predict(self):
        raise NotImplementedError('Predict procedure not implemented.')


class MedianKNNRegressor(KNeighborsRegressor):
    def predict(self, X):
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = np.median(_y[neigh_ind], axis=1)
        else:
            raise NotImplementedError("weighted median")

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred


class KNNMedian(CloudinessCalculator):
    def __init__(self, k):
        self.k = k
        self.knn = MedianKNNRegressor(n_neighbors=k)

    def fit(self, x, y, fit_results):
        self.knn.fit(np.column_stack((x, y)), fit_results)

    def predict(self, x, y):
        return self.knn.predict(np.column_stack((x, y)))


class RunningAvg(CloudinessCalculator):
    def __init__(self, radius, method='mean', **kwargs):
        self.radius = radius
        if method == 'mean':
            self._fun = np.mean
        if method == 'median':
            self._fun = np.median
        if method == 'percentile':
            try:
                self._fun = lambda x: np.percentile(x, p)
            except NameError:
                raise NameError('Keyword p has to be passed when method percentile is used.')

    def fit(self, x, y, fit_results):
        self.kdtree = cKDTree(np.column_stack((x, y)))
        self.fit_results = fit_results

    def predict(self, x, y):
        idx = self.kdtree.query_ball_point(np.column_stack((x, y)),
                                           self.radius)
        return np.array([self._fun(self.fit_results[i])
                         for i in idx])


def cloud_map(x, y, fit_results, extent, size, cloudiness_calc, smoothing=None,
              **kwargs):
    tx = np.linspace(0, extent[0], size[0])
    ty = np.linspace(0, extent[1], size[1])
    mx, my = np.meshgrid(tx, ty)
    reg = cloudiness_calc(**kwargs)
    print("Fitting model ...")
    reg.fit(x, y, fit_results)
    print("Applying model ...")
    pred = np.zeros(mx.shape)
    for i in tqdm(range(len(mx))):
        pred[i, :] = reg.predict(mx[i,:], my[i,:])
    if smoothing is not None:
        pred = gaussian(pred, smoothing)
    return pred
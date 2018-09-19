# quocca - QUick Observation of Cloud Coverage using All sky images
![Teaser](http://www.hoinka.net/quocca_teaser.png)

This module implements tools to analyse night-time all sky images with the goal to estimate the transmissivity of the atmosphere.

## Examples

### Process a Single Image

To process an image, objects of the classes `Camera`, `Catalog` and `Image` are put together and processed in a method from `quocca.detection`. In this case the Image `test.mat` recorded by the camera `cta` is analysed using the `hipparcos` catalog.

```python
from quocca.camera import Camera


cam = Camera('cta')
img = cam.read('path/to/test.mat')
img.add_catalog('hipparcos', max_mag=5.5, min_dist=12.0)
res = img.detect('llh', sigma=1.7, fit_size=8)
res = res.merge(img.stars, on='id', left_index=True)
``` 

`res` is then a pandas DataFrame containing the results of the detection in the attribute `visibility`.
Additional information from the star catalog can be added in addition.


Based on the results, a cloud map can be produced (e.g. with the method `GaussianRunningAvg`.

```python
cmap = cloud_map(res.x_fit, res.y_fit, res.visibility,
                 extent=(cam.resolution['x'], cam.resolution['y']),
                 size=(200, 200),
                 cloudiness_calc=GaussianRunningAvg,
                 radius=25,
                 smoothing=3)

fig, ax = plt.subplots(figsize=(8,8))
ax = show_img(img, ax=ax)
ax = show_clouds(img, cmap, opaque=True, ax=ax)
```

### Calibrating Camera Parameters

Calibrating a camera is necessary for multiple reasons, namely
* The intrinsic parameters of the camera are usually only known inaccurately. Calibrating the camera using a clear sky image can help out.
* The methods used for star detection have no concept of the absolute brightness of the recorded image.

Methods for calibration are found in `quocca.utilities`.

```python
from quocca.utilities import fit_camera_params

cam = Camera('cta')
fit_camera_params('2015_11_04-00_01_31.mat', cam, 
          kwargs_catalog={'catalog':'hipparcos', 'max_mag': 6, 
                            'min_dist': 12.0, 'max_var': 2, 
                            'min_alt': 30},
                  update=True)
```
This fits camera parameters (e.g. position of the zenith in the image, or an azimut offset) to a clear sky image and updates the configs automatically.

```python
from quocca.utilities import calibrate_method


cam = Camera('cta')
calibrate_method('2015_11_04-00_01_31.mat', 
                 cam, method='llh', 
                 kwargs_catalog={'catalog':'hipparcos', 'max_mag': 6, 
                           'min_dist': 12.0, 'max_var': 2, 
                           'min_alt': 30}, 
                 kwargs_method={'sigma':1.6, 'fit_size': 4},
                 update=True)
```
This calibrates the estimated visibility of the method `StarDetectionLLH`. The configs are updated automatically.
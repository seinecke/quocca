# quocca - QUality Observation of Cloud Coverage using All sky images
or
# quocca - QUalitative Observables from Cloud Camera Analysis

This module implements tools to analyse night-time all sky images with the goal to estimate the transmissivity of the atmosphere.

## Examples

### Process a Single Image

To process an image, objects of the classes `Camera`, `Catalog` and `Image` are put together and processed in a method from `quocca.detection`. In this case the Image `test.mat` recorded by the camera `cta` is analysed using the `hipparcos` catalog.

```python
from quocca.camera import Camera


cam = Camera('cta')
img = cam.read('path/to/test.mat')
img.add_catalog('hipparcos', max_mag=5.5, min_dist=12.0)
results = img.detect('llh', sigma=1.7, fit_size=8)
``` 

`results` is then a pandas DataFrame containing the results of the detection in the attribute `visibility`.

### Calibrating Camera Parameters

Calibrating a camera is necessary for multiple reasons, namely
* The intrinsic parameters of the camera are usually only known inaccurately. Calibrating the camera using a clear sky image can help out.
* The methods used for star detection have no concept of the absolute brightness of the recorded image.

Methods for calibration are found in `quocca.utilities`.

```python
from quocca.utilities import fit_camera_params

cam = Camera('cta')
fit_camera_params('2015_11_04-00_01_31.mat', cam, update=True)
```
This fits camera parameters (e.g. position of the zenith in the image, or an azimut offset) to a clear sky image and updates the configs automatically.

```python
from quocca.utilities import calibrate_method


cam = Camera('cta')
det = StarDetectionLLH(cam, sigma=1.7, fit_size=8)
calibrate_method('2015_11_04-00_01_31.mat', cam, det, update=True)
```
This calibrates the estimated visibility of the method `StarDetectionLLH`. The configs are updated automatically.

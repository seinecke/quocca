# quocca - QUality Observation of Cloud Coverage using All sky images
or
# quocca - QUalitative Observables from Cloud Camera Analysis

This module implements tools to analyse night-time all sky images with the goal to estimate the transmissivity of the atmosphere.

## Examples

### Process a Single Image

To process an image, objects of the classes `Camera`, `Catalog` and `Image` are put together and processed in a method from `quocca.detection`. In this case the Image `test.mat` recorded by the camera `cta` is analysed using the `hipparcos` catalog.

```python
from quocca.camera import Camera
from quocca.catalog import Catalog
from quocca.image import Image
from quocca.detection import FilterStarDetection


cat = Catalog('hipparcos')
cam = Camera('cta')
img = Image('test.mat', cam, cat)
det = FilterStarDetection(1.7, 8)
result = det.detect(img, max_mag=20.0, min_dist=16)
``` 

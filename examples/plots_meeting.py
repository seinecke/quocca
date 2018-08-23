import numpy as np

from quocca.catalog import Catalog
from quocca.camera import Camera
from quocca.image import Image
from quocca.detection import StarDetectionLLH
from quocca.plotting import show_img, add_circle, show_clouds
from quocca.plotting import compare_estimated_to_true_magnitude
from quocca.cloudiness import GaussianRunningAvg, RunningAvg, cloud_map

from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

rcParams['font.family'] = 'Akkurat'

cat = Catalog('hipparcos')
cam = Camera('cta')

images = ['/Users/thoinka/2015_11_04-00_01_31.mat',
          '/Users/thoinka/2015_11_09-06_03_26.mat']
appends = ['clear', 'cloudy']

for img_path, append in zip(images, appends):
	img = Image(img_path, cam, cat)
	det = StarDetectionLLH(cam, 2.4, 5)
	result = det.detect(img, max_mag=5.0, min_dist=12,
		                remove_detected_stars=True)

	# Plot of sky cam image without any decorations.
	ax = img.show()
	plt.tight_layout()
	plt.savefig('bare_{}_asci.pdf'.format(append), transparent=True)

	# Plot of sky cam image with catalog stars.
	ax = img.show()
	ax = add_circle(result.y, result.x, result.v_mag,
	                size=20 * (5.0 - result.v_mag),
	                color='w', ax=ax)
	plt.tight_layout()
	plt.savefig('catalog_{}_asci.pdf'.format(append), transparent=True)

	# Plot of sky cam image with catalog stars and fits.
	ax = img.show()
	ax = add_circle(result.y, result.x, result.v_mag,
	                size=20 * (5.0 - result.v_mag),
	                color='w', ax=ax)
	ax = add_circle(result.y_fit, result.x_fit, result.v_mag,
	                size=20 * (5.0 + 0.6 * np.log(result.M_fit)),
	                color='#7ac143', ax=ax)
	plt.tight_layout()
	plt.savefig('fits_{}_asci.pdf'.format(append), transparent=True)

	# Vector field plot with fit shifts.
	ax = img.show()
	length = np.sqrt((result.y_fit - result.y) ** 2\
		   + (result.x_fit - result.x) ** 2)
	# Remove misfits
	sel = length < 1.5
	plt.quiver(result.y[sel], result.x[sel],
	           (result.y_fit - result.y)[sel],
	           (result.x_fit - result.x)[sel], color='#7ac143')
	plt.tight_layout()
	plt.savefig('quiver_{}_asci.pdf'.format(append), transparent=True)

	# Cloudmap
	cloudmap = cloud_map(result.x_fit, result.y_fit,
		                 np.clip(result.visibility, 0.0, 1.0),
                         (1699, 1699), (200, 200), GaussianRunningAvg,
                         radius=30, weights=np.exp(-1.0 * result.v_mag))
	ax = show_img(img, upper=99.9)
	ax = show_clouds(img, cloudmap, opaque=True, ax=ax)
	plt.tight_layout()
	plt.savefig('cloud_{}_asci.pdf'.format(append), transparent=True)

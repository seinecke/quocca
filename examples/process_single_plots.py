from quocca.camera import Camera
from quocca.cloudiness import cloud_map, GaussianRunningAvg
from quocca.plotting import show_img, show_clouds
from quocca.detection import get_calibration

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import click


@click.command()
@click.option('-i', '--input', help='Input File', type=str)
@click.option('-c', '--camera', help='Camera Identifier', type=str)


def process(input, camera):
	cam = Camera(camera)
	img = cam.read(input)
	img.add_catalog(catalog='hipparcos', 
		            max_mag=10, 
		            min_dist=12.0, 
                    max_var=1.5, 
                    min_alt=20)
	res = img.detect('llh', 
		             sigma=1.6, 
		             fit_size=4, 
		             tol=1e-15,
                     presmoothing=0.5, 
                     remove_detected_stars=True,)
	res = res.merge(img.stars, on='id', left_index=True)

	# Store results
	res.to_csv('{}_{}.csv'.format(camera, img.time))

	# Calculate cloudmap
	mask = res.mag < 7
	cmap = cloud_map(res[mask].x_fit,
                     res[mask].y_fit,
                 	 res[mask].visibility,
                 	 extent=(cam.resolution['x'], cam.resolution['y']),
                 	 size=(200, 200),
                 	 cloudiness_calc=GaussianRunningAvg,
                 	 smoothing=3,
                 	 radius=25)

	# Make and store plots
	fig, ax = plt.subplots(1,3,figsize=(16, 5),
                       gridspec_kw={'width_ratios':[4,4,5]})

	ax[0] = show_img(img, ax=ax[0])

	ax[1] = show_img(img, ax=ax[1])
	ax[1] = show_clouds(img, cmap, opaque=True, ax=ax[1])

	_, _, _, pos = ax[2].hist2d(res.mag, np.log(res.M_fit),
	                         norm=LogNorm(), bins=(90, 60), 
	                         range=((0,10),(-15,1)),
	                         cmap='Greys_r')
	calib = get_calibration(cam.name, 'llh_star_detection', img.time)
	ax[2].plot([-5,25], -np.log(calib) - np.array([-5,25]), 
	            '-', color='#7ac143', lw=3, label='Clear Sky')
	fig.colorbar(pos, ax=ax[2])
	ax[2].set_xlabel('True Magnitude')
	ax[2].set_ylabel('Estimated Magnitude')
	ax[2].legend(frameon=False, loc='lower left')

	plt.tight_layout()
	plt.savefig('{}_{}.pdf'.format(camera, img.time))


if __name__ == '__main__':
	process()
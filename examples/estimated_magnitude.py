from quocca.camera import Camera
from quocca.detection import get_calibration

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
import pandas as pd

import click


@click.command()
@click.option('-i', '--input', help='Input File', 
              type=str)
@click.option('-c', '--camera', help='Camera', 
              type=str, default='cta')
@click.option('-d', '--detect', help='Detection method to be used',
              type=str, default='llh')
@click.option('-m', '--mag', help='Maximum magnitude to be considered',
              type=float, default=10)
@click.option('-a', '--alt', help='Minimum altitude angle in deg',
			  type=float, default=20)
@click.option('-o', '--output', help='Output path', 
              type=str, default=None)


def process(input, camera, detect, mag, alt, output):
	cam = Camera(camera)
	img = cam.read(input)
	img.add_catalog(max_mag=mag, min_dist=12, min_alt=alt, max_var=1.5)

	if detect == 'llh':
		res = img.detect('llh')
	elif detect == 'filter':
		res = img.detect('filter')
	res = res.merge(img.stars, on='id', left_index=True)

	fig, ax = plt.subplots(figsize=(8, 4))

	lowx = 0.9*res.mag.min()
	upx = 1.01*res.mag.max()
	binsx = (upx-lowx)/0.1

	_, _, _, pos = ax.hist2d(res.mag, np.log(res.M_fit),
	                         norm=LogNorm(), bins=(binsx, 60), 
	                         range=((lowx,upx),(-15,3)),
	                         cmap='YlGn_r')

	calib = get_calibration(cam.name, detect+'_star_detection', img.time)

	ax.plot([-5,25], -np.log(calib) - np.array([-5,25]), 
	            '-', color='red', lw=2, label='Clear Sky')

	fig.colorbar(pos, ax=ax)
	ax.set_xlabel('True Magnitude')
	ax.set_ylabel('Estimated Magnitude')
	plt.legend(loc='lower left', frameon=False)

	plt.tight_layout()

	if output == None:
		plt.show()
	else:
		plt.savefig(output+'EstimatedMagnitude_{}.png'.format(detect))

	plt.close()


if __name__ == '__main__':
    process()
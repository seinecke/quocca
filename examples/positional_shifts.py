from quocca.camera import Camera

from matplotlib import pyplot as plt
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
              type=float, default=5.0)
@click.option('-o', '--output', help='Output path', 
              type=str, default=None)


def process(input, camera, detect, mag, output):
	cam = Camera(camera)
	img = cam.read(input)
	img.add_catalog(max_mag=mag, min_dist=12, min_alt=0, max_var=1.5)

	if detect == 'llh':
		res = img.detect('llh')
	elif detect == 'filter':
		res = img.detect('filter')
	elif detect == 'blob':
		res = img.detect('blob')
		res = res[res.visibility==1.0]

	res = res.merge(img.stars, on='id', left_index=True)

	fig, ax = plt.subplots(2,1,figsize=(8, 10), 
	                       gridspec_kw={'height_ratios':[6,1]})

	ax[0] = img.show(ax=ax[0])
	ax[0].quiver(res.y, res.x, 
	           res.y_fit, res.x_fit,
	         color='#7ac143')

	res['radius'] = np.sqrt( (res.x - res.x_fit)**2 
		                + (res.y - res.y_fit)**2 )

	ax[1].hist(res.radius, 
	           bins=int((res.radius.max()-res.radius.min())/0.1), 
	           range=(res.radius.min(), res.radius.max()),
	           alpha=0.7, color='#7ac143')

	ax[1].set_xlabel('Distance / pixels')
	ax[1].set_ylabel('Number of Stars')

	plt.tight_layout()
	plt.subplots_adjust(hspace=-0.1)

	if output == None:
		plt.show()
	else:
		plt.savefig(output+'PositionalShifts_{}.png'.format(detect))

	plt.close()


if __name__ == '__main__':
    process()
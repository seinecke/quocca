from quocca.camera import Camera
import click


@click.command()
@click.option('-i', '--input', help='Input File', type=str)
@click.option('-c', '--camera', help='Camera Identifier', type=str)
@click.option('-m', '--magnitude', help='Maximum Magnitude', type=float)
def process(input, camera, magnitude):
	cam = Camera(camera)
	img = cam.read(input)
	img.add_catalog(max_mag=magnitude, min_dist=12, min_alt=30)
	result = img.detect('llh', sigma=1.6, fit_size=3,
	                    remove_detected_stars=True, presmoothing=False)
	result = result.merge(img.stars, on='id', left_index=True)
	result.to_csv('{}_{}.csv'.format(camera, img.time))


if __name__ == '__main__':
	process()
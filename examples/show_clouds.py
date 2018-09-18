from quocca.camera import Camera
from quocca.plotting import show_circle, show_img, show_clouds
from quocca.cloudiness import GaussianRunningAvg, cloud_map

from matplotlib import pyplot as plt

import click


@click.command()
@click.option('-i', '--input', help='Input File', type=str)
@click.option('-r', '--radius', help='Smoothing Radius', type=float,
              default=25.0)
@click.option('-c', '--camera', help='Camera', type=str, default='cta')
def process(input, radius, camera):
    cam = Camera(camera)
    img = cam.read(input)
    img.add_catalog(max_mag=7.0, min_dist=7, min_alt=0, max_var=1.5)
    result_llh = img.detect('llh', sigma=1.6, fit_size=5, tol=1e-15,
                            remove_detected_stars=True, presmoothing=0.5)
    cmap = cloud_map(result_llh.x_fit,
                     result_llh.y_fit,
                     result_llh.visibility,
                     extent=(cam.resolution['x'], cam.resolution['y']),
                     size=(200, 200),
                     cloudiness_calc=GaussianRunningAvg,
                     radius=radius)
    fig, ax = plt.subplots(figsize=(8,8))
    ax = show_img(img, ax=ax)
    ax = show_clouds(img, cmap, opaque=True, ax=ax)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    process()
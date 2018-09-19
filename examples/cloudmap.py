from quocca.camera import Camera
from quocca.plotting import show_circle, show_img, show_clouds
from quocca.cloudiness import GaussianRunningAvg, RunningAvg
from quocca.cloudiness import KNNMedian, cloud_map

from matplotlib import pyplot as plt

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
@click.option('-cm', '--cloudmap', help='Cloud map method to be used',
              type=str, default='GaussianRunningAvg')
@click.option('-r', '--radius', help='Smoothing radius', 
              type=float, default=25.0)
@click.option('-k', '--k', help='k of kNN cloud map generation', 
              type=int, default=5)
@click.option('-s', '--smooth', help='Smoothing of the cloud map', 
              type=int, default=0)
@click.option('-o', '--output', help='Output path', 
              type=str, default=None)


def process(input, camera, detect, mag, cloudmap, radius, k, smooth, output):
    cam = Camera(camera)
    img = cam.read(input)
    img.add_catalog(max_mag=mag, min_dist=7, min_alt=0, max_var=1.5)

    if detect == 'llh': 
        res = img.detect('llh', sigma=1.6, fit_size=4, tol=1e-15,
                            remove_detected_stars=True, presmoothing=0.5)
    elif detect == 'filter':
        res = img.detect('filter', sigma=1.0, fit_size=4)
    elif detect == 'blob':
        res = img.detect('blob')

    if cloudmap == 'GaussianRunningAvg':
        cmap = cloud_map(res.x_fit,
                        res.y_fit,
                        res.visibility,
                        extent=(cam.resolution['x'], cam.resolution['y']),
                        size=(200, 200),
                        cloudiness_calc=GaussianRunningAvg,
                        radius=radius,
                        smoothing=smooth)
    if cloudmap == 'RunningAvg':
        cmap = cloud_map(res.x_fit,
                        res.y_fit,
                        res.visibility,
                        extent=(cam.resolution['x'], cam.resolution['y']),
                        size=(200, 200),
                        cloudiness_calc=RunningAvg,
                        radius=radius,
                        method='median')
    elif cloudmap == 'KNNMedian':
        cmap = cloud_map(res.x_fit,
                        res.y_fit,
                        res.visibility,
                        extent=(cam.resolution['x'], cam.resolution['y']),
                        size=(200, 200),
                        cloudiness_calc=KNNMedian,
                        k=k,
                        smoothing=smooth)

    fig, ax = plt.subplots(figsize=(8,8))
    ax = show_img(img, ax=ax)
    ax = show_clouds(img, cmap, opaque=True, ax=ax)
    plt.tight_layout()

    if output == None:
        plt.show()
    else:
        plt.savefig(output+'Cloudmap_{}_{}.png'.format(detect, cloudmap))

    plt.close()


if __name__ == '__main__':
    process()
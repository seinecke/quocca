"""
Example of creating the following plots for a single image:
- Simple skymap with coordinate system
- Visibility values for each star on a skymap
- Comparison between fitted and true star positions on a skymap
- Comparison between serlected stars for fit and all catalog stars on a skymap
- 2D histogram of estimated and true magnitude
- 2D histogram of visibility and true magnitude

Usage: 
--------
>>> python plots_single_image.py 2015_11_04-03_43_02.mat
>>> python plots_single_image.py 2015_11_04-03_43_02.mat ../../
"""

import sys
import os

from matplotlib import pyplot as plt
import numpy as np

from quocca.camera import Camera
from quocca.image import Image
from quocca.catalog import Catalog
from quocca.detection import StarDetectionLLH
#from quocca.plotting import skymap_visibility, compare_used_stars_to_catalog
#from quocca.plotting import compare_fitted_to_true_positions
#from quocca.plotting import compare_estimated_to_true_magnitude
#from quocca.plotting import compare_visibility_to_magnitude

def compare_used_stars_to_catalog(img, res, max_mag=3.0):
    color_catalog = '#7ac143'
    color_used_stars = 'royalblue'

    ax = img.show()
    ax = add_circle(img.stars.y, img.stars.x, img.stars.mag, 
                    ax=ax, max_mag=max_mag, 
                    color=color_catalog, size=20)
    ax = add_circle(res.y, res.x, res.v_mag, ax=ax, max_mag=max_mag, 
                    color=color_used_stars, size=60)

    ax.text(0.99, 0.99, 'Catalog Stars', color=color_catalog,
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)
    ax.text(0.99, 0.99, '\nUsed Stars', color=color_used_stars,
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)
    return ax


def compare_fitted_to_true_positions(img, res, max_mag=3.0):
    color_fitted = 'darkorange'
    color_true = 'royalblue'

    ax = img.show()
    ax = add_circle(res.y, res.x, res.v_mag, ax=ax, max_mag=max_mag, 
                    color=color_true, size=60)
    ax = add_circle(res.y_fit, res.x_fit, res.v_mag, ax=ax, max_mag=max_mag, 
                    color=color_fitted, size=20)


    ax.text(0.99, 0.99, 'Fitted', color=color_fitted,
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)
    ax.text(0.99, 0.99, '\nTrue', color=color_true,
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)
    return ax


def compare_estimated_to_true_magnitude(res, det, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    lowx = res.v_mag.min()
    upx = res.v_mag.max()
    binsx = (upx-lowx)/0.1

    lowy = np.log(res.M_fit).min()
    upy = np.log(res.M_fit).max()
    binsy = (upy-lowy)/0.1

    _, _, _, pos = ax.hist2d((res.v_mag), np.log(res.M_fit), 
                              bins=(binsx,binsy), 
                              range=((lowx, upx),(lowy,upy)), 
                              norm=LogNorm(),
                              cmap='Greens_r'
                             )
    ax.plot([-5,25], -np.log(det.calibration) - np.array([-5,25]), 
            'r-', label='Clear Sky')

    fig.colorbar(pos, ax=ax)
    ax.legend(frameon=False, loc='lower left')
    ax.set_xlabel('True Magnitude')
    ax.set_ylabel('Estimated Magnitude')

    return ax


def compare_visibility_to_magnitude(res, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    lowx = res.v_mag.min()
    upx = res.v_mag.max()
    binsx = (upx-lowx)/0.1

    _, _, _, pos = ax.hist2d(res.v_mag, res.visibility,
                             norm=LogNorm(), bins=(binsx, 60), 
                             range=((lowx,upx),(-0.1,3)),
                             cmap='Greens_r')

    fig.colorbar(pos, ax=ax)
    ax.set_xlabel('True Magnitude')
    ax.set_ylabel('Visibility')

    return ax


def plot_visibility(vis, color='#7ac143', label='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(vis, bins=50, range=(-0.1,5),
         color=color, alpha=0.5, normed=True,
         label=label
        )
    ax.set_xlabel('Visibility')
    ax.set_ylabel('Normalised Number of Events')
    ax.legend(frameon=False)

    return ax


def skymap_visibility(img, res, max_mag=5.0):
    color1 = 'red'
    color2 = 'yellow'
    color3 = 'lime'

    ax = img.show()

    mask = res.visibility < 0.4
    ax = add_circle(res.y_fit[mask], res.x_fit[mask], res.v_mag[mask], 
                    ax=ax, max_mag=max_mag, 
                    color=color1, size=20)
    ax.text(0.99, 0.99, 'Visibility < 0.4', color=color1,
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)

    mask = (res.visibility > 0.4) & (res.visibility < 0.9)
    ax = add_circle(res.y_fit[mask], res.x_fit[mask], res.v_mag[mask], 
                    ax=ax, max_mag=max_mag, 
                    color=color2, size=20)
    ax.text(0.99, 0.97, '0.4 < Visibility < 0.9', color=color2,
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)

    mask = (res.visibility > 0.9)
    ax = add_circle(res.y_fit[mask], res.x_fit[mask], res.v_mag[mask], 
                    ax=ax, max_mag=max_mag, 
                    color=color3, size=20)
    ax.text(0.99, 0.95, '0.9 < Visibility', color=color3,
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)
    return ax
    

if __name__ == '__main__':


    filename = sys.argv[1]
    name = os.path.split(os.path.splitext(filename)[0])[1]

    if len(sys.argv) > 2:
        outdir = sys.argv[2]
    else:
        outdir = ''

    cam = Camera('cta')
    cat = Catalog('hipparcos')

    img = Image(filename, cam, cat)
    det = StarDetectionLLH(cam)
    res = det.detect(img, max_mag=20)

    ax = img.show()
    plt.tight_layout()
    plt.savefig(outdir + name + '_skymap.pdf')

    ax = skymap_visibility(img, res, max_mag=7)
    plt.tight_layout()
    plt.savefig(outdir + name + '_skymap-visibility.pdf')

    ax = compare_used_stars_to_catalog(img, res, max_mag=7)
    plt.tight_layout()
    plt.savefig(outdir + name + '_skymap-used_stars_and_catalog.pdf')

    ax = compare_fitted_to_true_positions(img, res, max_mag=7)
    plt.tight_layout()
    plt.savefig(outdir + name + '_skymap-fitted_and_true_positions.pdf')

    ax = compare_estimated_to_true_magnitude(res, det)
    plt.tight_layout()
    plt.savefig(outdir + name + '_hist-estimated_and_true_magnitude.pdf')

    ax = compare_visibility_to_magnitude(res)
    plt.tight_layout()
    plt.savefig(outdir + name + '_hist-true_magnitude_and_visibility.pdf')





"""
Example of creating the following plots for a single image:
- Visibility values for each star on a skymap

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
from quocca.plotting import skymap_visibility, compare_used_stars_to_catalog
from quocca.plotting import compare_fitted_to_true_positions
from quocca.plotting import compare_estimated_to_true_magnitude



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

    ax = skymap_visibility(img, res, max_mag=7)
    plt.tight_layout()
    plt.savefig(outdir + name + '_skymap-visibility.pdf')

    ax = compare_used_stars_to_catalog(img, res, max_mag=7)
    plt.tight_layout()
    plt.savefig(name+'_skymap-used_stars_to_catalog.pdf')

    ax = compare_fitted_to_true_positions(img, res, max_mag=7)
    plt.tight_layout()
    plt.savefig(outdir + name + '_skymap-fitted_to_true_positions.pdf')
    plt.show()

    ax = compare_estimated_to_true_magnitude(res, det)
    plt.tight_layout()
    plt.savefig(outdir + name + '_skymap-estimated_to_true_magnitude.pdf')



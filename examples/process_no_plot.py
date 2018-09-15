import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')

from quocca.camera import Camera
from quocca.image import Image
from quocca.catalog import Catalog
from quocca.detection import StarDetectionLLH, StarDetectionFilter
from quocca.plotting import show_img, add_circle, show_clouds
from quocca.plotting import compare_estimated_to_true_magnitude
from quocca.cloudiness import GaussianRunningAvg, RunningAvg, cloud_map

from glob import glob

import click


@click.command()
@click.option('-i', '--input', type=str)
@click.option('-o', '--output', type=str)
def process(input, output):
    cam = Camera('iceact')
    if '*' in input:
        files = glob(input)
    else:
        files = [input]
    for f in files:
        img = cam.read(f)
        img.add_catalog(max_mag=5.5, min_dist=12.0)
        result_llh = img.detect('llh', sigma=1.2, fit_size=4, presmoothing=0.5,
                                verbose=False)
        out_file_data = '{}/{}_llh_{}.csv'.format(output, cam.name, img.time)
        pd.DataFrame(result_llh).to_csv(out_file_data)

        result_flt = img.detect('filter', sigma=1.2, fit_size=4, verbose=False)
        out_file_data = '{}/{}_flt_{}.csv'.format(output, cam.name, img.time)
        pd.DataFrame(result_flt).to_csv(out_file_data)


if __name__ == '__main__':
    process()

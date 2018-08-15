"""quocca: All Sky cameraera Analysis Tools

Plotting.

2018"""

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from astropy import units as u


def show_img(img, ax=None, show_stars=False, max_mag=3.0, upper=99.8):
    """Shows the image img.

    Parameters
    ----------
    ax : matplotlib.Axes object
        Subplot in which to put the image. If None is passed, an Axes object
        will be given.
    show_stars : bool
        Whether or not to display stars up to magnitude max_mag in the image
    max_mag : float
        Maxmimum magnitude.
    upper : float
        Quantile to define the brightest spot in the image.

    Returns
    -------
    ax : matplotlib.Axes object
        Axes object containing the image.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))

    if upper > 100.0 or upper < 0.0:
        raise ValueError('upper needs to be in [0, 100]!')

    ax.imshow(img.image, vmin=0.0, vmax=np.percentile(img.image, upper),
               cmap='gray')
    if show_stars:
        display = img.star_mag < max_mag
        ax.scatter(img.star_pos[display, 1], img.star_pos[display, 0], s=50,
               marker='o', facecolor='', edgecolor='#7ac143')
    
    angles = [15, 30, 45, 60, 75, 90]
    circles = [Circle((img.camera.zenith['x'],
                       img.camera.zenith['y']),
                       img.camera.theta2r(angle * u.deg),
                       facecolor='none', ec='w', linestyle=':')
               for angle in angles]
    circles[-1].set_linestyle('-')
    for angle in angles:
        ax.text(img.camera.zenith['x'],
                 img.camera.zenith['y'] + img.camera.theta2r(angle * u.deg) - 10,
                 "{}".format(angle), color='w')
    ax.axvline(img.camera.zenith['x'], color='w', lw=1)
    ax.axhline(img.camera.zenith['y'], color='w', lw=1)
    for c__ in circles:
        ax.add_patch(c__)
    ax.axis('off')
    ax.text(0.01, 0.99, '{}\n{}'.format(img.time, img.camera.name), color='w',
             horizontalalignment='left', verticalalignment='top',
             transform=ax.transAxes)

    return ax
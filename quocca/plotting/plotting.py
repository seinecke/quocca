"""quocca: All Sky cameraera Analysis Tools

Plotting.

2018"""

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from astropy import units as u


def show_img(img, ax=None, show_stars=False, max_mag=3.0, color='#7ac143', upper=99.8):
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
               marker='o', facecolor='', edgecolor=color)
    
    angles = [30, 60, 90]
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


def add_circle(y, x, mag, ax, max_mag=20.0, size=20, color='red'):

    mask = mag < max_mag
    ax.scatter(y[mask], x[mask], s=size,
               marker='o', facecolor='', edgecolor=color)

    return ax


def compare_used_stars_to_catalog(img, res, max_mag=3.0):

    color_catalog = '#7ac143'
    color_used_stars = 'mediumblue'

    ax = img.show(show_stars=True, max_mag=max_mag, color=color_catalog)
    ax = add_circle(res.y, res.x, res.v_mag, ax, max_mag=max_mag, 
                    color=color_used_stars)

    ax.text(0.99, 0.95, 'Catalog Stars', color=color_catalog,
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)
    ax.text(0.99, 0.95, '\nUsed Stars', color=color_used_stars,
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)

    return ax


def compare_fitted_to_true_positions(img, res, max_mag=3.0):

    color_fitted = 'darkorange'
    color_true = 'mediumblue'

    ax = img.show(show_stars=False)
    ax = add_circle(res.y_fit, res.x_fit, res.v_mag, ax, max_mag=max_mag, 
                    color=color_fitted, size=20)
    ax = add_circle(res.y, res.x, res.v_mag, ax, max_mag=max_mag, 
                    color=color_true, size=60)

    ax.text(0.99, 0.95, 'Fitted', color=color_fitted,
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)
    ax.text(0.99, 0.95, '\nTrue', color=color_true,
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)

    return ax


def plot_visibility(res, ax=None):

    lowx = res.v_mag.min()
    upx = res.v_mag.max()
    binsx = (upx-lowx)/0.1

    lowy = np.log(res.M_fit).min()
    upy = np.log(res.M_fit).max()
    binsy = (upy-lowy)/0.1

    plt.hist2d((res.v_mag), np.log(res.M_fit), 
           bins=(binsx,binsy), range=((lowx, upx),(lowy,upy)), 
           norm=LogNorm(),
           cmap='Greens'
          )
    plt.colorbar()
    plt.xlabel('True Magnitude')
    plt.ylabel('Estimated Magnitude')





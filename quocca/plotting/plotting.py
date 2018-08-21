"""quocca: All Sky cameraera Analysis Tools

Plotting.

2018"""

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

from astropy import units as u


def show_img(img, ax=None, upper=99.8, alt_circles=[30, 60]):
    """Shows the image img.

    Parameters
    ----------
    ax : matplotlib.Axes object
        Subplot in which to put the image. If None is passed, an Axes object
        will be given.
    upper : float
        Quantile to define the brightest spot in the image.
    alt_circles : list, default=[30, 60]
        List of altitudes for which to draw circles.
    
    Returns
    -------
    ax : matplotlib.Axes object
        Axes object containing the image.
    """
    # Plotting style defintions for matplotlib.patches.Circle objects
    alt_circle_style = dict(
        facecolor='none',
        ec='w',
        linestyle='--'
    )
    # Plotting style defintions for matplotlib.pyplot.text objects
    alt_label_style = dict(
        color='w',
        horizontalalignment='left',
        verticalalignment='bottom'
    )
    # Plotting style defintions for matplotlib.pyplot.text objects
    card_label_style = dict(
        color='w',
        horizontalalignment='center',
        verticalalignment='center',
        fontdict={'fontsize': 12}
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))

    if upper > 100.0 or upper < 0.0:
        raise ValueError('upper needs to be in [0, 100]!')

    ax.imshow(img.image, vmin=0.0, vmax=np.percentile(img.image, upper),
               cmap='gray', interpolation='none')
    
    r = img.camera.radius
    phi = np.deg2rad(img.camera.az_offset.to_value())
    x0 = img.camera.zenith['x']
    y0 = img.camera.zenith['y']
    dir_ax1 = np.array([np.cos(phi), np.sin(phi)])
    dir_ax2 = np.array([-np.sin(phi), np.cos(phi)])
    alt_circles.append(90)
    
    circles = [Circle((img.camera.zenith['x'], img.camera.zenith['y']),
                      img.camera.theta2r(angle * u.deg),
                      **alt_circle_style)
               for angle in alt_circles]
    circles[-1].set_linestyle('-')
    rot_angles = np.array([-np.rad2deg(phi) + a
                           for a in [-270, -180, -90, 0, 90, 180, 270]]) 
    rot_angle = rot_angles[np.argwhere((rot_angles > -45) & (rot_angles < 45))[0]][0]
    
    for angle in alt_circles:
        pos = x0 + dir_ax1 * (img.camera.theta2r(angle * u.deg))
        ax.text(pos[0], pos[1], u' {}˚'.format(angle), rotation=rot_angle,
                **alt_label_style)
        pos = x0 + dir_ax2 * (img.camera.theta2r(angle * u.deg))
        ax.text(pos[0], pos[1], u' {}˚'.format(angle), rotation=rot_angle,
                **alt_label_style)
        pos = x0 - dir_ax1 * (img.camera.theta2r(angle * u.deg))
        ax.text(pos[0], pos[1], u' {}˚'.format(angle), rotation=rot_angle,
                **alt_label_style)
        pos = x0 - dir_ax2 * (img.camera.theta2r(angle * u.deg))
        ax.text(pos[0], pos[1], u' {}˚'.format(angle), rotation=rot_angle,
                **alt_label_style)
    
    pos = x0 + dir_ax1 * (img.camera.theta2r(98 * u.deg))
    ax.text(pos[0], pos[1], 'W', rotation=rot_angle, **card_label_style)
    pos = x0 + dir_ax2 * (img.camera.theta2r(98 * u.deg))
    ax.text(pos[0], pos[1], 'S', rotation=rot_angle, **card_label_style)
    pos = x0 - dir_ax1 * (img.camera.theta2r(98 * u.deg))
    ax.text(pos[0], pos[1], 'E', rotation=rot_angle, **card_label_style)
    pos = x0 - dir_ax2 * (img.camera.theta2r(98 * u.deg))
    ax.text(pos[0], pos[1], 'N', rotation=rot_angle, **card_label_style)
    
    ax.plot([-1.01 * r * np.cos(phi) + x0, 1.01 * r * np.cos(phi) + x0],
            [-1.01 * r * np.sin(phi) + y0, 1.01 * r * np.sin(phi) + y0], 'w')
    ax.plot([-1.01 * r * np.cos(phi + np.pi * 0.5) + x0,
             1.01 * r * np.cos(phi + np.pi * 0.5) + x0],
            [-1.01 * r * np.sin(phi + np.pi * 0.5) + y0,
             1.01 * r * np.sin(phi + np.pi * 0.5) + y0], 'w')
    for c__ in circles:
        ax.add_patch(c__)
    ax.text(0.01, 0.99, '{}\n{}'.format(img.time, img.camera.name), color='w',
             horizontalalignment='left', verticalalignment='top',
             transform=ax.transAxes)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_facecolor('k')

    # Inserts image into a larger paspartout
    larger_frame_x_hi = img.camera.resolution['x'] * 1.06
    larger_frame_x_lo = -img.camera.resolution['x'] * 0.06
    larger_frame_y_hi = img.camera.resolution['y'] * 1.06
    larger_frame_y_lo = -img.camera.resolution['y'] * 0.06
    ax.set_xlim([larger_frame_x_lo, larger_frame_x_hi])
    ax.set_ylim([larger_frame_y_lo, larger_frame_y_hi])
    return ax


def show_stars(posx, posy, mag, max_mag, ax=None, detected=None, star_size=30, color='#7ac143'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
        
    display = mag < max_mag
    ax.scatter(posx[display], posy[display],
               s=star_size, marker='o', facecolor='', edgecolor=color)
    if detected is not None:
        ax.scatter(posx[display], posy[display],
                   s=star_size * detected, marker='o', facecolor=color, edgecolor='')
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


def compare_estimated_to_true_magnitude(res, det, ax=None):
    lowx = res.v_mag.min()
    upx = res.v_mag.max()
    binsx = (upx-lowx)/0.1

    lowy = np.log(res.M_fit).min()
    upy = np.log(res.M_fit).max()
    binsy = (upy-lowy)/0.1

    plt.hist2d((res.v_mag), np.log(res.M_fit), 
           bins=(binsx,binsy), range=((lowx, upx),(lowy,upy)), 
           norm=LogNorm(),
           cmap='Greens_r'
          )
    plt.plot([-5,25], -np.log(det.calibration) - np.array([-5,25]), 
        'r-', label='Clear Sky')

    plt.colorbar()
    plt.legend(frameon=False, loc='lower left')
    plt.xlabel('True Magnitude')
    plt.ylabel('Estimated Magnitude')

    return ax


def compare_visibility_to_magnitude(res, ax=None):
    lowx = res.v_mag.min()
    upx = res.v_mag.max()
    binsx = (upx-lowx)/0.1

    plt.hist2d(res.v_mag, res.visibility,
               norm=LogNorm(), bins=(binsx, 60), 
               range=((lowx,upx),(-0.1,3)),
               cmap='Greens_r')

    plt.colorbar()
    plt.xlabel('True Magnitude')
    plt.ylabel('Visibility')

    return ax

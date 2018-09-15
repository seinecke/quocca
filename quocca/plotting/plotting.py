"""quocca: All Sky cameraera Analysis Tools

Plotting.

2018"""

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap

from astropy import units as u

from skimage.transform import resize


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
    pos0 = np.array([x0, y0])
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
    rot_angle = rot_angles[np.argwhere((rot_angles > -45)
                                       & (rot_angles < 45))[0]][0]
    
    for angle in alt_circles:
        pos = pos0 + dir_ax1 * (img.camera.theta2r(angle * u.deg))
        ax.text(pos[0], pos[1], u' {}˚'.format(angle), rotation=rot_angle,
                **alt_label_style)
        pos = pos0 + dir_ax2 * (img.camera.theta2r(angle * u.deg))
        ax.text(pos[0], pos[1], u' {}˚'.format(angle), rotation=rot_angle,
                **alt_label_style)
        pos = pos0 - dir_ax1 * (img.camera.theta2r(angle * u.deg))
        ax.text(pos[0], pos[1], u' {}˚'.format(angle), rotation=rot_angle,
                **alt_label_style)
        pos = pos0 - dir_ax2 * (img.camera.theta2r(angle * u.deg))
        ax.text(pos[0], pos[1], u' {}˚'.format(angle), rotation=rot_angle,
                **alt_label_style)
    
    pos = pos0 + dir_ax1 * (img.camera.theta2r(98 * u.deg))
    ax.text(pos[0], pos[1], 'W', rotation=rot_angle, **card_label_style)
    pos = pos0 + dir_ax2 * (img.camera.theta2r(98 * u.deg))
    ax.text(pos[0], pos[1], 'S', rotation=rot_angle, **card_label_style)
    pos = pos0 - dir_ax1 * (img.camera.theta2r(98 * u.deg))
    ax.text(pos[0], pos[1], 'E', rotation=rot_angle, **card_label_style)
    pos = pos0 - dir_ax2 * (img.camera.theta2r(98 * u.deg))
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
    frame_size = np.max([img.camera.resolution['x'],
                         img.camera.resolution['y']])
    larger_frame_x_hi = x0 + frame_size * 0.57
    larger_frame_x_lo = x0 - frame_size * 0.57
    larger_frame_y_hi = y0 + frame_size * 0.57
    larger_frame_y_lo = y0 - frame_size * 0.57
    
    ax.set_xlim([larger_frame_x_lo, larger_frame_x_hi])
    ax.set_ylim([larger_frame_y_hi, larger_frame_y_lo])
    return ax


def show_clouds(img, cloudmap, ax=None, color='#7ac143', opaque=False,
                **kwargs):
    cmap = plt.cm.rainbow
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)
    
    cloudmap_fit = resize(cloudmap, (img.camera.resolution['x'],
                                     img.camera.resolution['y']))
    tx = np.arange(img.camera.resolution['x'])
    ty = np.arange(img.camera.resolution['y'])
    mx, my = np.meshgrid(tx, ty)
    cloudmap_fit[img.camera.mask((mx, my)) == 0] = 0.0
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    if opaque:
        cloudmap_fit[cloudmap_fit == 0.0] = np.nan
        ax.contourf(tx, ty, 1.0 - (cloudmap_fit).T, cmap=my_cmap,
                    levels=np.linspace(0.0, 1.0, 20))
    else:
        ax.contour(tx, ty, (cloudmap_fit).T, colors=color,
                   linestyles=['-', '--', ':'],
                   levels=[0.25, 0.5, 0.75])
    return ax


def add_circle(posx, posy, mag, max_mag=20.0, size=30, color='#7ac143',
               ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
        
    display = mag < max_mag
    ax.scatter(posx[display], posy[display],
               s=size, marker='o', facecolor='', edgecolor=color)

    return ax


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

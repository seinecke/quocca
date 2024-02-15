# quocca: QUick Observation of Cloud Coverage using All sky images
#  
# Contains rudimentary plotting methods that you may not want to implement on
# your own, such as
#
# 1. `show_img`: Plots a neat looking picture of a sky cam image with some
#     additional guides, such as altitude lines etc.
# 2. `show_clouds`: Overlays an image of a cloud.
# 3. `show_circle`: Draws circles into a plot, e.g. to plot stars detected
#     in an image etc.
#
# The basic idea behind using this submodule is that you initialize an Axes
# object with one of the method and pass it through the other methods, e.g.
# ```
# ax = show_img(img)
# ax = show_clouds(cmap, ax=ax)
# ax = show_circle(x, y, m, ax=ax)
# ```
# etc.
#
# Authors: S. Einecke <sabrina.einecke@adelaide.edu.au>
#          T. Hoinka <tobias.hoinka@icecube.wisc.edu>
#          H. Nawrath <helena.nawrath@tu-dortmund.de>

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
    phi = np.deg2rad(img.camera.az_offset)
    x0 = img.camera.zenith['x']
    y0 = img.camera.zenith['y']
    pos0 = np.array([x0, y0])
    dir_ax1 = np.array([np.cos(phi), np.sin(phi)])
    dir_ax2 = np.array([-np.sin(phi), np.cos(phi)])

    # Append 90 deg circle, which marks the horizon and should always be
    # present.
    alt_circles = alt_circles + [90]
    
    # Define altitude circles.
    circles = [Circle((img.camera.zenith['x'], img.camera.zenith['y']),
                      img.camera.theta2r(angle),
                      **alt_circle_style)
               for angle in alt_circles]
    circles[-1].set_linestyle('-')
    for c__ in circles:
        ax.add_patch(c__)

    # Calculate rotation angle that is closest to horizontal for all text
    # elements.
    rot_angles = np.array([-np.rad2deg(phi) + a
                           for a in [-270, -180, -90, 0, 90, 180, 270]]) 
    rot_angle = rot_angles[np.argwhere((rot_angles > -45)
                                       & (rot_angles < 45))[0]][0]
    
    # Add labels for altitude circles.
    for angle in alt_circles:
        pos = pos0 + dir_ax1 * (img.camera.theta2r(angle))
        ax.text(pos[0], pos[1], u' {}˚'.format(angle), rotation=rot_angle,
                **alt_label_style)
        pos = pos0 + dir_ax2 * (img.camera.theta2r(angle))
        ax.text(pos[0], pos[1], u' {}˚'.format(angle), rotation=rot_angle,
                **alt_label_style)
        pos = pos0 - dir_ax1 * (img.camera.theta2r(angle))
        ax.text(pos[0], pos[1], u' {}˚'.format(angle), rotation=rot_angle,
                **alt_label_style)
        pos = pos0 - dir_ax2 * (img.camera.theta2r(angle))
        ax.text(pos[0], pos[1], u' {}˚'.format(angle), rotation=rot_angle,
                **alt_label_style)
    
    # Add labels for cardinal directions.
    pos = pos0 + dir_ax1 * (img.camera.theta2r(98))
    ax.text(pos[0], pos[1], 'W', rotation=rot_angle, **card_label_style)
    pos = pos0 + dir_ax2 * (img.camera.theta2r(98))
    ax.text(pos[0], pos[1], 'S', rotation=rot_angle, **card_label_style)
    pos = pos0 - dir_ax1 * (img.camera.theta2r(98))
    ax.text(pos[0], pos[1], 'E', rotation=rot_angle, **card_label_style)
    pos = pos0 - dir_ax2 * (img.camera.theta2r(98))
    ax.text(pos[0], pos[1], 'N', rotation=rot_angle, **card_label_style)
    
    # Add cross.
    ax.plot([-1.01 * r * np.cos(phi) + x0, 1.01 * r * np.cos(phi) + x0],
            [-1.01 * r * np.sin(phi) + y0, 1.01 * r * np.sin(phi) + y0], 'w')
    ax.plot([-1.01 * r * np.cos(phi + np.pi * 0.5) + x0,
             1.01 * r * np.cos(phi + np.pi * 0.5) + x0],
            [-1.01 * r * np.sin(phi + np.pi * 0.5) + y0,
             1.01 * r * np.sin(phi + np.pi * 0.5) + y0], 'w')

    # Short descriptor of image with cam name and time.
    ax.text(0.01, 0.99, '{}\n{}'.format(img.time.datetime, img.camera.name),
             color='w', horizontalalignment='left', verticalalignment='top',
             transform=ax.transAxes)
    
    # Axes begone!
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


def show_clouds(img, cloudmap, ax=None, **kwargs):
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
    cloudmap_fit[cloudmap_fit == 0.0] = np.nan
    ax.contourf(tx, ty, 1.0 - (cloudmap_fit).T, cmap=my_cmap,
                levels=np.linspace(0.0, 1.0, 20))
    return ax


def show_circle(posx, posy, mag, max_mag=20.0, size=30, color='#7ac143',
              ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
        
    display = mag < max_mag
    ax.scatter(posy[display], posx[display],
               s=size, marker='o', facecolor=[0,0,0,0], edgecolor=color)

    return ax
    

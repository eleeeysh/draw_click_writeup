import numpy as np
import matplotlib.pyplot as plt


""" for gaze features """
from matplotlib.patches import Wedge
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_single_1d_gaze(d1_gaze_pattern, angle, ax, set_title=True):
    norm_values = d1_gaze_pattern / np.max(np.abs(d1_gaze_pattern))
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = cm.bwr
    n_bins = len(d1_gaze_pattern)
    angle_step = 360 / n_bins
    for i in range(n_bins):
        end_angle = (90-(i * angle_step)) % 360
        start_angle = (90-((i+1) * angle_step)) % 360
        color = cmap(norm(norm_values[i]))
        wedge = Wedge((0, 0), 1, start_angle, end_angle, color=color)
        ax.add_patch(wedge)

    # plot the grating bar
    if angle is not None:
        bar_length = 0.5
        bar_rad = np.radians(90 - angle)
        ax.plot(
            [-np.cos(bar_rad)*bar_length, np.cos(bar_rad)*bar_length],
            [-np.sin(bar_rad)*bar_length, np.sin(bar_rad)*bar_length],
            lw=50, c='purple')
    
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    if set_title:
        ax.set_title(f'{angle:.0f}°', fontsize=AX_TITLE_SIZE*3)

def plot_single_2d_gaze(d2_gaze_pattern, settings, angle, ax, show_grating=True, set_title=True):
    # normalize
    norm_values = d2_gaze_pattern / np.max(np.abs(d2_gaze_pattern))
    # norm = mcolors.Normalize(vmin=-1, vmax=1)
    # cmap = cm.bwr
    cmap = 'coolwarm'

    # reshape
    width = settings['2dhist']['vecmap']['n_bins']
    reshaped = np.reshape(norm_values, (width, width))
    ax.imshow(
        reshaped, cmap=cmap, 
        # norm=norm,
        origin='lower')

    # plot the grating bar
    if show_grating:
        bar_length = (width/2) * 0.5
        lw = 15 * (width/2) * 0.5
        bar_rad = np.radians(90 - angle)
        x_offset = width / 2 - 0.5
        y_offset = width / 2 - 0.5
        ax.plot(
            [x_offset-np.cos(bar_rad)*bar_length, x_offset+np.cos(bar_rad)*bar_length],
            [y_offset-np.sin(bar_rad)*bar_length, y_offset+np.sin(bar_rad)*bar_length],
            lw=lw, c='purple')
    
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_xticks([])
    ax.set_yticks([])

    if set_title:
        ax.set_title(f'{angle:.0f}°', fontsize=AX_TITLE_SIZE*3)



LABEL_FONT_SIZE = 20
AX_TITLE_SIZE=40

""" for hand motion features """
def plot_single_1d_plot_vectors(d1_pattern, angle, ax, set_title=True):
    # re-center
    d1_pattern = d1_pattern - 0.5

    # rescale
    scale = np.max(np.abs(d1_pattern))
    baseline_scale = 0.7
    rescale_factor = 0.3
    norm_values = d1_pattern / scale * rescale_factor + baseline_scale

    n_bins = len(d1_pattern)
    angle_step = 360 / n_bins

    # define start points
    plot_angles = (90-((np.arange(n_bins)+0.5) * angle_step)) % 360
    plot_rads = np.deg2rad(plot_angles)
    center_offset = 0.05
    xs, ys = center_offset * np.cos(plot_rads), center_offset * np.sin(plot_rads)

    # define vectors
    vec_len_factor = 1.2
    vec_len = norm_values * vec_len_factor
    us, vs = np.cos(plot_rads) * vec_len, np.sin(plot_rads) * vec_len

    # plot a circle at background
    cir_r = vec_len_factor * baseline_scale + center_offset
    circle = plt.Circle(
        (0, 0), radius=cir_r, fill=False, edgecolor='lightgray', linewidth=2)
    ax.add_patch(circle)

    # quiver
    ax.quiver(
        xs, ys, us, vs,
        angles='xy', scale_units='xy', scale=1,
        width=0.03, color='gray',
        headwidth=2, headaxislength=2, headlength=2)

    
    # plot the grating bar
    if angle is not None:
        print(angle)
        bar_length = 0.5
        bar_rad = np.radians(90 - angle)
        ax.plot(
            [-np.cos(bar_rad)*bar_length, np.cos(bar_rad)*bar_length],
            [-np.sin(bar_rad)*bar_length, np.sin(bar_rad)*bar_length],
            lw=50, c='purple')
    
    plot_scale = 1.2
    ax.set_xlim([-plot_scale, plot_scale])
    ax.set_ylim([-plot_scale, plot_scale])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    if set_title:
        ax.set_title(f'{angle:.0f}°', fontsize=AX_TITLE_SIZE*3)

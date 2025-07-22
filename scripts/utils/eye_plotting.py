import matplotlib.pyplot as plt
import numpy as np

STIM1_COLOR = '#A96FAE'
STIM2_COLOR = '#83B174'

def set_general_plt_styles():
    # set background of plots
    plt.rcParams['figure.facecolor'] = 'none'
    plt.rcParams['savefig.facecolor'] = 'none'

    # set the font
    plt.rcParams['font.family'] = 'Arial'

    # set axis colors
    axis_color = '#444444'
    ## Set axis edge color to gray
    plt.rcParams['axes.edgecolor'] = axis_color
    plt.rcParams['xtick.color'] = axis_color
    plt.rcParams['ytick.color'] = axis_color
    plt.rcParams['axes.labelcolor'] = axis_color

    # set axis title color
    plt.rcParams['axes.titlecolor'] = axis_color

    # make the axis line thicker
    plt.rcParams['axes.linewidth'] = 1.2

def annotate_time_line(ax, events, min_time=None, max_time=None,
        plot_ymin=None, plot_ymax=None, 
        hide_ymin=None, hide_ymax=None,
        to_simplify=False):    
    min_time = 0 if min_time is None else min_time
    max_time = max(events.values())+5000 if max_time is None else max_time

    # remove any x label or ticks
    ax.set_xticks([])

    cue_name_to_annot_mappings = {
        's1 onset': 'S1',
        's2 onset': 'S2',
        's1 cue onset': 'S1+cue',
        's2 cue onset': 'S2+cue',
        's1 delay mask onset': 'mask',
        's2 delay mask onset': 'mask',
        's1 delay onset': 'ISI',
        's2 delay onset': 'delay',
        'response': 'response',
    }

    plot_transform = plt.gca().get_xaxis_transform()

    # drop some annotations if to_simplify is True
    simplified_set = set([
        's1 onset', 's2 onset',
        's1 delay mask onset', 's2 delay mask onset',
        'response'
    ])
    if to_simplify:
        cue_name_to_annot_mappings['s1 delay mask onset'] = 'ISI'
        cue_name_to_annot_mappings['s2 delay mask onset'] = 'Delay'
        cue_name_to_annot_mappings['response'] = 'Recall'
    
    # text annotation
    for time_name, time_point in events.items():
        if time_point < min_time or time_point > max_time:
            continue

        if to_simplify and time_name not in simplified_set:
            continue

        # replace the delay name
        time_name = cue_name_to_annot_mappings.get(time_name, time_name)
        
        # s1, s2, or response?
        text_color = '#333333'
        if time_name.startswith('S1'):
            text_color = STIM1_COLOR
        if time_name.startswith('S2'):
            text_color = STIM2_COLOR
            
        # specify line alpha
        line_alpha = 0.7
        if time_name.endswith('cue onset') or time_name.endswith('delay mask onset'):
            line_alpha = 0.3
        
        # mark time line
        all_plot_ymins = [plot_ymin, hide_ymax]
        all_plot_ymaxs = [hide_ymin, plot_ymax]
        if hide_ymax is None:
            all_plot_ymins = [plot_ymin,]
            all_plot_ymaxs = [plot_ymax,]
        for pymin, pymax in zip(all_plot_ymins[:1], all_plot_ymaxs[:1]):
        # for pymin, pymax in zip(all_plot_ymins, all_plot_ymaxs):
            ax.plot(
                [time_point, time_point], 
                [pymin, pymax], color=text_color, alpha=line_alpha,
                linewidth=2, linestyle='--'
            )

        # add time point annotation text
        annot_x, annot_y = time_point, -0.02
        text_rotate = 0
        text_font_size = 18
        text_ha, text_va = 'center', 'top'
        if not to_simplify:
            annot_x += 400
            text_rotate = 30
            text_font_size = 16
            text_ha, text_va = 'right', 'top'
            if len(time_name) < 3:
                annot_x = annot_x - 200
        else:
            # move the last label a bit inward
            if time_name == 'Response':
                annot_x -= 400
        ax.text(
            annot_x, annot_y, time_name, 
            color=text_color, rotation=text_rotate, 
            fontsize=text_font_size,
            ha=text_ha, va=text_va, transform=plot_transform,
        )


""" plot distribution of angles """
def plot_angle_distrib_polar(ax, hist, bin_edges, stim_align=True):    
    # Plot the bars in polar coordinates
    n_bins = len(bin_edges) - 1
    width = bin_edges[1] - bin_edges[0]
    width = (2 * np.pi) / n_bins 
    bars = ax.bar(bin_edges[:-1], hist, width=width, align='edge', edgecolor='black')

    # Optional: Set the direction and offset (if needed)
    if stim_align:
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_theta_offset(np.pi/2)  # Start from top (90°)
    ax.set_yticks([])

def plot_mag_and_angle_distrib(ax, H, angle_bin_edges, mag_bin_edges):
    _ = ax.imshow(H, interpolation='nearest', origin='lower')
    ax.set_xticks([-0.5, len(angle_bin_edges)-1.5])
    ax.set_xticklabels([angle_bin_edges[0], f'{angle_bin_edges[-1]:.1f}'])
    ax.set_yticks([-0.5, len(mag_bin_edges)-1.5])
    ax.set_yticklabels([mag_bin_edges[0], f'{mag_bin_edges[-1]:.1f}'])
    ax.set_xlabel('direction')
    ax.set_ylabel('magnitude')
    
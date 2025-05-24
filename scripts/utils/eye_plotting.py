import matplotlib.pyplot as plt
import numpy as np

def annotate_time_line(ax, events, min_time=None, max_time=None,
        plot_ymin=-0.15, plot_ymax=0.35, hide_ymin=-0.08, hide_ymax=0.3):    
    min_time = 0 if min_time is None else min_time
    max_time = max(events.values())+5000 if max_time is None else max_time

    # remove any x label or ticks
    ax.set_xticks([])

    cue_name_to_annot_mappings = {
        's1 onset': 's1',
        's2 onset': 's2',
        's1 cue onset': 's1+cue',
        's2 cue onset': 's2+cue',
        's1 delay mask onset': 'mask',
        's2 delay mask onset': 'mask',
        's1 delay onset': 'ISI',
        's2 delay onset': 'delay',
        'response': 'response',
    }

    plot_transform = plt.gca().get_xaxis_transform()
    
    # text annotation
    for time_name, time_point in events.items():
        if time_point < min_time or time_point > max_time:
            continue

        # replace the delay name
        time_name = cue_name_to_annot_mappings.get(time_name, time_name)
        
        # s1, s2, or response?
        text_color = 'gray'
        if time_name.startswith('s1'):
            text_color = 'purple'
        if time_name.startswith('s2'):
            text_color = 'green'
            
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
        for pymin, pymax in zip(all_plot_ymins, all_plot_ymaxs):
            ax.plot(
                [time_point, time_point], 
                [pymin, pymax], color=text_color, alpha=line_alpha,
                linewidth=2, linestyle='--'
            )

        # add time point annotation text
        annot_x, annot_y = time_point + 400, -0.02
        if len(time_name) < 3:
            annot_x = annot_x - 200
        ax.text(
            annot_x, annot_y, time_name, color=text_color, rotation=30, fontsize=16,
            ha='right', va='top', transform=plot_transform,
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
    
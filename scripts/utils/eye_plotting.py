import matplotlib.pyplot as plt
import numpy as np

def annotate_time_line(ax, events, min_time=None, max_time=None):    
    min_time = 0 if min_time is None else min_time
    max_time = max(events.values())+5000 if max_time is None else max_time
    
    # text annotation
    for time_name, time_point in events.items():
        if time_point < min_time or time_point > max_time:
            continue

        # replace the delay name
        if time_name.startswith('s1 delay'):
            time_name = time_name.replace('s1 delay', 'ISI')
        if time_name.startswith('s2 delay'):
            time_name = time_name.replace('s2 delay', 'delay')
        
        # s1, s2, or response?
        text_color = 'black'
        if time_name.startswith('s1'):
            text_color = 'purple'
        if time_name.startswith('s2'):
            text_color = 'green'
            
        # specify line alpha
        line_alpha = 0.6
        if time_name.endswith('cue onset') or time_name.endswith('delay mask onset'):
            line_alpha = 1.0
        
        ax.axvline(time_point, color=text_color, alpha=line_alpha)
        ax.text(
            time_point, 1.05, time_name, color=text_color, rotation=45, fontsize=10,
            ha='left', va='bottom', transform=plt.gca().get_xaxis_transform()
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
    
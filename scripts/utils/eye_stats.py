import numpy as np
from scipy.ndimage import convolve1d
from .eye_data import XYData

""" compute frequency of events """
def calculate_freq_of_events_times(event_times, smooth_window, max_time):
    count_table = np.zeros(max_time)
    np.add.at(count_table, (event_times,), 1)
    
    # smoothing
    weights = np.ones(smooth_window)/smooth_window
    smoothed = convolve1d(count_table, weights, mode='constant', cval=0)
    return smoothed

def calculate_freq_of_events_table(events, smooth_window):
    happen_mask = np.any((np.array(events) != 0), axis=0)
    event_ts = np.nonzero(happen_mask)[1]
    max_time = happen_mask.shape[-1]
    return calculate_freq_of_events_times(event_ts, smooth_window, max_time)


""" extract non-zero event and convert to angles """
def collapse_event_data(xs, ys):
    mask = (xs != 0) | (ys != 0) 
    idx = np.nonzero(mask)
    xs_masked = xs[idx]
    ys_masked = ys[idx]
    return xs_masked, ys_masked

""" extract distribution of angles """
def generate_angle_distrib(angle, mag, mag_weight=False, min_mag_thresh=0, max_mag_thresh=np.inf, n_bins=36):    
    # filter out those too small or too large?
    mask = (mag >= min_mag_thresh) & (mag <= max_mag_thresh)
    angle = angle[mask]
    mag = mag[mask]
    
    # convert to rad
    rads = np.deg2rad(angle)
    rads = rads % (np.pi * 2)

    # plot distribution
    bins = np.linspace(0, 2 * np.pi, n_bins+1)  # 19 edges to create 18 bins
    hist_weights = mag if mag_weight else None
    hist, bin_edges = np.histogram(rads, bins=bins, weights=hist_weights)

    return hist, bin_edges

def generate_mag_and_angle_distrib(angle, mag, n_angle_bins, n_mag_bins, min_mag_thresh=0, max_mag_thresh=np.inf, log_transform=False):
    # filter out those too small or too large?
    mask = (mag >= min_mag_thresh) & (mag <= max_mag_thresh)
    angle = angle[mask]
    mag = mag[mask]

    # apply log transformation (if required)
    if log_transform:
        mag = np.log(mag+1)
        mag_bins = np.linspace(np.log(min_mag_thresh+1), np.log(max_mag_thresh+1), n_mag_bins+1)
    else:
        mag_bins = np.linspace(min_mag_thresh, max_mag_thresh, n_mag_bins+1)

    angle_bins = np.linspace(0, 360, n_angle_bins+1)

    H, angle_bins, mag_bins = np.histogram2d(angle, mag, bins=(angle_bins, mag_bins))

    # convert mag_bins back
    if log_transform:
        mag_bins = np.exp(mag_bins) - 1

    H = H.T

    return H, angle_bins, mag_bins

""" compute 2d-hist of data  """
def compute_vecmap(xs, ys, 
        x_center=0, y_center=0, 
        x_radius=80, y_radius=80, 
        n_bins=50, log_transform=True):
    xs, ys = collapse_event_data(xs, ys)
    if log_transform:
        x_bins = np.linspace(-np.log(x_radius+1), np.log(x_radius+1), n_bins+1)
        x_bins = (np.exp(np.abs(x_bins)+1)-1) * np.sign(x_bins)
        y_bins = np.linspace(-np.log(y_radius+1), np.log(y_radius+1), n_bins+1)
        y_bins = (np.exp(np.abs(y_bins)+1)-1) * np.sign(y_bins)
    else:
        x_bins = np.linspace(-x_radius, x_radius, n_bins+1)
        y_bins = np.linspace(-y_radius, y_radius, n_bins+1)
    
    x_bins += x_center
    y_bins += y_center
    hist, x_bins, y_bins = np.histogram2d(xs, ys, bins=(x_bins, y_bins))
    hist = hist.T # make it cartesian
    return hist, x_bins, y_bins


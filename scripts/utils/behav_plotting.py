import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from scipy.stats import sem

def deg_signed_diff(d1, epoch=180):
    d2 = - np.sign(d1) * (epoch - np.abs(d1))
    mask = np.abs(d1) < np.abs(d2)
    d = mask * d1 + (~mask) * d2
    # finally, convert 90 to -90
    half_epoch = epoch // 2
    mask_90 = d == half_epoch
    d = mask_90 * (-half_epoch) + (~mask_90) * d
    return d

smart_diff = lambda x1, x2: deg_signed_diff(x1-x2, epoch=180)

""" functions to plot contextual biases """
# apply IQR filtering
def get_IQR_mask(ys, factor=1.5):
    q1 = np.percentile(ys, 25)
    q3 = np.percentile(ys, 75)
    lower = q1 - factor * (q3 - q1)
    upper = q3 + factor * (q3 - q1)
    filter_mask = (ys >= lower) & (ys <= upper)
    return filter_mask
    
# plot binned data
def get_binned_impact_data(
        xs, ys, 
        max_y, min_n_points,
        n_flip, always_pos,
        apply_IQR_mask=True,
        sample_width=None, bin_width=None):
    
    # preprocess: filter NA and flipping
    to_keep = (~np.isnan(xs)) & (~np.isnan(ys)) 
    xs, ys = xs[to_keep], ys[to_keep]
    if n_flip == 1:
        to_flip_mask = xs < 0
        xs[to_flip_mask] = - xs[to_flip_mask]
        if not always_pos:
            ys[to_flip_mask] = -ys[to_flip_mask]
    elif n_flip == 2:
        # correct xs by categories
        # 0: the cardinal axis
        # 45: the oblique axis
        xs = xs % 90
        to_flip_mask = xs > 45
        xs[to_flip_mask] = 90 - xs[to_flip_mask]
        if not always_pos:
            ys[to_flip_mask] = -ys[to_flip_mask]

    # step 1: bin data
    if sample_width is None:
        sample_width = 10 if n_flip <= 1 else 5
    
    if n_flip == 0:
        bin_centers = np.arange(-90, 91, sample_width)
    elif n_flip == 1:
        bin_centers = np.arange(0, 91, sample_width)
    elif n_flip == 2:
        bin_centers = np.arange(0, 46, sample_width)
    else:
        raise NotImplementedError(f'wrong n flips: {n_flip}')
    
    # step 2: compute mean and sem
    x_centers = []
    y_means = []
    y_sems = []
    x_data_grouped = []
    y_data_grouped = []
    if bin_width is None:
        bin_width = 10
    for i in range(len(bin_centers)):
        bin_center = bin_centers[i]
        bin_mask = np.abs(smart_diff(xs, bin_center)) <= bin_width
        x_binned = xs[bin_mask]
        y_binned = ys[bin_mask]
        
        if np.sum(bin_mask) >= min_n_points and apply_IQR_mask:
            # apply IQR filtering to filter outliers
            bin_mask = get_IQR_mask(y_binned)
            x_binned = x_binned[bin_mask]
            y_binned = y_binned[bin_mask]
        
        if np.sum(bin_mask) >= min_n_points:
            x_centers.append(bin_center)
            y_means.append(np.mean(y_binned))
            y_sems.append(sem(y_binned))
            x_data_grouped.append(np.copy(x_binned))
            y_data_grouped.append(np.copy(y_binned))
    
    # step 3: return results
    return {
        'x': x_centers,
        'x_grouped': x_data_grouped,
        'y': y_data_grouped,
        'x_raw': xs,
        'y_raw': ys,
        'y_mean': y_means,
        'y_sem': y_sems,
        'n_total': len(xs), 
    }

# plot binned data
def plot_binned_impact(
        ax, xs, ys, 
        max_y, min_n_points,
        n_flip, always_pos, label,
        apply_IQR_mask=True,
        sample_width=None, bin_width=None):
    
    binned_data = get_binned_impact_data(
        xs, ys, 
        max_y, min_n_points,
        n_flip, always_pos,
        apply_IQR_mask=apply_IQR_mask,
        sample_width=sample_width, bin_width=bin_width)
    
    ax.errorbar(
        binned_data['x'], 
        binned_data['y_mean'], 
        yerr=binned_data['y_sem'], 
        label=label)
    
    return binned_data
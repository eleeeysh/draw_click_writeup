import numpy as np
import pandas as pd
from .eye_preprocess import convert_movement_to_angle


### detect strokes
def detect_derivative(xdata, ydata, frequency):
    # get the timing
    n_timing = xdata.shape[-1]
    timing = 1000 / frequency * np.arange(n_timing)
    # get derivatives
    der_x = xdata[..., 1:] - xdata[..., :-1]
    der_y = ydata[..., 1:] - ydata[..., :-1]
    der_t = (timing[1:] + timing[:-1]) / 2
    # a motion is considered a 
    # make der_t integert
    der_t = der_t.astype(int)
    return der_x, der_y, der_t

DEFAULT_DM_SETTING = {
    'MIN_N': 3,
}

def filter_motion(dx, dy, dt, settings=DEFAULT_DM_SETTING):
    # convert to angle and magnitude
    angles, mags = convert_movement_to_angle(dx, dy, compute_mag=True)
    motion_raw_mask = mags > 0
    
    # discard jittering
    filtered_mask = np.zeros_like(motion_raw_mask, dtype=int)
    min_n = settings['MIN_N']
    for i, row in enumerate(motion_raw_mask):
        ones_regions = np.split(
            np.where(row == 1)[0], 
            np.where(np.diff(np.where(row == 1)[0]) > 1)[0] + 1)
        
        # Keep only the sequences of at least length N
        for region in ones_regions:
            if len(region) >= min_n:
                filtered_mask[i, region] = 1
                
    motion_mask = filtered_mask
    mags[~motion_mask] = 0
    angles[~motion_mask] = 0
    
    return angles, mags, motion_mask

def angle_diff_abs(x1, x2, epoch=360):
    d = np.abs(x1 - x2)
    d = np.min([d, epoch-d], axis=0)
    return d

def create_stroke(dx, dy, dt):
    time_range = (dt[0], dt[-1])
    dx_all = np.sum(dx)
    dy_all = np.sum(dy)
    angle, mag = convert_movement_to_angle(dx_all, dy_all, compute_mag=True)
    return {
        'angle': angle,
        'mag': mag,
        'tstart': time_range[0],
        'tend': time_range[-1],
    }

def concatenate_strokes(dx, dy, dt, angles, mags, max_diff):
    mask = mags > 0
    strokes = []
    last_xs, last_ys, last_ts, last_angles = [], [], [], []
    last_idx = -1
    for idx in np.where(mask)[0]:
        if (idx - last_idx > 1) & (len(last_angles) > 0):
            # there is a gap
            strokes.append(create_stroke(last_xs, last_ys, last_ts))
            last_xs, last_ys, last_ts, last_angles = [], [], [], []

        # check whether the current one should belong to the last stroke
        if len(last_angles) > 0:
            cur_angle = angles[idx]
            angle_diffs = angle_diff_abs(cur_angle, last_angles)
            if np.max(angle_diffs) > max_diff:
                # the current angle does not match others
                strokes.append(create_stroke(last_xs, last_ys, last_ts))
                last_xs, last_ys, last_ts, last_angles = [], [], [], []
        
        # update
        last_xs.append(dx[idx])
        last_ys.append(dy[idx])
        last_ts.append(dt[idx]) 
        last_angles.append(angles[idx])

        # update
        last_idx = idx
        
    # if there are unprocessed
    if len(last_xs) > 0:
        strokes.append(create_stroke(last_xs, last_ys, last_ts))
        
    # finally, kick out all whose length is less than or equal 1
    strokes = [s for s in strokes if s['tstart'] != s['tend']]
        
    return strokes

def compute_one_trial_movement(trial_id, dx, dy, dt, angles, mags, max_diff=15):
    return concatenate_strokes(
        dx[trial_id], 
        dy[trial_id],
        dt,
        angles[trial_id], 
        mags[trial_id], max_diff=max_diff)

PHASE_OFFSETS = {
    'display_1': 1000,
    'post_stim_1': 1750,
    'display_2': 3250,
    'post_stim_2': 4000,
    'delay': 5500,
}

T_TOTAL = 10000

def aggregate_stroke_events(subject_source, phase_offsets, filter_settings=DEFAULT_DM_SETTING, concat_max_diff=15):
    results = {}
    
    N_trials = len(subject_source['delay.x'])
    for phase_name, phase_offset in phase_offsets.items():
        phase_results = []
        npz_xs, npz_ys = subject_source[f'{phase_name}.x'], subject_source[f'{phase_name}.y']
        # print(npz_xs.shape, npz_ys.shape)
        frequency = subject_source['frequency'].item()
        dx, dy, dt = detect_derivative(npz_xs, npz_ys, frequency)
        dt += phase_offset
        angles, mags, mask = filter_motion(dx, dy, dt, settings=filter_settings)
        
        for trial_id in range(N_trials):
            strokes = concatenate_strokes(
                dx[trial_id], 
                dy[trial_id],
                dt,
                angles[trial_id], 
                mags[trial_id], max_diff=concat_max_diff)
            phase_results.append(strokes)
            
        results[phase_name] = phase_results
        
    return results

### read and write the strokes extracted
def stroke_detected_to_dicts(events_raw):
    # convert dictionary of phases of events to npz-format data
    events = []
    for phase, phase_events in events_raw.items():
        for tid, trial_events in enumerate(phase_events):
            trial_events = [{**d, "trial": tid, } for d in trial_events]
            events += trial_events
            
    df = pd.DataFrame(events)
    
    # sort by trial id and time
    df = df.sort_values(by=["trial", "tstart"])
    
    # convert to dictionary
    dicts = df.to_dict(orient="list")
    
    return dicts

def dicts_to_stroke_events(dicts):
    # convert them back to events
    df = pd.DataFrame({key: dicts[key] for key in dicts})
    events = df.to_dict(orient="records")
    return events

def convert_cleaned_to_table(events, N_trials, t_total, mag_normalize=True):
    subject_result = np.zeros((N_trials, t_total, 2)) # angle, mag
    
    for stroke in events:
        trial_id, tstart, tend = stroke['trial'], stroke['tstart'], stroke['tend']
        subject_result[trial_id, tstart:tend+1, 0] = stroke['angle']
        if mag_normalize:
            subject_result[trial_id, tstart:tend+1, 1] = stroke['mag'] / (tend-tstart+1)
        else:
            subject_result[trial_id, tstart:tend+1, 1] = 1
                
    return subject_result 

### generate motion vec
def angle_diff(x1, x2, epoch=180):
    d1 = x1 - x2
    d2 = - np.sign(d1) * (epoch - np.abs(d1))
    ds = np.array([d1, d2])
    mask = np.abs(d1) < np.abs(d2)
    d = mask * d1 + (~mask) * d2
    # finally, convert 90 to -90
    half_epoch = epoch // 2
    mask_90 = d == half_epoch
    d = mask_90 * (-half_epoch) + (~mask_90) * d
    return d

def create_motion_angle_vec(angles, mags, targets, nbins, timestep):
    N_trials, N_time = angles.shape
    
    # fold along 180
    valid_mask = mags > 0
    angles = angles % 180
    
    # diff from target
    bins = np.linspace(-90, 90, nbins+1)
    target_diffs = angle_diff(angles, targets[:, None])
    bin_ids = np.digitize(target_diffs, bins) - 1
    
    # also assign the time bin ids
    time_bin_ids = np.arange(N_time) // timestep
    time_bin_ids = np.tile(time_bin_ids, (N_trials, 1))
    
    # convert these vectors
    N_time_bins = N_time // timestep + 1
    result_table = np.zeros((N_time_bins, nbins))
    np.add.at(result_table, (time_bin_ids.flatten(), bin_ids.flatten()), mags.flatten())
    
    # get time_points
    time_bins = (np.arange(N_time_bins) + 0.5) * timestep
    
    return result_table, time_bins, bins

def compute_angle_profile(
            subj_source_data, behav_df,
            target_name, lmb,
            n_angle_bins=9, timestep=250,
            mag_normalize=True, vec_normalize=False):
    # load behavior data
    N_trials = len(behav_df)
    subj_events = dicts_to_stroke_events(subj_source_data)
    subj_motions = convert_cleaned_to_table(
        subj_events, N_trials, T_TOTAL, mag_normalize=mag_normalize)
    
    # get the magnitude to cap at
    mags = subj_motions[..., 1]
    # get subject magnitude before any filteirng or masking
    subj_mag_max = np.quantile(mags[mags>0], 0.95)
    
    # masking
    if lmb is not None:
        mask = lmb(behav_df)
        behav_df = behav_df[mask]
        subj_motions = subj_motions[mask]

    # fetch the target
    subj_target = behav_df[target_name].to_numpy()
    valid_mask = ~np.isnan(subj_target)
    subj_target = subj_target[valid_mask]
    # print(np.min(subj_target), np.max(subj_target))
    subj_motions = subj_motions[valid_mask]
    subj_angle = subj_motions[..., 0]
    subj_mag = subj_motions[..., 1]
    
    # normalization
    subj_mag = subj_mag / subj_mag_max
    # remove those of too large mag (probably outliers)
    subj_mag[subj_mag>1] = 0    
    
    # aggregate
    angle_profile, time_bins, angle_bins = create_motion_angle_vec(
        angles=subj_angle, mags=subj_mag, targets=subj_target, 
        nbins=n_angle_bins, timestep=timestep)
    
    # further normalize (if needed)
    if vec_normalize:
        angle_profile += 1e-4
        angle_profile = angle_profile / np.sum(angle_profile, axis=1, keepdims=True)
    
    return angle_profile, time_bins, angle_bins


### compute the magnitude
def compute_subj_motion_mags(subj_motion_data, subj_df, lmb, mag_normalize=True):    
    N_trials = len(subj_df)
    subj_events = dicts_to_stroke_events(subj_motion_data)
    subj_motions = convert_cleaned_to_table(
        subj_events, N_trials, T_TOTAL, mag_normalize=mag_normalize)
    subj_mag = subj_motions[..., 1]
            
    # normalization
    subj_mag_max = np.quantile(subj_mag[subj_mag>0], 0.95)
    subj_mag = subj_mag / subj_mag_max
            
    # masking
    if lmb is not None:
        mask = lmb(subj_df)
        subj_mag = subj_mag[mask]
            
    subj_mag_aggregated = np.mean(subj_mag, axis=0)

    return subj_mag_aggregated


### compute the target relevance using the profile
from scipy.stats import entropy

def compute_target_relevance(distrib, time_bins, angle_bins):
    distrib = distrib / np.sum(distrib, axis=-1, keepdims=True)
    target_pos = np.argmin(np.abs(angle_bins))
    target_distrib = np.zeros(distrib.shape[-1]) + 1e-4 # offset
    target_distrib[target_pos] = 1
    target_distrib = target_distrib / np.sum(target_distrib)
    kl_divergences = np.apply_along_axis(lambda p: entropy(p, target_distrib), axis=-1, arr=distrib)

    # now convert the kl divergences to relevence
    uniform = np.ones_like(target_distrib)
    uniform = uniform / np.sum(uniform)
    baseline_div = entropy(uniform, target_distrib)
    best_div = 0
    relevance = (baseline_div - kl_divergences) / baseline_div

    return relevance, time_bins
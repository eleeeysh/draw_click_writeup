import numpy as np
from .eye_data import TrialData, XYData
from .eye_preprocess import convert_movement_to_angle

""" Helper function for computing subject-wiese stats and apply normalization """
def subjectwise_stats(xy_data, base_df, func, phases):
    # group data by subject
    subject_grouped = xy_data.groupby(base_df, by='participant', phases=phases)
    subj_results = {}
    for subj_data in subject_grouped:
        subject, subj_trialids, subj_loaded = subj_data
        # apply func on the subject data
        subject_stats = func(subj_loaded)
        # record data
        subj_results[subject] = subj_trialids, subject_stats
    return subj_results

def find_median(xy_data):
    xs, ys = xy_data
    x_median = np.median(xs)
    y_median = np.median(ys)
    return np.array([x_median, y_median])

def apply_subjectwise_func(trial_data: TrialData, subj_stats, func):
    original_data = trial_data.read()
    
    if trial_data.__class__ is XYData:
        original_data = np.stack([
            original_data[0], original_data[1]],
            axis=-1)
    
    original_tids = trial_data.trial_ids[:]
    tid_subj_mapping = {}
    for subj, (tids, _) in subj_stats.items():
        for tid in tids:
            tid_subj_mapping[tid] = subj
    
    results = []
    for d, tid in zip(original_data, original_tids):
        subj = tid_subj_mapping[tid]
        _, stats = subj_stats[subj]
        normalized = func(d, stats)
        results.append(normalized)
    results = np.array(results)
    
    new_data = None
    if trial_data.__class__ is XYData:
        new_data = XYData(results[..., 0], results[..., 1], original_tids)
    else:
        new_data = TrialData(results, original_tids)
    return new_data

    
""" Helper function for computing stats """
from .eye_stats import collapse_event_data, generate_angle_distrib, compute_vecmap

def extract_stats_with_name_params(xs, ys, func_name, params):
    # we assume all xs and ys are to be collapsed
    xs, ys = collapse_event_data(xs, ys)
    if func_name == 'angle_distrib':
        # generate a distribution of directions
        xs, ys = collapse_event_data(xs, ys)
        angle, mag = convert_movement_to_angle(xs, ys,
            compute_mag=True, stim_align=True)
        angle_distrib = generate_angle_distrib(
            angle, mag, 
            **params)
        return angle_distrib
    
    elif func_name == 'vecmap':
        # generate a normalizaed map of movement
        vec_map = compute_vecmap(xs, ys, **params)
        return vec_map
    
    else:
        raise ValueError(f"Unknown function name: {func_name}")
    
""" PIPELINE for feature extraction """
def filter_data_by_magnitude(source_data, min_mag_thresh, max_mag_thresh):
    # extract angles and magnitudes
    xs, ys = source_data.read()
    mask = (xs != 0) | (ys != 0) 
    rids, cids = np.where(mask)
    _, mags = convert_movement_to_angle(
        xs[rids, cids], ys[rids, cids], compute_mag=True)
    
    # create a map of angle and magnitude
    mag_map = np.zeros_like(xs)
    mag_map[rids, cids] = mags
    mag_mask = (mag_map >= min_mag_thresh) & (mag_map <= max_mag_thresh)
        
    # set all filtered to 0
    xs[~mag_mask] = 0
    ys[~mag_mask] = 0
    filtered = XYData(xs, ys, source_data.trial_ids[:])
    return filtered

def normalize_data(source_data, base_df, normalize_phases):
    # compute median position
    median_pos = subjectwise_stats(
        source_data, base_df, func=find_median, phases=normalize_phases)
    # apply normalization
    lmb_zero_center = lambda d, stats: (d-stats)
    normalized_data = apply_subjectwise_func( 
        source_data, median_pos, lmb_zero_center)
    return normalized_data

""" convert stats of distribution of angle (mag-weighted) """
DEFAULT_SACC_DIR_VEC_SETTINGS = {
    'occurence': {
        'n_angle_bins':360,
        'n_mag_bins': 10,
        'min_mag_thresh': 15,
        'max_mag_thresh': 150,
        'log_transform': False,
    }, # how to filter event data
}

from .eye_stats import generate_mag_and_angle_distrib
from scipy.ndimage import gaussian_filter1d

class SaccadeAngleStats:
    def __init__(self, settings):
        self.settings = settings

    def mag2weight(self, mag, method='log', **params):
        if method == 'identity':
            return mag
        elif method == 'log':
            offset = params.get('offset', 1)
            return np.log(mag+offset)
        else:
            raise ValueError(f'Unknown transformation: {method}')

    """ subject-wise processing """
    def convert_subject_occurence_to_angle_weight(self, xs, ys):
        """
            To convert EACH data to a vector
            We need to compute 
            - its position (mapping to 0-180 degrees)
            - magnitude
            - bias to be removed
            
        """
        settings = self.settings.get('occurence', DEFAULT_SACC_DIR_VEC_SETTINGS['occurence'])
        all_angles, all_mags = convert_movement_to_angle(xs, ys, compute_mag=True)
        
        # first, compute the bias map (without smoothing)
        subject_bias_H, angle_bins, mag_bins =  generate_mag_and_angle_distrib(
            angle=all_angles, mag=all_mags, 
            n_angle_bins=settings['n_angle_bins'],
            n_mag_bins=settings['n_mag_bins'], 
            min_mag_thresh=settings['min_mag_thresh'], 
            max_mag_thresh=settings['max_mag_thresh'], 
            log_transform=settings['log_transform'])    
        
        # - normalize the map
        subject_bias_H = subject_bias_H / np.sum(subject_bias_H)
        
        # compute the magnitude amplitude
        bin_mags = np.arange(len(mag_bins)-1)+0.5
        bin_mag_weights = self.mag2weight(bin_mags)
        
        # compute subject weighted bias
        subject_bias_vec = bin_mag_weights @ subject_bias_H
        
        # next, for each data point
        # we compute the angle bin it should belong to
        # and it's corresponding weight
        all_angle_ids = np.digitize(all_angles, angle_bins) - 1 # left aligned
        all_mag_ids = np.digitize(all_mags, mag_bins) - 1
        # valid_mag_mask = all_angle_ids >= 0
        valid_mag_mask = all_mag_ids >= 0

        all_mag_binned = all_mag_ids + 0.5
        all_mag_binned[~valid_mag_mask] = 1 # temporary
        all_mag_weights = self.mag2weight(all_mag_binned)
        all_mag_weights[~valid_mag_mask] = 0 # invalid mag --> 0 weight
        return subject_bias_vec, all_angle_ids, all_mag_weights
    
    """ convert the data obtained using function above to feature vec """
    def fold_angles(self, angles):
        angles = angles % 180
        return angles

    def transform_vec_epoch(self, vec):
        if vec.shape[-1] == 360:
            vecs = np.split(vec, 2, axis=-1)
            vec = np.sum(vecs, axis=0)
        elif vec.shape[-1] != 180:
            print(f'Unknown vec shape {vec.shape[-1]}')
        return vec

    def align_angle_with(self, vecs, standard_centers, align_dirs, stim_epoch=180):
        epoch = vecs.shape[-1]
        stim_center_id = stim_epoch // 2
        shift_factor = epoch // stim_epoch
        # align_dirs: the standard 'positive' directions
        rows, cols = vecs.shape
        shift = ((stim_center_id - standard_centers) % stim_epoch) * shift_factor
        indices = (np.arange(cols)[None, :] - shift[:, None]) % cols
        vecs = vecs[np.arange(rows)[:, None], indices]
        # flipped those whose aligh dirs are positive
        if align_dirs is not None:
            to_flip_mask = align_dirs < 0
            colids = np.arange(vecs.shape[-1])
            colids_mapto = (epoch - colids) % epoch 
            flipped = np.copy(vecs)
            flipped[..., colids_mapto] = vecs
            combined = to_flip_mask[:, None] * flipped + (1 - to_flip_mask)[:, None] * vecs
        else:
            combined = vecs
        return combined

    def batch_to_angle_vecs_collapse(self, subject_bias_vec, angle_ids, mag_weights, filter_zero, epoc):
        angle_ids = angle_ids.flatten().astype(int)
        mag_weights = mag_weights.flatten()
        if filter_zero:
            mask = mag_weights > 0
            angle_ids = angle_ids[mask]
            mag_weights = mag_weights[mask]
        # sum up
        N = len(angle_ids)
        expanded = np.zeros(epoc)
        np.add.at(expanded, angle_ids, mag_weights)
        expanded -= subject_bias_vec * N
        return expanded

    def batch_to_angle_vecs_trialwise(self, subject_bias_vec, angle_ids, mag_weights, filter_zero, epoc):
        results = []
        tids = range(len(angle_ids))
        for tid in tids:
            trial_angle = angle_ids[tid]
            trial_mag = mag_weights[tid]
            trial_expanded = self.batch_to_angle_vecs_collapse(
                subject_bias_vec, 
                trial_angle, trial_mag, filter_zero, epoc)
            
            results.append(trial_expanded)
        return np.array(results)
    
    def postprocess_angle_vec(self, vec, sigma=15, normalize=True):
        # smoothing
        vec = gaussian_filter1d(vec, sigma=sigma, mode='wrap', axis=-1, radius=15)
        # normalize
        if normalize:
            vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec
    
    def plot_sacc_vec(self, ax, vec, sigma=15, xtick=True):
        vec = self.postprocess_angle_vec(vec, sigma=sigma)
        epoch = vec.shape[-1]
        ax.plot(vec)
        ax.axhline(0, linestyle='--', color='gray')
        ax.axvline(epoch//2, linestyle='--', color='yellow')
        ax.set_ylim([-0.25, 0.25])
        if xtick:
            ax.set_xticks(np.array([30, 90, 155]) * (epoch // 180))
            ax.set_xticklabels(['opposite dir', 'unbiased', 'same dir'])
        return ax, vec
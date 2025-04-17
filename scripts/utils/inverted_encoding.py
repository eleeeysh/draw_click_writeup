import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

""" Functions to compute the distance """
from .distrib import color_smart_diff, color_smart_diff_outer
from scipy.spatial.distance import cdist

def deg_signed_diff(d1, epoch=180):
    d2 = - np.sign(d1) * (epoch - np.abs(d1))
    mask = np.abs(d1) < np.abs(d2)
    d = mask * d1 + (~mask) * d2
    # finally, convert 90 to -90
    half_epoch = epoch // 2
    mask_90 = d == half_epoch
    d = mask_90 * (-half_epoch) + (~mask_90) * d
    return d

class DistFunctions:
    """ define all distance function """
    @classmethod
    def diff(cls, x1, x2, dist_name, pairwise):
        if dist_name == 'cos':
            return cls.cos_diff(x1, x2, pairwise)
        elif dist_name == 'rad':
            return cls.rad_diff(x1, x2, pairwise)
        elif dist_name == 'deg':
            return cls.deg_diff(x1, x2, pairwise)
        elif dist_name == 'euc':
            return cls.euclidean_diff(x1, x2, pairwise)
        else:
            raise NotImplementedError(f'Unknown distance {dist_name}')

    @classmethod
    def cos_diff(cls, x1, x2, pairwise):
        x1 = x1 / np.linalg.norm(x1, axis=-1, keepdims=True)
        x2 = x2 / np.linalg.norm(x2, axis=-1, keepdims=True)
        if pairwise:
            dists = cdist(x1, x2, metric='cosine')
        else:
            similarity = np.sum(x1 * x2, axis=-1)
            dists = 1 - similarity
        return dists
    
    @classmethod
    def euclidean_diff(cls, x1, x2, pairwise):
        if pairwise:
            dists = cdist(x1, x2, metric='euclidean')
        else:
            diffs = x1 - x2
            dists = np.linalg.norm(diffs, axis=-1)
        return dists
    
    @classmethod
    def deg_diff(cls, x1, x2, pairwise):
        if pairwise:
            assert len(x2.shape) == 1
            x1 = x1[..., np.newaxis]
            d1 = x1 - x2 # expand last dim
        else:
            d1 = x1 - x2 # elementwise

        # convert dists to degree dists
        deg_diffs = np.abs(deg_signed_diff(d1))
        return deg_diffs
    
    @classmethod
    def rad_diff(cls, x1, x2, pairwise):
        dists = np.abs(cls.deg_diff(x1, x2, pairwise))      
        dists = np.deg2rad(dists)
        return dists
    
""" Functions to downgrade the resolution of the features """
from scipy.ndimage import zoom

class FeatureDowngrade:
    def __init__(self, params):
        self.params = params

    def get_converted_features(self, Xs):
        new_features = []
        # 0d
        if '0d' in self.params:
            d0_features = self.get_0d(Xs)
            new_features.append(d0_features)
        #1d
        if '1d' in self.params:
            d1_features = self.shrink_1d(Xs)
            new_features.append(d1_features)
        #2d
        if '2d' in self.params:
            d2_features = self.shrink_2d(Xs)
            new_features.append(d2_features)
        new_features = np.concatenate(new_features, axis=-1)
        return new_features

    def get_0d(self, Xs):
        d0_params = self.params['0d']
        d0_features = Xs[..., d0_params['old']]
        return d0_features

    def shrink_1d(self, Xs):
        d1_params = self.params['1d']
        d1_old = Xs[..., d1_params['old']]
        d1_new = d1_old.copy()
        d1_shrink_ratio = d1_params.get('zoom_ratio', None)
        if d1_shrink_ratio is None:
            return d1_new
        else:
            d1_shrink_ratio_list = np.ones(len(d1_new.shape))
            d1_shrink_ratio_list[-1] = d1_shrink_ratio
            d1_new = zoom(d1_new, 
                zoom=d1_shrink_ratio_list, mode='wrap', order=1)
            return d1_new

    def shrink_2d(self, Xs):
        d2_params = self.params['2d']
        d2_old = Xs[..., d2_params['old']]
        d2_new = d2_old.copy()
        d2_shrink_ratio = d2_params.get('zoom_ratio', None)
        if d2_shrink_ratio is None:
            return d2_new
        else:
            # reformat d2
            H, W = d2_params['H'], d2_params['W']
            original_shape = d2_new.shape
            new_shape = (*original_shape[:-1], H, W)
            d2_new = d2_new.reshape(new_shape)

            # zoom
            d2_shrink_ratio_list = np.ones(len(d2_new.shape))
            d2_shrink_ratio_list[-2:] = d2_shrink_ratio
            d2_new = zoom(d2_new, zoom=d2_shrink_ratio_list, mode='wrap', order=1)

            # flatten
            new_shape = (*original_shape[:-1], d2_new.shape[-2] * d2_new.shape[-1])
            d2_new = d2_new.reshape(new_shape)

            return d2_new

""" The inverted encoding model """
class ForwardModel:
    def __init__(self, feature_conversion_params, n_channels):
        self.feature_conversion = FeatureDowngrade(feature_conversion_params)
        # create channels and channel centers
        self.n_channels = n_channels
        self.channel_bins = np.linspace(0, 180, n_channels+1)
        self.channel_cenetrs = (self.channel_bins[1:] + self.channel_bins[:-1]) / 2

    def raw_ys_to_channel_weights(self, ys, sharpness):
        # convert first to radian differece
        rad_diffs = DistFunctions.diff(
            ys, self.channel_cenetrs, 
            dist_name='rad', pairwise=True)
        # rad diffs to weights
        weights = np.exp(sharpness * np.cos(rad_diffs))
        # normalize
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        return weights

    def convert_xinputs(self, Xs):
        Xs = self.feature_conversion.get_converted_features(Xs)
        return Xs

    def get_channel_patterns(self, Xs, ys, item_weights, sharpness):
        # convert xs
        Xs = self.convert_xinputs(Xs)
        # get the channel weights
        ys = self.raw_ys_to_channel_weights(ys, sharpness)
        # get the total weights
        ys = item_weights[..., None] * ys
        ys = np.sum(ys, axis=-2) # across items
        ys = ys / np.sum(ys, axis=-1, keepdims=True) # normalize
        # least square
        C_hat = np.linalg.pinv(ys.T @ ys) @ ys.T @ Xs
        return Xs, ys, C_hat
        
    def find_best_y_transform(self, Xs, ys, item_weights, sharpness_range, x_dist_func, x_thresh):
        # find the best y transform
        best_sharpness = None
        best_loss = np.inf
        for sharpness in sharpness_range:
            Xs_test, ys_test, C_hat_test = self.get_channel_patterns(
                Xs, ys, item_weights, sharpness)
            Xs_pred = ys_test @ C_hat_test
            # first check if the range is ok
            x_outlier_ratio = np.mean((Xs_pred > x_thresh) | (Xs_pred < -x_thresh))
            loss = DistFunctions.diff(
                Xs_pred, Xs_test, x_dist_func, pairwise=False)
            loss = np.mean(loss)
            print(f'sharpness {sharpness:.2f} loss {loss:.6f} (invalid: {x_outlier_ratio:.2f})')
            if loss < best_loss:
                best_loss = loss
                best_sharpness = sharpness

        return best_sharpness, best_loss
    
    def predict(self, Xs, C_hat):
        Xs = self.convert_xinputs(Xs)
        ys = Xs @ C_hat.T @ np.linalg.pinv(C_hat @ C_hat.T)
        return Xs, ys
    
""" Functions to reformat loaded """
def raw_reformat_all_loaded(load_func, subjs, phase, stim_names, cond_lmb):
    """
        load_func: function to load the data (take subject and phase)
    """
    all_xs, all_ys, all_dfs, all_tags = [], [], [], []
    for subj in subjs:
        features, behavior_df = load_func(subj, phase)
        mask = cond_lmb(behavior_df) if cond_lmb is not None else np.ones(len(behavior_df), dtype=bool)
        features = features[mask]
        behavior_df = behavior_df[mask]
        behavior_ys = behavior_df[stim_names].values
        all_xs.append(features)
        all_ys.append(behavior_ys)
        all_dfs.append(behavior_df)
        all_tags.append([subj] * len(features))
    # compute the default weights
    all_xs = np.concatenate(all_xs, axis=0)
    all_ys = np.concatenate(all_ys, axis=0)
    all_tags = np.concatenate(all_tags, axis=0)
    all_dfs = pd.concat(all_dfs, axis=0, ignore_index=True)
    return all_xs, all_ys, all_dfs, all_tags

def convert_df_to_delay_design_matrix(df):
    # item x to remember: stim_x_to_report or trial_code is 1
    design_matrix = np.zeros((len(df), 2))
    design_matrix[:, 0] = (df['stim_1_to_report'] | (df['trial_code'] == 1)).to_numpy().astype(int)
    design_matrix[:, 1] = (df['stim_2_to_report'] | (df['trial_code'] == 1)).to_numpy().astype(int)
    design_matrix = design_matrix / np.sum(design_matrix, axis=-1, keepdims=True)
    return design_matrix

def convert_df_to_isi_design_matrix(df):
    # item x to remember: stim_1_to_report or trial_code is 1
    design_matrix = np.zeros((len(df), 2))
    design_matrix[:, 0] = (df['stim_1_to_report'] | (df['trial_code'] == 1)).to_numpy().astype(int)
    design_matrix[:, 1] = 0
    return design_matrix

""" Functions for training and testing """
from tqdm import tqdm

def raw_across_subj_cross_phase_iterator(
        data_load_reformat_func,
        phase1, phase2, 
        phase1_stim_types, phase2_stim_types,
        phase1_lmb, phase2_lmb, item_weights_lmb,
        kfold, use_tqdm=True):
    # load all data for all subjects
    xs1, ys1, df1, tags1 = data_load_reformat_func(
        phase1, phase1_stim_types, phase1_lmb)
    xs2, ys2, df2, tags2 = data_load_reformat_func(
        phase2, phase2_stim_types, phase2_lmb)

    # cv
    all_subj_ids = list(set(tags1))
    np.random.shuffle(all_subj_ids)
    heldout_subj_ids = np.array_split(all_subj_ids, kfold)
    heldout_subj_ids = [set(subj_ids) for subj_ids in heldout_subj_ids]
    fold_iterator = tqdm(range(kfold)) if use_tqdm else range(kfold)
    for i in fold_iterator:
        # train and test mask
        train_heldout_mask = np.array(
            [s in heldout_subj_ids[i] for s in tags1])
        train_mask = ~train_heldout_mask
        train_subj_ids = set(tags1[train_mask])
        test_heldout_mask = np.array(
            [s in train_subj_ids for s in tags2])
        test_mask = ~test_heldout_mask
        # fetch data
        fold_train_xs, fold_train_ys = xs1[train_mask], ys1[train_mask]
        fold_test_xs, fold_test_ys = xs2[test_mask], ys2[test_mask]
        fold_train_df, fold_test_df = df1[train_mask], df2[test_mask]
        # yield data
        yield {
            'xs1': fold_train_xs,
            'ys1': fold_train_ys,
            'df1': fold_train_df,
            'xs2': fold_test_xs,
            'ys2': fold_test_ys,
            'df2': fold_test_df,
            'item_weights': item_weights_lmb(fold_train_df),
        }

def raw_within_subj_cross_phase_iterator(
        data_load_reformat_func,
        phase1, phase2, 
        phase1_stim_types, phase2_stim_types,
        phase1_lmb, phase2_lmb, item_weights_lmb,
        kfold, use_tqdm=True):
    
    # load all data for all subjects
    xs1, ys1, df1, tags1 = data_load_reformat_func(
        phase1, phase1_stim_types, phase1_lmb)
    xs2, ys2, df2, tags2 = data_load_reformat_func(
        phase2, phase2_stim_types, phase2_lmb)

    # get subjects    
    phase1_unique_subjects = set(tags1)
    phase2_unique_subjects = set(tags2)
    valid_subjects = phase1_unique_subjects.intersection(phase2_unique_subjects)
    if len(valid_subjects) == 0:
        raise ValueError('No common subjects between the two phases.')
    
    subj_iterator = tqdm(valid_subjects) if use_tqdm else valid_subjects
    for subj in subj_iterator:
        train_subj_mask = np.array([s == subj for s in tags1])
        train_trial_ids = df1[train_subj_mask]['TRIALID'].values
        test_subj_mask = np.array([s == subj for s in tags2])
        test_trial_ids = df2[test_subj_mask]['TRIALID'].values
        heldout_ids = np.array_split(np.arange(len(train_trial_ids)), kfold)
        for i in range(kfold):
            heldout_mask = np.zeros(len(train_trial_ids), dtype=bool)
            heldout_mask[heldout_ids[i]] = True
            # trials not in heldout are used for training
            selected_train_trial_ids = train_trial_ids[~heldout_mask]
            selected_train_mask = ~heldout_mask
            # trials not used for training are all used for testing
            selected_train_set = set(selected_train_trial_ids)
            selected_test_trial_ids = np.array([
                (s not in selected_train_set) for s in test_trial_ids])
            # fetch the data
            fold_train_xs, fold_train_ys = xs1[train_subj_mask][selected_train_mask], ys1[train_subj_mask][selected_train_mask]
            fold_test_xs, fold_test_ys = xs2[test_subj_mask][selected_test_trial_ids], ys2[test_subj_mask][selected_test_trial_ids]
            fold_train_df, fold_test_df = df1[train_subj_mask][selected_train_mask], df2[test_subj_mask][selected_test_trial_ids]
            # yield data
            yield {
                'xs1': fold_train_xs,
                'ys1': fold_train_ys,
                'df1': fold_train_df,
                'xs2': fold_test_xs,
                'ys2': fold_test_ys,
                'df2': fold_test_df,
                'item_weights': item_weights_lmb(fold_train_df),
            }

def train_test_invert_encoding(
        model_params, xs1, xs2, ys1, ys2, df1, df2, item_weights):
    # get the weight of items
    """
    if item_weights is None:
        item_weights = ~(np.isnan(ys1))
        item_weights = item_weights / np.sum(
            item_weights, axis=-1, keepdims=True)
    """
        
    model = ForwardModel(**model_params['init'])
    sharpness = model_params['forward']['sharpness']
    # get pattern
    _, _, patterns = model.get_channel_patterns(
        xs1, ys1, item_weights, sharpness)
    # predict
    xs2_trans, ys_pred = model.predict(xs2, patterns)
    # also get the converted ys2
    ys2_trans = model.raw_ys_to_channel_weights(ys2, sharpness)
    # collect results
    results = {
        'list': {
            # 'train_xs': xs1,
            # 'train_xs_converted': xs1_trans,
            # 'train_ys': ys1,
            # 'train_ys_converted': ys1_trans,
            'test_xs': xs2,
            'test_xs_converted': xs2_trans,
            'test_ys': ys2,
            'test_ys_converted': ys2_trans,
            'preds': ys_pred,
        },
        'df': {
            # 'train_df': df1,
            'test_df': df2,
        },
        'np': {
            'pattern': patterns,
        }
        
    }
    return results

from tqdm import tqdm

def raw_cv_train_test_invert_encoding(
        iterator_func,
        model_params, phase1, phase2, 
        phase1_stim_types, phase2_stim_types,
        phase1_lmb, phase2_lmb,
        item_weights_lmb, kfold, use_tqdm=True):
    results = []
    progress_bar = tqdm(total=kfold, desc="Processing") if use_tqdm else None
    for data in iterator_func(
            phase1, phase2, phase1_stim_types, phase2_stim_types,
            phase1_lmb, phase2_lmb, item_weights_lmb, kfold, use_tqdm=use_tqdm):
        result = train_test_invert_encoding(model_params, **data)
        results.append(result)
        if use_tqdm:
            progress_bar.update(1)

    collected = {}
    for k in results[0]['list']:
        collected[k] = np.concatenate(
            [r['list'][k] for r in results], axis=0)
    for k in results[0]['df']:
        collected[k] = pd.concat(
            [r['df'][k] for r in results], ignore_index=True)
    for k in results[0]['np']:
        collected[k] = [r['np'][k] for r in results]
        collected[k] = np.mean(collected[k], axis=0)
    return collected

""" Functions to convert predictions """
def raw_channel_weights_to_pseudo_distrib(
        channel_weights, model, sharpness,
        delay_channel_pred_mean,
        delay_channel_pred_std):
    default_degs = np.arange(180)
    projection = model.raw_ys_to_channel_weights(
        default_degs, sharpness)
    
    # pos_thresh, neg_thresh = 6, -4
    # channel_weights = np.clip(channel_weights, neg_thresh, pos_thresh)
    
    channel_weights = (channel_weights - delay_channel_pred_mean) / delay_channel_pred_std
    channel_weights = np.clip(channel_weights, -4, 4)
    
    channel_weights = channel_weights - np.min(channel_weights, axis=-1, keepdims=True) + 1e-5
    pdistrib = channel_weights @ projection.T

    # baseline_vec = projection / np.linalg.norm(projection, axis=-1, keepdims=True)
    # dist_to_baseline = DistFunctions.cos_diff(channel_weights, baseline_vec, pairwise=True)
    # sim_to_baseline = 2 - dist_to_baseline # scale to 0-2
    # pdistrib = sim_to_baseline / np.sum(sim_to_baseline, axis=-1, keepdims=True)

    # now make them a distribution
    distrib = pdistrib / np.sum(pdistrib, axis=-1, keepdims=True)

    return distrib

def shift_align_distrib(distrib, targets, refs=None):
    # filter out nan
    valid_mask = ~(np.isnan(targets))
    default_results = np.zeros_like(distrib)
    if refs is not None:
        valid_mask = valid_mask & ~(np.isnan(refs))
    distrib = distrib[valid_mask]
    targets = targets[valid_mask]
    if refs is not None:
        refs = refs[valid_mask]
    default_results[~valid_mask] = np.nan

    targets = targets.astype(int) % 180

    # determine the new x for each distrib
    default_degs = np.arange(180)
    default_diffs = deg_signed_diff(
        np.subtract.outer(targets, default_degs))
    relative_xs = - default_diffs
    if refs is not None:
        sd_diffs = deg_signed_diff(refs-targets)
        flip_mask = sd_diffs < 0
        relative_xs[flip_mask] = - relative_xs[flip_mask]
    relative_xs = (np.round(relative_xs) % 180).astype(int)
    # shift it: error = 0, 1,... 89, 90, -89,...-1
    shifted = np.zeros_like(distrib)
    row_indices = np.arange(len(distrib))[:, np.newaxis]
    col_indices = np.arange(180)
    shifted[row_indices, relative_xs] = distrib[
        row_indices, col_indices]
    default_results[valid_mask] = shifted

    return default_results, valid_mask

def raw_display_shifted_distrib(
        ax, distrib, mask=None, label=None, ref_type=None,
        ylim_min=0.0045, ylim_max=0.0065):
    if mask is not None:
        distrib = distrib[mask]
    
    # sort from -180 to 180 (0 for target)
    summed = np.mean(distrib, axis=0)
    summed_xs = deg_signed_diff(np.arange(180))
    summed_idx = np.argsort(summed_xs)
    summed = summed[summed_idx]
    summed_xs = summed_xs[summed_idx]

    ax.plot(summed_xs, summed, label=label)
    ax.axvline(0, color='red', linestyle='--')
    if ref_type == 'previous':
        ax.set_xticks([-45, 45])
        ax.set_xticklabels([
            '<<away from previous resp<<', 
            '>>towards previous resp>>'], fontsize=14)
        ax.set_xlabel('error (corrected)', fontsize=14)
    elif ref_type == 'nontarget':
        ax.set_xticks([-45, 45])
        ax.set_xticklabels([
            '<<away from non-target<<', 
            '>>towards non-target>>'], fontsize=14)
        ax.set_xlabel('error (corrected)', fontsize=14)
    elif ref_type == 'target':
        ax.set_xlabel('error', fontsize=14)
    else:
        raise  NotImplementedError(f'Unknown ref_type {ref_type}')
    ax.set_ylabel('probability', fontsize=14)
    ax.axhline(1/180, color='red', linestyle='--')
    ax.set_ylim([ylim_min, ylim_max])


""" More for quantifying the fitted results """
def compute_accuracy(distrib, T=180):
    # 0 --> err = 0
    # 0.5 --> max err
    degs = np.linspace(0, 360, T, endpoint=False)
    errs = deg_signed_diff(degs, epoch=360)
    baseline_vec = np.cos(np.deg2rad(errs))

    # baseline_vec = baseline_vec / np.linalg.norm(baseline_vec)
    # distrib_vec = distrib
    # distrib_vec = distrib / np.sum(distrib, axis=-1, keepdims=True)
    distrib_vec = distrib / np.linalg.norm(distrib, axis=-1, keepdims=True)
    acc = np.sum(distrib_vec * baseline_vec, axis=-1)
    return acc

def compute_bias(distrib, T=180, err_thresh=40):
    # 0 --> no bias
    # 0.25 --> most positive bias
    # 0.5 --> neutral?
    # 0.75 --> most negative bias
    degs = np.linspace(0, 360, T, endpoint=False)
    errs = deg_signed_diff(degs, epoch=360)
    baseline_vec = np.sin(np.deg2rad(errs))

    # baseline_vec = baseline_vec / np.linalg.norm(baseline_vec)
    # distrib_vec = distrib
    distrib_vec = distrib / np.linalg.norm(distrib, axis=-1, keepdims=True)

    # error filtering
    total_weights = np.linalg.norm(baseline_vec)
    if err_thresh is None:
        err_thresh = 180
    mask = np.abs(errs) <= (err_thresh * 360 / T)
    distrib_vec = distrib_vec[mask]
    baseline_vec = baseline_vec[mask]
    kept_weights = np.linalg.norm(baseline_vec)

    ratio = total_weights / kept_weights
    bias = np.sum(distrib_vec * baseline_vec, axis=-1) * ratio
    return bias

stat_func_mapping = {
    'accuracy': compute_accuracy,
    'bias': compute_bias,
}

def subjlevel_bias_stats(pred_distrib, df, cond_lmb, stat_type):
    if cond_lmb is not None:
        valid_mask = cond_lmb(df)
        pred_distrib = pred_distrib[valid_mask]
        df = df[valid_mask]

    subjs = df['participant'].unique()
    subjs = [str(s) for s in subjs]
    subj_stats = {}
    for subj in subjs:
        subj_mask = df['participant'] == int(subj)
        subj_bias = np.nan
        if np.sum(subj_mask) > 0:
            subj_distrib = pred_distrib[subj_mask]
            subj_distrib = np.mean(subj_distrib, axis=0)
            subj_bias = stat_func_mapping[stat_type](subj_distrib)
        subj_stats[subj] = subj_bias
    return subj_stats

""" Finally, the most tedious to display the result and to collect stats """
def raw_display_stats_and_distrib(
        ax, # set to None to disable visualization
        results, stats_type, 
        common_lmb=None, condition_lmbs={}, item_weights_lmb=None, 
        sharpness=None, # sharpness to convert to distrib
        return_subj_stats=False, # if to return each subject's stats
        surrogate_model=None, # any inverted encoding model works
        prediction_conversion_func=None, # channel_weights_to_pseudo_distrib
        display_shifted_distrib_func=None, # display_shifted_distrib
    ):
    y_pred_distrib = prediction_conversion_func(
        results['preds'],surrogate_model, sharpness=sharpness)
    y_df = results['test_df']
    ys = results['test_ys']
    if common_lmb is not None:
        # common lmb to select a subset to analyze
        valid_mask = common_lmb(y_df)
        y_df = y_df[valid_mask]
        y_pred_distrib = y_pred_distrib[valid_mask]
        ys = ys[valid_mask]

    # combined should combined with masking
    if item_weights_lmb is None:
        raise ValueError('item_weights_lmb should be specified')
    item_weights = item_weights_lmb(y_df)

    # compute distribs
    y_distribs_results = {}
    for sid, stim_name in enumerate(['stim 1', 'stim 2']):
        # specify the references to compare with
        refs = None
        if stats_type == 'accuracy':
            pass
        elif stats_type == 'sd':
            refs = y_df['prev_last_response'].to_numpy(copy=True)
        elif stats_type == 'sur':
            ref_code = 2 - sid
            refs = y_df[f'stim_{ref_code}'].to_numpy(copy=True)
        else:
            raise NotImplementedError(f'Unknown stats type {stats_type}')

        y_distrib, _ = shift_align_distrib(
            y_pred_distrib, ys[:, sid], refs=refs)
        # remove the irrelevant item bias
        n_irr_id = 1 - sid
        n_has_irr = item_weights[:, n_irr_id] > 0
        irr_w = 1 + n_has_irr
        irr_bias = n_has_irr / 180
        y_distrib = y_distrib * irr_w[:, None] - irr_bias[:, None]
        y_distrib = y_distrib / np.sum(y_distrib, axis=-1, keepdims=True)
        # update
        y_distribs_results[stim_name] = y_distrib

    # further masking if it applies...
    if 'stim 1' in condition_lmbs:
        lmb = condition_lmbs['stim 1']['lmb']
        if lmb is not None:
            stim1_mask = lmb(y_df)
            item_weights[~stim1_mask, 0] = 0
    if 'stim 2' in condition_lmbs:
        lmb = condition_lmbs['stim 2']['lmb']
        if lmb is not None:
            stim2_mask = lmb(y_df)
            item_weights[~stim2_mask, 1] = 0

    item_weights = item_weights.T
    combined_valid_mask = np.sum(item_weights, axis=0) > 0 # at least one of two stims are included
    y_df = y_df[combined_valid_mask]
    y_distribs_results['stim 1'] = y_distribs_results['stim 1'][combined_valid_mask]
    y_distribs_results['stim 2'] = y_distribs_results['stim 2'][combined_valid_mask]
    
    item_weights = item_weights[:, combined_valid_mask]
    item_weights = item_weights / np.sum(item_weights, axis=0, keepdims=True)

    y_combined_distrib = np.array([
        y_distribs_results['stim 1'], 
        y_distribs_results['stim 2']])
    y_combined_distrib = np.sum(
        y_combined_distrib * item_weights[..., None], axis=0)
    y_distribs_results['combined'] = y_combined_distrib
    
    all_stats_vals = {}
    all_subj_stats_vals = {}
    for condname, cond_settings in condition_lmbs.items():
        cond_target = cond_settings['target']
        cond_lmb = cond_settings['lmb']
        cond_mask = None
        if cond_lmb is not None:
            cond_mask = cond_lmb(y_df)
        cond_distrib = y_distribs_results[cond_target]

        # compute stats
        stats_strs = []
        stats_vals = {}
        subj_stats_vals = {}
        # compute accuracy
        all_stats_names = []
        if stats_type == 'accuracy':
            all_stats_names = ['accuracy', 'bias']
        elif stats_type == 'sd':
            all_stats_names = ['bias']
        elif stats_type == 'sur':
            all_stats_names = ['bias']
        else:
            raise NotImplementedError(f'Unknown stats type {stats_type}')

        for stat_name in all_stats_names:
            # get each subject stats
            stats = subjlevel_bias_stats(
                cond_distrib, y_df, cond_lmb, stat_name)
            subj_stats_vals[stat_name] = stats

            # compute mean and sem
            stats = np.array(list(stats.values()))
            stats = stats[~np.isnan(stats)]
            subj_mean = np.mean(stats)
            subj_sem = np.std(stats) / np.sqrt(len(stats))
            stats_strs.append(f'{stat_name}: {subj_mean:.3f}\u00B1{subj_sem:.3f}')
            stats_vals[stat_name] = {
                'mean': subj_mean,
                'sem': subj_sem,
            }
        stats_str = ', '.join(stats_strs)

        # plot the distribution
        if ax is not None:
            ref_type = {
                'accuracy': 'target',
                'sd': 'previous',
                'sur': 'nontarget',
            }[stats_type]
            display_shifted_distrib_func(ax, cond_distrib, 
                mask=cond_mask, label=f'{condname}:: {stats_str}', 
                ref_type=ref_type)
        
        # update stats
        all_subj_stats_vals[condname] = subj_stats_vals
        all_stats_vals[condname] = stats_vals

    if ax is not None:
        ax.legend(
            fontsize=14, loc="lower center", bbox_to_anchor=(0.5, 0.05))

    final_stats_results = (all_stats_vals, all_subj_stats_vals) if return_subj_stats else all_stats_vals
    return final_stats_results



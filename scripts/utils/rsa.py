import numpy as np

""" model for representations """
from .distrib import ValueRangeManager, OriDistribModel
from copy import deepcopy

class RepresentationModel:
    def __init__(self, base_params):
        self.value_manager = ValueRangeManager(np.arange(180))
        # base distrib
        self.base_distrib = OriDistribModel(self.value_manager, base_params)
        # distrib to incorporate SD
        sd_distrib_params = deepcopy(base_params)

    def get_representation(self, data):
        return self.base_distrib.loc(data)
    
from abc import ABC, abstractmethod
from utils.distrib import color_smart_diff, color_smart_diff_outer
from scipy.spatial.distance import cdist

smart_diff = lambda x1, x2: color_smart_diff(x1, x2, vmin=-90, vmax=90)
smart_diff_outer = lambda x1, x2: color_smart_diff_outer(x1, x2, vmin=-90, vmax=90)

class DistFunctions:
    """ define all distance function """
    @classmethod
    def diff(cls, x1, x2, dist_name, pairwise):
        if dist_name == 'cos':
            return cls.cos_diff(x1, x2, pairwise)
        elif dist_name == 'euc':
            return cls.euclidean_diff(x1, x2, pairwise)
        else:
            raise NotImplementedError(f'Unknown distance {dist_name}')

    @classmethod
    def cos_diff(cls, x1, x2, pairwise):
        # normalize
        norm = np.linalg.norm(x1, axis=-1, keepdims=True)
        x1 = np.divide(x1, norm, where=(norm != 0))
        norm = np.linalg.norm(x2, axis=-1, keepdims=True)
        x2 = np.divide(x2, norm, where=(norm != 0))
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

class RSAHelper(ABC):
    @abstractmethod
    def compute_channel_pattern(self, neural_data, *args):
        pass

    @abstractmethod
    def compute_one_subj_corr(self, neural_data, *args):
        # for each trial
        # compute the pattern excluding that trial
        # compute the correlation for that trial over time
        # note that for SD: replace 'nan' previous response with current stim
        pass

STD_THRESH = 1e-8

class RepRSAHelper(RSAHelper):
    def __init__(self, channels, channel_k, rep_model: RepresentationModel):
        self.rep_model = rep_model
        self.channels = np.arange(channels) * (180 // channels)
        self.channel_k = channel_k
        self.channel_resps = self.rep_model.get_representation(self.channels) #N_CHANN * 180
        
    def compute_channel_pattern(self, neural_data, stim_rep): 
        # compute their diff with channels
        # result: N x c
        data_channel_diff = DistFunctions.diff(
            stim_rep, self.channel_resps, dist_name='cos', pairwise=True)

        # channel weights
        data_channel_weights = np.exp((1 - data_channel_diff) * self.channel_k)
        data_channel_weights = data_channel_weights / np.sum(data_channel_weights, axis=1, keepdims=True)

        # compute channel-wise pattern
        channel_patterns = np.tensordot(data_channel_weights.T, neural_data, axes=(1, 0)) # neural data may have >=2 dimensions
        channel_weights = np.sum(data_channel_weights, axis=0)

        # normalize: (by weights)
        weight_safe = np.where(channel_weights == 0, 1, channel_weights)
        shape_expansion = [slice(None)] + [None] * (channel_patterns.ndim - 1)
        expanded_weights = weight_safe[tuple(shape_expansion)]  # Shape: c x ...
        normalized_channel_pattern = channel_patterns / expanded_weights

        return normalized_channel_pattern, channel_weights

    def compute_trial_n(self, neural_data, stim_rep, cur_trial_id, dist_method, valid_channel_w_thresh=0.1):        
        target_mask = np.zeros(len(neural_data)).astype(bool)
        target_mask[cur_trial_id] = True

        # get the target
        target_data = neural_data[target_mask]
        target_stim_rep = stim_rep[target_mask]

        # get the rest
        rest_data = neural_data[~target_mask]
        rest_stim_rep = stim_rep[~target_mask]

        # compute the pattern from the rest
        rest_channel_pattern, rest_channel_weights = self.compute_channel_pattern(rest_data, rest_stim_rep)

        # the diff between channel pattern and target pattern
        neural_diffs = DistFunctions.diff(
            rest_channel_pattern, target_data, 
            dist_name=dist_method, pairwise=False)

        # the diff between channel and target corresponding stims
        stim_diffs = DistFunctions.diff(
            self.channel_resps, target_stim_rep, 
            dist_name='cos', pairwise=False)

        # filter out missing channel?
        w_thresh = valid_channel_w_thresh * 1 / len(self.channels)
        valid_channel_mask = rest_channel_weights > w_thresh
        corr = np.nan
        if np.sum(valid_channel_mask) >= 2:
            # to compute correlation we need at least 2 data
            neural_diffs = neural_diffs[valid_channel_mask]
            stim_diffs = stim_diffs[valid_channel_mask]

            # skip those invalid time steps
            corr = np.zeros(neural_diffs.shape[-1])
            corr_mask = np.std(neural_diffs, axis=0) > STD_THRESH
            masked_corr = np.corrcoef(
                stim_diffs, neural_diffs[:, corr_mask], rowvar=False)[0, 1:]
            corr[corr_mask] = masked_corr
            
        return corr

    def compute_one_subj_corr(self, neural_data, target, dist_method, valid_channel_w_thresh=0.0):
        valid_mask = ~np.isnan(target)
        neural_data = neural_data[valid_mask]
        target = target[valid_mask]

        n_trials = len(neural_data)

        corr_scores = []
        if n_trials <= 2:
            # we need at least 2 trials to compute correlation
            corr_scores = np.full(n_trials, np.nan)
        else:
            target_rep = self.rep_model.get_representation(target)

            for trial_id in range(n_trials):
                # print(trial_id)
                corr = self.compute_trial_n(
                    neural_data, target_rep, trial_id, dist_method,
                    valid_channel_w_thresh=valid_channel_w_thresh)
                corr_scores.append(corr)
            
            # compute the average
            corr_scores = np.array(corr_scores)

        return corr_scores

""" for conditional RSA """
ALL_TIME_STEPS = np.arange(200)

def raw_conditional_rsa_subj(
        subj, lmb, feature_mask, y_name, feature_dist_method,
        load_subj_feature_func, rsa_helper, time_steps, window_size=1):
    # masking
    fetched_features = []
    ys = None
    window = np.arange(window_size) - window_size // 2
    for t in time_steps:
        step_window = window + t
        features, behav_df = load_subj_feature_func(subj, step_window)
        mask = lmb(behav_df) & ~np.isnan(behav_df[y_name].to_numpy())

        if np.sum(mask) >= 2:
            features = features[mask]
            behav_df = behav_df[mask]

            # get the target
            ys = behav_df[y_name].to_numpy()

            # fetch only relevant features
            features = features[:, feature_mask]
            fetched_features.append(features)

    if len(fetched_features) == 0:
        # there could be empty case...
        return None

    fetched_features = np.stack(fetched_features, axis=1)
    rsa_scores = rsa_helper.compute_one_subj_corr(
        fetched_features, ys, feature_dist_method)
    avg_scores = np.nanmean(rsa_scores, axis=0)

    return avg_scores

""" for display """
from tqdm import tqdm
from collections import OrderedDict
from utils.eye_plotting import annotate_time_line
from utils.eye_trial import generate_events

EVENTS = generate_events()

def raw_get_everyone_corr(lmb, feature_mask, y_name, feature_dist_method,
        compute_subj_rsa_func, subjs):
    all_subj_corr = []
    for subj in tqdm(subjs):
    # for subj in subjs:
        subj_data = compute_subj_rsa_func(subj, lmb, feature_mask, y_name, feature_dist_method)
        if subj_data is not None:
            all_subj_corr.append(subj_data)
    all_subj_corr = np.array(all_subj_corr)
    return all_subj_corr

def raw_display_conditional_rsa(
        ax, lmb, lmb_name, feature_mask, y_name, feature_dist_method, 
        color=None, alpha=1, linestyle='-',
        get_rsa_corr_func=None,
        display_time_steps=None):
    all_subj_corr = get_rsa_corr_func(lmb, feature_mask, y_name, feature_dist_method)
    mean_corr = np.mean(all_subj_corr, axis=0)
    sem_corr = np.std(all_subj_corr, axis=0) / np.sqrt(len(all_subj_corr))
    actual_time_points = display_time_steps
    ax.plot(
        actual_time_points, mean_corr, 
        label=lmb_name, c=color, alpha=alpha, linestyle=linestyle)
    ax.fill_between(
        actual_time_points, mean_corr-sem_corr, 
        mean_corr+sem_corr, alpha=alpha*0.4, facecolor=color)
    
def raw_display_lmb_dicts_rsa(
        ax, lmb_dicts, feature_mask, y_name, feature_dist_method, 
        colors=None, alphas=None, linestyles=None,
        show_legend=True,
        display_rsa_func=None):
    for lmb_name, lmb in lmb_dicts.items():
        c = None
        if colors is not None:
            c = colors[lmb_name]
        a = 1.0
        if alphas is not None:
            a = alphas[lmb_name]
        ls = '-'
        if linestyles is not None:
            ls = linestyles[lmb_name]
        display_rsa_func(
            ax, lmb, lmb_name, feature_mask, y_name, 
            feature_dist_method, 
            color=c, alpha=a, linestyle=ls)
    
    annotate_time_line(ax, EVENTS)
    last_time_point = EVENTS['response']+500
    ax.set_ylim([-0.15, 0.35])
    ax.set_yticks([-0.1, 0.0, 0.1, 0.2, 0.3])
    ax.hlines(0, last_time_point,0,linestyles='dashed',colors='black')
    if show_legend:
        ax.legend(bbox_to_anchor=(1.2, 1.0))

    # finally, remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

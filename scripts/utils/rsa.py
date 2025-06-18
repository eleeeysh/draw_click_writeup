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
VALID_CHANNEL_W_THRESH = 0.1
MIN_N_VALID_CHANNELS = 5

from scipy.stats import spearmanr

def fast_corr_computation(m1, m2, permute=False):
    n = m1.shape[0]
    if permute:
        perm = np.random.permutation(n)
        m2 = m2[perm][:, perm]  # permute m2 is enough?

    # Mi,j: i's similarity with tiral j
    mask = ~np.eye(n, dtype=bool)
    m1_masked = m1[mask].reshape(n, n - 1)
    m2_masked = m2[mask].reshape(n, n - 1)

    # compute the standard deviations
    std_1 = np.std(m1_masked, axis=1, ddof=1)  # sample std
    std_2 = np.std(m2_masked, axis=1, ddof=1)  # sample std

    corr = np.nan
    if (std_1 > 0).all() and (std_2 > 0).all():
        # valid only both are non zeros
        # compute covariance
        m1_centered = m1_masked - m1_masked.mean(axis=1, keepdims=True)
        m2_centered = m2_masked - m2_masked.mean(axis=1, keepdims=True)

        # Compute rowwise covariance (sample, so divide by n-2)
        cov = np.sum(m1_centered * m2_centered, axis=1) / (n - 2)

        # compute the corr coefficient
        corr = np.mean(cov / (std_1 * std_2))

    return corr

class RepRSAHelper(RSAHelper):
    def __init__(self, channels, channel_k, rep_model: RepresentationModel, use_spearman=False):
        self.rep_model = rep_model
        self.channels = np.arange(channels) * (180 // channels)
        self.channel_k = channel_k
        self.channel_resps = self.rep_model.get_representation(self.channels) #N_CHANN * 180
        self.use_spearman = use_spearman
        
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

    def compute_trial_n(self, neural_data, stim_rep, cur_trial_id, dist_method,
            valid_channel_w_thresh=VALID_CHANNEL_W_THRESH):        
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
        if np.sum(valid_channel_mask) >= MIN_N_VALID_CHANNELS:
            # to compute correlation we need at least 2 data
            neural_diffs = neural_diffs[valid_channel_mask]
            stim_diffs = stim_diffs[valid_channel_mask]

            # skip those invalid time steps
            corr = np.zeros(neural_diffs.shape[-1])
            corr_mask = np.std(neural_diffs, axis=0) > STD_THRESH

            masked_neural_diffs = neural_diffs[:, corr_mask]
            if self.use_spearman:
                # use spearman correlation
                masked_corr = np.array([
                    spearmanr(stim_diffs, t_neural_diffs)[0]
                    for t_neural_diffs in masked_neural_diffs.T])
            else:
                # use pearson correlation
                masked_corr = np.corrcoef(
                    stim_diffs, masked_neural_diffs, rowvar=False)[0, 1:]
            
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

    def compute_one_subj_time_diffs(self, neural_data, target, dist_method):
        """ compute the correlation scores for each trial with permutation """
        valid_mask = ~np.isnan(target)
        neural_data = neural_data[valid_mask]
        target = target[valid_mask]

        # generate the difference between neural data
        # the diff between channel pattern and target pattern
        neural_diffs = DistFunctions.diff(
            neural_data, neural_data, 
            dist_name=dist_method, pairwise=True)

        # generate the similarity between targets
        # the diff between channel and target corresponding stims
        stim_reps = self.rep_model.get_representation(target)
        stim_diffs = DistFunctions.diff(
            stim_reps, stim_reps, 
            dist_name='cos', pairwise=True)

        return neural_diffs, stim_diffs
    
    def compute_one_subj_corr_trialwise_pert(self, neural_data, target, dist_method, min_trials=3, n_permutations=100):
        """ compute the correlation scores for each trial with permutation """
        # filter out the nan
        valid_mask = ~np.isnan(target)
        neural_data = neural_data[valid_mask]
        target = target[valid_mask]

        n_trials = len(neural_data)
        
        actual_corr = None
        permuted_corrs = np.zeros(n_permutations).astype(float)
        if n_trials < min_trials:
            # we need at least 2 trials to compute correlation
            actual_corr = np.nan
        else:
            neural_diffs, stim_diffs = self.compute_one_subj_time_diffs(
                neural_data, target, dist_method)

            # compute the actual correlation
            actual_corr = fast_corr_computation(
                neural_diffs, stim_diffs, permute=False)
            # compute the permuted correlation
            for i in range(n_permutations):
                permuted_corr = fast_corr_computation(
                    neural_diffs, stim_diffs, permute=True)
                permuted_corrs[i] = permuted_corr

        return actual_corr, permuted_corrs


""" for conditional RSA """
ALL_TIME_STEPS = np.arange(200)

""" old version
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
"""

from tqdm import tqdm
from collections import OrderedDict
from utils.eye_plotting import (
    annotate_time_line,

)
from utils.eye_trial import generate_events
from scipy.stats import ttest_1samp
from mne.stats import permutation_cluster_1samp_test

EVENTS = generate_events()
RSA_PLOT_SIZE = (10, 5)

RSA_PLOT_YMIN = -0.1 # -0.15
RSA_PLOT_YMAX = 0.28 # 0.35
RSA_PLOT_YMIN_HIDDEN = -0.05 # -0.08
RSA_PLOT_YMAX_HIDDEN = 0.23 # 0.3
RSA_PLOT_YTICKS = [0.0, 0.1, 0.2] # [-0.1, 0.0, 0.1, 0.2, 0.3]

PERM_TEST_N = 1000

class ConditionalRSAFullHelper:
    def __init__(self, rsa_helper: RSAHelper, plot_time_steps, plot_window_size, permutation_level=0):
        self.rsa_helper = rsa_helper
        self.plot_time_steps = plot_time_steps # time steps to plot/process
        self.plot_window_size = plot_window_size # gaze: set to 3

        self.permutation_level = permutation_level # 0: grand level, 1: trial-wise, 2: time-wise

    def load_subj_feature(self, *args, **kwargs):
        """ load the subject feature, return features and behav_df """
        raise NotImplementedError("This method should be implemented in subclass")

    """ compute the scores for each subject """
    def conditional_rsa_subj(self,
            subj, lmb, feature_mask, y_name, feature_dist_method):
        # masking
        ys = None
        window = np.arange(self.plot_window_size) - self.plot_window_size // 2
        avg_scores = []

        n_perm = 0 if self.permutation_level == 0 else 100
        permutated_scores = []

        for t in self.plot_time_steps:
            step_window = window + t
            features, behav_df = self.load_subj_feature(subj, step_window)
            mask = lmb(behav_df) & ~np.isnan(behav_df[y_name].to_numpy())

            if np.sum(mask) < 2:
                # not enough trials, skip this subject
                return None, None

            features = features[mask]
            behav_df = behav_df[mask]

            # get the target
            ys = behav_df[y_name].to_numpy()

            # fetch only relevant features
            features = features[:, feature_mask]

            t_corr, t_permuted = self.rsa_helper.compute_one_subj_corr_trialwise_pert(
                features, ys, feature_dist_method, n_permutations=n_perm)
            avg_scores.append(t_corr)
            permutated_scores.append(t_permuted)

        return avg_scores, permutated_scores # actual v.s. permuted

    """ for display """
    def get_everyone_corr(self, 
            lmb, feature_mask, y_name, feature_dist_method, subjs):
        all_subj_corr = []
        all_subj_permute_corr = []
        filtered_subjs = []
        for subj in tqdm(subjs):
        # for subj in subjs:
            subj_corr, subj_corr_permuted = self.conditional_rsa_subj(
                subj, lmb, feature_mask, y_name, feature_dist_method)
            if subj_corr is not None:
                # there are data...
                filtered_subjs.append(subj)
                all_subj_corr.append(subj_corr)
                all_subj_permute_corr.append(subj_corr_permuted)

        if self.permutation_level > 0:
            # do something with the trial-wise permutation
            raise NotImplementedError

        all_subj_corr = np.array(all_subj_corr)
        return all_subj_corr, filtered_subjs
    
    def permutation_test_within_cond(self, all_subj_corr):
        if self.permutation_level == 0:
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                all_subj_corr,
                threshold=None,         # non-parametric threshold
                tail=1,                 # one-sided (RSA > 0)
                n_permutations=PERM_TEST_N,
                seed=42, out_type='mask', verbose=False)
            significant_masks = []
            for cluster, p in zip(clusters, cluster_p_values):
                if p < 0.05:
                    significant_masks.append(cluster)
            return significant_masks
        else:
            raise NotImplementedError
        
    def permutation_test_across_cond(self, corr_1, subj_1, corr_2, subj_2):
        if self.permutation_level == 0:
            shared_subjs = list(set(subj_1).intersection(subj_2))
            subj1_id_map = {s:i for i, s in enumerate(subj_1)}
            subj2_id_map = {s:i for i, s in enumerate(subj_2)}
            subj1_indices = [subj1_id_map[s] for s in shared_subjs]
            subj2_indices = [subj2_id_map[s] for s in shared_subjs]
            corr1_shared = corr_1[subj1_indices]
            corr2_shared = corr_2[subj2_indices]
            
            # paorwise difference
            corr_diffs = corr1_shared - corr2_shared
            # print(corr_diffs.shape)
            # print(np.min(corr_diffs), np.max(corr_diffs))
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                corr_diffs,
                threshold=None,       # non-parametric cluster-forming threshold
                tail=0,               # two-sided test
                n_permutations=PERM_TEST_N,
                seed=42, out_type='mask', verbose=False
            )
            significant_masks = []
            for cluster, p in zip(clusters, cluster_p_values):
                if p < 0.05:
                    significant_masks.append(cluster)
            return significant_masks
        else:
            raise NotImplementedError("Only grand-level permutation test is supported")
    

    def display_conditional_rsa(
            self,
            ax, lmb, lmb_name, feature_mask, y_name, feature_dist_method, subjs,
            color=None, alpha=1, linestyle='-',
            display_time_steps=None,
            sig_yoffset=None):
        all_subj_corr, valid_subjs = self.get_everyone_corr(
            lmb, feature_mask, y_name, feature_dist_method, subjs)
        # plot the mean and sem
        mean_corr = np.mean(all_subj_corr, axis=0)
        sem_corr = np.std(all_subj_corr, axis=0) / np.sqrt(len(all_subj_corr))
        actual_time_points = display_time_steps
        ax.plot(
            actual_time_points, mean_corr, 
            label=lmb_name, c=color, alpha=alpha, linestyle=linestyle)
        ax.fill_between(
            actual_time_points, mean_corr-sem_corr, 
            mean_corr+sem_corr, alpha=alpha*0.4, facecolor=color)
        
        # display the significance
        if sig_yoffset is not None:
            significant_masks = self.permutation_test_within_cond(all_subj_corr)
            for cluster_mask in significant_masks:
                cluster_time_points = actual_time_points[cluster_mask]
                ax.plot(
                    [cluster_time_points[0], cluster_time_points[-1]],
                    [sig_yoffset, sig_yoffset],
                    c=color, alpha=alpha*0.7, 
                    linestyle=linestyle,
                    linewidth=6)
                
        # return the stats
        return all_subj_corr, valid_subjs, actual_time_points
    
    def display_lmb_dicts_rsa(self,
            ax, lmb_dicts, feature_mask, y_name, feature_dist_method, subjs, display_time_steps,
            colors=None, alphas=None, linestyles=None, sig_yoffsets={}, show_legend=True, 
            comparison_sig_pairs=[], pairwise_sig_styles=[]):
        lmb_corr_stats = {}
        display_timepoints = None

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
            sig_yoffset = sig_yoffsets.get(lmb_name, None)
            cond_corrs, cond_subjs, display_timepoints = self.display_conditional_rsa(
                ax, lmb, lmb_name, feature_mask, y_name, 
                feature_dist_method, subjs,
                color=c, alpha=a, linestyle=ls,
                display_time_steps=display_time_steps,
                sig_yoffset=sig_yoffset
            )
            lmb_corr_stats[lmb_name] = {
                'corrs': cond_corrs,
                'subjs': cond_subjs
            }
            
        # plot pairwise comparison
        for sig_pair, sig_style in zip(comparison_sig_pairs, pairwise_sig_styles):
            assert len(sig_pair) == 2
            significant_masks = self.permutation_test_across_cond(
                lmb_corr_stats[sig_pair[0]]['corrs'],
                lmb_corr_stats[sig_pair[0]]['subjs'],
                lmb_corr_stats[sig_pair[1]]['corrs'],
                lmb_corr_stats[sig_pair[1]]['subjs'])
            for cluster_mask in significant_masks:
                cluster_time_points = display_timepoints[cluster_mask]
                ax.plot(
                    [cluster_time_points[0], cluster_time_points[-1]],
                    [sig_style['yoffset'], sig_style['yoffset']],
                    c=sig_style['color'], alpha=sig_style['alpha'],
                    linestyle=sig_style['linestyle'], linewidth=6)
        
        annotate_time_line(
            ax, EVENTS,
            plot_ymin=RSA_PLOT_YMIN, plot_ymax=RSA_PLOT_YMAX,
            hide_ymin=RSA_PLOT_YMIN_HIDDEN, hide_ymax=RSA_PLOT_YMAX_HIDDEN)
        last_time_point = EVENTS['response']+500
        ax.set_ylim([RSA_PLOT_YMIN, RSA_PLOT_YMAX])
        ax.set_yticks(RSA_PLOT_YTICKS)
        ax.set_yticklabels(RSA_PLOT_YTICKS, fontsize=12)
        ax.hlines(0, last_time_point,0,linestyles='dashed',colors='black')
        ax.set_xlim(0, last_time_point) # tight
        if show_legend:
            ax.legend(
                loc='upper left',
                bbox_to_anchor=(1.05, 1.0))

        # finally, remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

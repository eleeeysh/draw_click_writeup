# copy from https://github.com/eleeeysh/SerialDependence and 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from tqdm import tqdm

from abc import ABC, abstractmethod

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

""" diff functions """
def color_smart_diff(x1, x2, vmin=-180, vmax=180):
    period = vmax - vmin
    diff = x1 - x2
    diff = diff + period * (diff <= vmin) - period * (diff > vmax)
    return diff

def color_smart_diff_outer(xs, ys, vmin, vmax):
    period = vmax - vmin
    diffs = np.subtract.outer(xs, ys)
    # diffs = (diffs - vmin) % period + vmin
    diffs = diffs + period * (diffs <= vmin) - period * (diffs > vmax)
    return diffs

class ValueRangeManager:
    def __init__(self, X_RANGE):
        # for different dataset this should differ
        self.X_RANGE = X_RANGE.copy()
        self.X_NUM = len(X_RANGE)
        self.X_MIN, self.X_MAX = X_RANGE[0], X_RANGE[-1]
        self.DIFF_RADIUS = self.X_NUM // 2
        self.smart_diff = lambda x1, x2: color_smart_diff(x1, x2, vmin=-self.DIFF_RADIUS, vmax=self.DIFF_RADIUS)
        self.smart_diff_outer = lambda x1, x2: color_smart_diff_outer(x1, x2, vmin=-self.DIFF_RADIUS, vmax=self.DIFF_RADIUS)
        self.DIFF_RANGE = np.arange(-self.DIFF_RADIUS, self.DIFF_RADIUS)+1

    def values_to_ids(self, values):
        ids = values - self.X_MIN
        ids = ids.astype(int)
        return ids

    def ids_to_values(self, ids):
        values = ids + self.X_MIN
        return values

    def discretize(self, values):
        values = np.round(values - self.X_MIN) % self.X_NUM + self.X_MIN
        values = values.astype(int)
        return values

from scipy.optimize import linear_sum_assignment

def stochastic_assignment(prob_table):
    N = prob_table.shape[0]
    checked = np.zeros(N).astype(bool)  # Track checked objects
    assignment = np.arange(N)  # Initially, each object is assigned to itself
    
    for i in range(N):
        if checked[i]:
            continue  # Skip if already checked
        checked[i] = True
        
        # Sample an object to swap with, based on the probabilities in row i
        chosen_j = np.random.choice(N, p=prob_table[i])
        if not checked[chosen_j]:
            # If the chosen object has not been checked, do the swap
            assignment[i], assignment[chosen_j] = chosen_j, i
            checked[chosen_j] = True
    return assignment

class ModelFitHelper:
    def __init__(self, value_range_manager):
        self.value_range_manager = value_range_manager

    def sample_from_2D_cumsum_with_slicing(self, ids, distrib, mapped):
        # using ids to locate correct distrib
        # only to use when the ids is extremely large
        samples = np.zeros_like(ids)
        for id in range(self.value_range_manager.X_NUM):
            des_map = ids == id
            n_to_sample = np.sum(des_map)
            if n_to_sample == 0:
                continue
            sample_for_id = np.random.choice(
                self.value_range_manager.X_NUM, size=n_to_sample, p=distrib[id])
            samples[des_map] = sample_for_id

        if mapped is None:
            obs = self.value_range_manager.ids_to_values(samples)
        else:
            obs = mapped[samples]
        
        return obs

    def sample_from_2D_cumsum(self, cumu_distrib, mapped):
        # select by ids
        ids = np.random.rand(len(cumu_distrib), 1)
        # obs = (ids <= cumu_distrib).argmax(axis=1)
        obs = np.sum(ids > cumu_distrib, axis=1)
        # project data to correct range
        if mapped is None:
            obs = self.value_range_manager.ids_to_values(obs)
        else:
            obs = mapped[obs]
        return obs

    def sample_from_2D(self, distrib, mapped):
        # select by ids
        # cumu_distrib = np.cumsum(distrib, axis=1)
        cumu_distrib = np.cumsum(distrib, axis=-1) # SO SLOW (~0.3s) HELP
        return self.sample_from_2D_cumsum(cumu_distrib, mapped)

    def compute_weighted_angle(self, values, weights):
        factor = 360 // self.value_range_manager.X_NUM
        values_converted = self.value_range_manager.values_to_ids(values) * factor

        # convert degree to radians
        rads = np.deg2rad(values_converted)
        
        # normalize weights
        weight_sum = np.sum(weights, axis=-1)[:, None]
        weights = weights / weight_sum
        
        # weighted average
        weighted_sins = np.sum(np.sin(rads) * weights, axis=-1)
        weighted_coss = np.sum(np.cos(rads) * weights, axis=-1)
        results = np.arctan2(weighted_sins, weighted_coss)
        
        # convert back
        results = np.rad2deg(results) % 360
        results = np.round(results / factor).astype(int) % self.value_range_manager.X_NUM
        results = self.value_range_manager.ids_to_values(results)
        return results

    def align_inference(self, xs, gts):
        # print('DEBUG starting the optimal alignment ', time.time())

        N, _ = xs.shape
        sorted_xs = np.zeros_like(xs)
        for i in range(N):
            cost_matrix = np.abs(
                self.value_range_manager.smart_diff_outer(gts[i], xs[i]))
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            sorted_xs[i, :] = xs[i][col_ind]

        # print('DEBUG ending alignment ', time.time())
        
        return sorted_xs

    def dist_to_similarity(self, dists):
        """
            dist: ranging from 1 to 0
            similarity score: randing from exp(kx) to 1
            base swap rate: swap rate when the two are opposite
        """
        dists = np.abs(dists)
        dists = dists * 90 // self.value_range_manager.DIFF_RADIUS # ranging from 0 to 90
        similarity = np.cos(np.deg2rad(dists)) # convert 0-90 degrees to 1 - 0
        return similarity

    def vectorize_swap_rates(self, swap_rate, to_align_with):
        # check the input valie
        is_scalar = np.isscalar(swap_rate)
        is_valid_2d = isinstance(swap_rate, np.ndarray) and swap_rate.shape == to_align_with.shape
        assert is_scalar or is_valid_2d, 'The swap rate need to be either a scalar value or matches the shape of stims'
        # make it a 2D
        swap_rate = np.ones_like(to_align_with) * swap_rate
        return swap_rate


    def generate_assignment_rate_trialwise(self, decs, swap_params):
        assert len(decs.shape) == 1
        n_items = len(decs)

        # compute the dists between optimal assignment and others
        dists = self.value_range_manager.smart_diff_outer(decs, decs)
        similarity =self.dist_to_similarity(dists)
        
        # assignment rate prop to dists
        sim_factor = swap_params['similarity_bound_factor']
        base_swap = swap_params['base_swap_rate'] # scalar or 2D
        base_swap = self.vectorize_swap_rates(base_swap, dists[:, :-1]) # vectorization
        assignment_weights = np.exp(sim_factor * similarity)
        off_eye_mask = ~np.eye(n_items, dtype=bool)        
        assignment_weights[off_eye_mask] *= base_swap.flatten() # ?

        # normalize
        assignment_weights = assignment_weights / np.sum(assignment_weights, axis=1, keepdims=True)
        return assignment_weights

    def generate_assignment_rate_batchwise(self, items, target_ids, swap_params):
        """
            dist: ranging from 1 to 0
            similarity score: randing from exp(kx) to 1
            base swap rate: swap rate when the two are opposite
        """
        rid_helpers = np.arange(items.shape[0])

        # compute the dists between optimal assignment and others
        targets = items[rid_helpers, target_ids][:, None]
        dists = self.value_range_manager.smart_diff(items, targets)
        similarity = self.dist_to_similarity(dists)
        
        # apply similarity based assignment
        sim_factor = swap_params['similarity_bound_factor']
        assignment_weights = np.exp(sim_factor * similarity)
        
        # apply random swap rate
        base_swap = swap_params['base_swap_rate'] # 
        base_swap = self.vectorize_swap_rates(base_swap, dists[:, :-1]) # vectorization
        neighbor_mask = np.ones_like(items).astype(bool)
        neighbor_mask[rid_helpers, target_ids] = False
        assignment_weights[neighbor_mask] *= base_swap.flatten()

        # normalize
        assignment_weights = assignment_weights / np.sum(assignment_weights, axis=1, keepdims=True)
        return assignment_weights

    def align_inference_dependent(self, aligned, swap_params):
        """ suppose there are m items each trial, N trials in total
            random or non-random swap occurs to the optimal assignment of decoded to items
            this method gurantee consistency between responses to different items in the same trial
        """
        N, n_items = aligned.shape
        realigned = np.zeros_like(aligned)
        for trial_id in range(N):
            probs = self.generate_assignment_rate_trialwise(aligned[trial_id], swap_params)
            assignment = stochastic_assignment(probs)
            realigned[trial_id, :] = aligned[trial_id][assignment]
        
        return realigned

    def align_inference_independent(self, aligned, swap_params):
        """ suppose there are m items each trial, N trials in total
            random or non-random swap occurs to the optimal assignment of decoded to items
            this metho assume independence between asisgnments (which are not true)
        """
        # print('DEBUG: alignment start ', time.time())
        N, n_items = aligned.shape
        realigned = np.zeros_like(aligned)
        rid_helpers = np.arange(N)
        for target_id in range(n_items):
            target_ids = np.full((N,), target_id)
            probs = self.generate_assignment_rate_batchwise(aligned, target_ids, swap_params)
            sample_ids = self.sample_from_2D(probs, np.arange(n_items))
            realigned[:, target_id] = aligned[rid_helpers, sample_ids]

        # print('DEBUG: alignment end ', time.time())
        
        return realigned


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def wk_correct_distrib(distrib, w, k, offset=1e-5):
    # k-correction
    p = (distrib + offset) ** k
    p = p / np.sum(p, axis=1)[:, None]
    
    # w-correction
    p = p * w + (1 - w) / p.shape[-1] 
    
    return p

from copy import deepcopy

""" general distribution of response given target """
class baseDistribModel(ABC):
    def __init__(self, value_range_manager: ValueRangeManager, prior_params: dict):
        self.initial_params = deepcopy(prior_params)
        self.prior = None
        self.value_range_manager = value_range_manager
        self.create_prior(prior_params)
        self.create_cumu_prior()
        self.create_log_prior()

    @abstractmethod
    def create_prior(self, params):
        pass

    def create_cumu_prior(self):
        # create cumsum
        """ for acceleration, precompute the cumsum result """ # ACCELERTAED
        self.cumu_prior = np.cumsum(self.prior, axis=-1)

    def create_log_prior(self):
        # create cumsum
        """ for acceleration, precompute the cumsum result """ # ACCELERTAED
        self.log_prior = np.log(self.prior+1e-5)

    def loc(self, *args):
        """ give likelihood of p(~|arg1, arg2...) """ 
        args = tuple([self.value_range_manager.values_to_ids(arg) for arg in args])
        return self.prior[args]
    
    def loc_cumu(self, *args):
        """ give likelihood of p(~|arg1, arg2...) """ 
        args = tuple([self.value_range_manager.values_to_ids(arg) for arg in args])
        return self.cumu_prior[args]
    
    def loc_log(self, *args):
        """ give likelihood of p(~|arg1, arg2...) """ 
        args = tuple([self.value_range_manager.values_to_ids(arg) for arg in args])
        return self.log_prior[args]

BIAS_PATH = os.path.join(BASE_PATH, 'stats/bias_average.npy')
    
class OriDistribModel(baseDistribModel):
    def load_base_bias(self, prior_params):
        """ the very basic prior from external dataset """
        default_prior_bias_path = BIAS_PATH
        prior_bias_path = prior_params.get('bias_path', default_prior_bias_path)
        self.base_bias = np.load(prior_bias_path)

    def create_distrib(self, prior_params):
        k_base = prior_params['k_base']
        alpha = prior_params['alpha']
        beta = prior_params['beta']
        w = prior_params['w']

        N = self.value_range_manager.X_NUM

        # initialization
        xs = np.arange(N)/N*np.pi
        ks = k_base * (1 + beta * (np.cos(2*xs)**2)) # to match nature standards
        mu = (np.arange(N) + alpha * self.base_bias)/N*np.pi
    
        ys = np.arange(N)/N*np.pi
        diffs = color_smart_diff_outer(mu, ys, vmin=-np.pi, vmax=np.pi)
        distrib = np.exp(ks[:, None] * np.cos(diffs))
        distrib = distrib / np.sum(distrib, axis=1)[:, None]
        
        # incorporate random guess
        mixed = distrib * (1 - w) + w / N
        mixed = mixed / np.sum(mixed, axis=1)[:, None]
        return mixed

    def create_prior(self, prior_params):
        self.load_base_bias(prior_params)
        self.prior = self.create_distrib(prior_params)

""" For incoporation of context (history) """
def incorporate_distributions(base, to_incorporate):
    incorporated = base * to_incorporate
    incorporated = incorporated / np.sum(incorporated, axis=-1, keepdims=True)
    return incorporated


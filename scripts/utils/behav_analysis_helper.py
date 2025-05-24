from .inverted_encoding import (
    compute_accuracy, compute_bias, deg_signed_diff
)

import numpy as np
import pandas as pd

def df_to_errs(df, lmb, stim1_lmb=None, stim2_lmb=None):
    if lmb is not None:
        mask = lmb(df)
        if np.sum(mask) == 0:
            return None
        df = df[mask]    

    # tease out stim 1 and stim 2
    stims = np.concatenate([
        df['stim_1'].to_numpy(copy=True),
        df['stim_2'].to_numpy(copy=True)
    ])
    non_target = np.concatenate([
        df['stim_2'].to_numpy(copy=True),
        df['stim_1'].to_numpy(copy=True)
    ])
    resps = np.concatenate([
        df['resp_1'].to_numpy(copy=True),
        df['resp_2'].to_numpy(copy=True)
    ])
    subjects = np.concatenate([
        df['participant'].to_numpy(copy=True),
        df['participant'].to_numpy(copy=True)
    ])
    prev_ref_name = 'prev_last_response'
    # prev_ref_name = 'prev_last_resp_stim'
    prev_resps = np.concatenate([
        df[prev_ref_name].to_numpy(copy=True),
        df[prev_ref_name].to_numpy(copy=True)
    ])
    # remove nan
    valid_mask = (~(np.isnan(resps))) & (~(np.isnan(prev_resps)))

    # apply the per stim lmb
    stim1_mask = np.ones(len(df), dtype=bool)
    stim2_mask = np.ones(len(df), dtype=bool)
    if stim1_lmb is not None:
        stim1_mask = stim1_lmb(df)
    if stim2_lmb is not None:
        stim2_mask = stim2_lmb(df)
    per_stim_mask = np.concatenate([
        stim1_mask, stim2_mask
    ])
    valid_mask = valid_mask & per_stim_mask

    # compute erros and do filtering
    errs = deg_signed_diff(resps[valid_mask]-stims[valid_mask])
    subjects = subjects[valid_mask]
    stims = stims[valid_mask]
    non_target = non_target[valid_mask]
    prev_resps = prev_resps[valid_mask]
    # collect results
    results = pd.DataFrame({
        'subject': subjects,
        'stim': stims,
        'err': errs,
        'non_target': non_target,
        'prev_resp': prev_resps,
    })
    return results

def errdf_to_stim_distrib(err_df, T=180, ref=None):
    # convert err to distrib
    distrib = np.zeros((T, T))
    err_ids = err_df['err'].to_numpy().astype(int) 
    # flip to compare bias
    if ref is not None:
        ref_dir = deg_signed_diff(
            (err_df[ref] - err_df['stim']).to_numpy(), epoch=T)
        flip_mask = ref_dir < 0
        err_ids[flip_mask] = - err_ids[flip_mask]
    # get the aligned distribution
    err_ids = err_ids % T
    stim_ids = np.round(err_df['stim'].to_numpy()).astype(int) % T
    np.add.at(distrib, (stim_ids, err_ids), 1)
    # distrib = distrib / np.sum(distrib, axis=-1, keepdims=True)
    return distrib

def errdf_to_distrib(err_df, T=180, ref=None):
    # convert err to distrib
    distrib = np.zeros(T)
    err_ids = err_df['err'].to_numpy().astype(int) 
    # flip to compare bias
    if ref is not None:
        ref_dir = deg_signed_diff(
            (err_df[ref] - err_df['stim']).to_numpy(), epoch=T)
        flip_mask = ref_dir < 0
        err_ids[flip_mask] = - err_ids[flip_mask]
    # get the aligned distribution
    err_ids = err_ids % T
    np.add.at(distrib, err_ids, 1)
    distrib = distrib / np.sum(distrib)
    return distrib

def subj_behav_df_to_stats(subj_df, lmb, stim1_lmb, stim2_lmb, stat_type):
    err_df = df_to_errs(subj_df, lmb, stim1_lmb, stim2_lmb)
    if err_df is None:
        # subject do not have enough data...
        return None
    ref_type = {
        'accuracy': None,
        'bias': None,
        'sd': 'prev_resp',
        'sur': 'non_target',
    }[stat_type]
    stat_func = compute_accuracy if stat_type == 'accuracy' else compute_bias
    distrib = errdf_to_distrib(err_df, T=180, ref=ref_type)
    stats = stat_func(distrib, T=180)
    return stats

# helper function to add stats relate to sd analysis
def group_and_divie_from_median(A, B, N_groups):
    # Define the bins and labels
    bins = np.linspace(0, 90, num=N_groups+1)
    bin_ids = np.digitize(A, bins).astype(float) - 1 # nan -> N-group+1
    bin_ids[(bin_ids<0) | (bin_ids>=N_groups)]= np.nan
    bin_ids = bin_ids / N_groups # so it is comparable across different N_bins

    df = pd.DataFrame({'A': A, 'B': B, 'group': bin_ids})
    within_group_ids = np.full_like(B, np.nan, dtype='float')
    for group_id in df['group'].dropna().unique():
        group_mask = df['group'] == group_id
        B_values = df.loc[group_mask, 'B']
        B_median = B_values.median(skipna=True)

        if not np.isnan(B_median):
            # just set everything to nan
            within_group_ids[group_mask & (B_values < B_median)] = -1
            within_group_ids[group_mask & (B_values > B_median)] = 1
            within_group_ids[group_mask & (B_values == B_median)] = 0

    return bin_ids, within_group_ids

def df_pad_sd_stats(df, n_sd_bins):
    # df: for one subject ONLY
    for stim_id in [1, 2]:
        resp_errs = deg_signed_diff(df[f'resp_{stim_id}']-df[f'stim_{stim_id}']).values
        sd_diffs = deg_signed_diff(df['prev_last_response']-df[f'stim_{stim_id}']).values

        # flip it
        neg_sd_diff_mask = sd_diffs < 0
        resp_errs[neg_sd_diff_mask] = -resp_errs[neg_sd_diff_mask]
        sd_diffs[neg_sd_diff_mask] = -sd_diffs[neg_sd_diff_mask]

        temp_bin_ids = np.full_like(sd_diffs, np.nan, dtype='float')
        temp_within_group_ids = np.full_like(sd_diffs, np.nan, dtype='float')

        for mode in ['draw', 'click']:
            mode_mask = df['mode'] == mode
            # get group ids and within-group ids
            bin_ids, within_group_ids = group_and_divie_from_median(
                sd_diffs[mode_mask], resp_errs[mode_mask], n_sd_bins)
            temp_bin_ids[mode_mask] = bin_ids
            temp_within_group_ids[mode_mask] = within_group_ids
        
        df[f'sd_diff_group_{stim_id}'] = temp_bin_ids
        df[f'sd_diff_within_group_{stim_id}'] = temp_within_group_ids

    return df

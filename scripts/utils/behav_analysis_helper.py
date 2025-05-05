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
    prev_resps = np.concatenate([
        df['prev_last_response'].to_numpy(copy=True),
        df['prev_last_response'].to_numpy(copy=True)
    ])
    # remove nan
    valid_mask = (~(np.isnan(resps))) & (~(np.isnan(prev_resps)))

    # apply the per stim lmb
    per_stim_mask = np.concatenate([
        stim1_lmb(df),
        stim2_lmb(df),
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

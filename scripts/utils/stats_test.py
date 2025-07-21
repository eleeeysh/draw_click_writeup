import numpy as np
import scipy.stats as scipy_stats

def sem_func(xs, axis=None):
    if axis is None:
        return np.std(xs, ddof=1) / np.sqrt(len(xs))
    elif axis == 0:
        return np.std(xs, axis=axis, ddof=1) / np.sqrt(len(xs))
    else:
        raise NotImplementedError(f'axis {axis} not supported')

def stat_results_apply_ttest_2rel(stats_results, cond_names, use_wilcoxon=True):
    grouped = {}
    assert len(cond_names) == 2
    # regrouped
    for cond_name in cond_names:
        cond_stats = stats_results[cond_name]
        for stat_type, subj_stats in cond_stats.items():
            if stat_type not in grouped:
                grouped[stat_type] = []
            grouped[stat_type].append(subj_stats.copy())

    ttest_results = {}
    for stat_type, grouped_stats in grouped.items():
        # only include shared subjects
        shared_subjs = set.intersection(
            *[set(subj_stats.keys()) for subj_stats in grouped_stats])
        shared_subjs = list(shared_subjs)
        filtered_subj_stats = [
            [subj_stats[subj] for subj in shared_subjs] 
            for subj_stats in grouped_stats]
        # stat_t, stat_pval = scipy_stats.ttest_rel(*filtered_subj_stats)
        if use_wilcoxon:
            # use wilcoxon test
            stat_t, stat_pval = scipy_stats.wilcoxon(
                np.array(filtered_subj_stats[0])-np.array(filtered_subj_stats[1]), 
                alternative='two-sided')
        else:
            # use ttest
            stat_t, stat_pval = scipy_stats.ttest_rel(*filtered_subj_stats)
        ttest_results[stat_type] = {
            't_stat': stat_t,
            'p_val': stat_pval,
        }

    return ttest_results
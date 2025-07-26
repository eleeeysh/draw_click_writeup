import numpy as np
import scipy.stats as scipy_stats

def sem_func(xs, axis=None):
    if axis is None:
        return np.std(xs, ddof=1) / np.sqrt(len(xs))
    elif axis == 0:
        return np.std(xs, axis=axis, ddof=1) / np.sqrt(len(xs))
    else:
        raise NotImplementedError(f'axis {axis} not supported')

enforce_ttest = False

# 1 sample two tests
def stat_results_apply_ttest_1sample(stats):
    # apply normality test
    s_normal = scipy_stats.shapiro(stats)[1] > 0.05
    stats = np.array(stats)
    stats = stats[~(np.isnan(stats))]
    use_wilcoxon = not (s_normal or enforce_ttest)
    if not use_wilcoxon:
        # use ttest
        stat, stat_pval = scipy_stats.ttest_1samp(
            stats, 0,
            alternative='greater')
        return {
            't_stat': stat,
            'p_val': stat_pval,
            'df': len(stats) - 1,
        }
    else:
        # use wilcoxon test
        stat, stat_pval = scipy_stats.wilcoxon(
            stats, alternative='greater')
        return {
            'w_stat': stat,
            'p_val': stat_pval,
            'n': len(stats),
        }

def default_paired_test(s1, s2):
    # decide whether to use ttest or wilcoxon
    normality_test_p = scipy_stats.shapiro(s1-s2)[1]
    s_normal = normality_test_p > 0.05
    use_wilcoxon = not (s_normal or enforce_ttest)

    if not s_normal:
        print(f'Fail normality test: p={normality_test_p:.3f}')

    if use_wilcoxon:
        # use wilcoxon test
        stat_t, stat_pval = scipy_stats.wilcoxon(
            s1, s2, 
            alternative='two-sided')
        return {
            'w_stat': stat_t,
            'p_val': stat_pval,
            'n': len(s1),
        }
    else:
        # use ttest
        stat_t, stat_pval = scipy_stats.ttest_rel(
            s1, s2,
            alternative='two-sided')
        return {
            't_stat': stat_t,
            'p_val': stat_pval,
            'df': len(s1) - 1,
        }

# two paired test
def stat_results_apply_ttest_2rel(stats_results, cond_names):
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
        
        filtered_stat1 = np.array(filtered_subj_stats[0])
        filtered_stat2 = np.array(filtered_subj_stats[1])

        ttest_results[stat_type] = default_paired_test(
            filtered_stat1, 
            filtered_stat2
        )

    return ttest_results

def paired_test_to_str(stat_results):
    if 'w_stat' in stat_results:
        # wilcoxon test
        return f"W={int(stat_results['w_stat'])}, n={int(stat_results['n'])} (p={stat_results['p_val']:.3f})"
    else:
        return f"t({stat_results['df']})={stat_results['t_stat']:.2f} (p={stat_results['p_val']:.3f})"

def display_ttest_rel2_results(stats_results, cond_names):
    ttest_results = stat_results_apply_ttest_2rel(stats_results, cond_names)
    for cond_name in ttest_results:
        cond_stats = ttest_results[cond_name]
        print(f'{cond_name}: {paired_test_to_str(cond_stats)}')

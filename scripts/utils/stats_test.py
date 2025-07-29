import numpy as np
import scipy.stats as scipy_stats

def sem_func(xs, axis=None):
    if axis is None:
        return np.std(xs, ddof=1) / np.sqrt(len(xs))
    elif axis == 0:
        return np.std(xs, axis=axis, ddof=1) / np.sqrt(len(xs))
    else:
        raise NotImplementedError(f'axis {axis} not supported')

DEFAULT_TEST_TYPE = 'wilcoxon'

# 1 sample two tests
def stat_results_apply_ttest_1sample(
        stats, default_test_type=DEFAULT_TEST_TYPE,
        include_additional=False,
        test_side='greater'):
    # apply normality test
    stats = np.array(stats)
    stats = stats[~(np.isnan(stats))]
    normality_test_p = scipy_stats.shapiro(stats)[1]

    # set type of stats to use
    if default_test_type is not None:
        use_wilcoxon = default_test_type == 'wilcoxon'
    else:
        # decide whether to use ttest or wilcoxon
        s_normal = normality_test_p > 0.05
        use_wilcoxon = not s_normal

    if not use_wilcoxon:
        # use ttest
        stat, stat_pval = scipy_stats.ttest_1samp(
            stats, 0,
            alternative=test_side)
        stat_dict = {
            't_stat': stat,
            'p_val': stat_pval,
            'df': len(stats) - 1,
            'normality_p': normality_test_p,
        }
        if include_additional:
            stat_dict['mean'] = np.mean(stats)
            stat_dict['sem'] = sem_func(stats)
    else:
        # use wilcoxon test
        stat, stat_pval = scipy_stats.wilcoxon(
            stats, alternative=test_side)
        stat_dict = {
            'w_stat': stat,
            'p_val': stat_pval,
            'n': len(stats),
            'normality_p': normality_test_p,
        }
        if include_additional:
            stat_dict['mdn'] = np.median(stats)
            stat_dict['q1'] = np.quantile(stats, 0.25)
            stat_dict['q3'] = np.quantile(stats, 0.75)

    return stat_dict

def default_paired_test(s1, s2, default_test_type=DEFAULT_TEST_TYPE):
    diffs = s2 - s1
    normality_test_p = scipy_stats.shapiro(diffs)[1]

    # set type of stats to use
    if default_test_type is not None:
        use_wilcoxon = default_test_type == 'wilcoxon'
    else:
        # decide whether to use ttest or wilcoxon
        s_normal = normality_test_p > 0.05
        use_wilcoxon = not s_normal

    if use_wilcoxon:
        # use wilcoxon test
        stat_t, stat_pval = scipy_stats.wilcoxon(
            s1, s2, 
            alternative='two-sided')
        return {
            'w_stat': stat_t,
            'p_val': stat_pval,
            'n': len(s1),
            'normality_p': normality_test_p,
            'mdn': np.median(diffs),
            'q1': np.quantile(diffs, 0.25),
            'q3': np.quantile(diffs, 0.75),
        }
    else:
        # use ttest
        stat_t, stat_pval = scipy_stats.ttest_rel(
            s1, s2,
            alternative='two-sided')
        return {
            't_stat': stat_t,
            'p_val': stat_pval,
            'df': len(s1) - 2,
            'normality_p': normality_test_p,
            'mean': np.mean(diffs),
            'sem': sem_func(diffs),
        }

# two paired test
def stat_results_apply_ttest_2rel(stats_results, cond_names, 
        default_test_type=DEFAULT_TEST_TYPE):
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
            filtered_stat2,
            default_test_type=default_test_type
        )

    return ttest_results


def paired_test_to_str(stat_results, factor=1):
    if 'w_stat' in stat_results:
        # wilcoxon test
        diff_summary = (f"Mdn (Q1, Q3):"
            f"{stat_results['mdn']*factor:.2f} "
            f"({stat_results['q1']*factor:.2f},"
            f"{stat_results['q3']*factor:.2f})"
        )
        paired_test = (
            f"W={int(stat_results['w_stat'])},"
            f"n={int(stat_results['n'])},"
            f"p={stat_results['p_val']:.3f}"
        )
    else:
        diff_summary = (
            f"M \u00B1 SEM: {stat_results['mean']*factor:.2f} "
            f"\u00B1 {stat_results['sem']*factor:.2f}"
        )
        paired_test = (
            f"t({stat_results['df']})={stat_results['t_stat']:.2f},"
            f"p={stat_results['p_val']:.3f}"
        )
    return f"{diff_summary}\n{paired_test}"


def fast_1sample_test(stats, factor=1, test_side='greater'):
    for test_type in ['ttest', 'wilcoxon']:
        test_stats = stat_results_apply_ttest_1sample(
            stats, default_test_type=test_type, 
            test_side=test_side,
            include_additional=True)
        test_stat_str = paired_test_to_str(test_stats, factor=factor)
        if test_type == 'ttest':
            test_stat_str += f' (normality p={test_stats["normality_p"]:.3f})'

        print(f"1-sample test ({test_type}): \n{test_stat_str}\n")


def display_ttest_rel2_results(stats_results, cond_names, factor=1):
    # display both...
    sum_results = {}
    test_types = ['ttest', 'wilcoxon']

    for test_type in test_types:
        ttest_results = stat_results_apply_ttest_2rel(
            stats_results, cond_names,
            default_test_type=test_type)        
        for cond_name in ttest_results:
            cond_stats = ttest_results[cond_name]
            cond_stats_str = paired_test_to_str(cond_stats, factor=factor)
            if cond_name not in sum_results:
                sum_results[cond_name] = {}
                sum_results[cond_name]['normality_p'] = cond_stats['normality_p']
            sum_results[cond_name][test_type] = cond_stats_str

    for cond_name, cond_stats in sum_results.items():
        print(f'Group: {cond_name}')
        print(f': normality_p = {cond_stats["normality_p"]:.3f}')
        for test_type in test_types:
            print(f':: {test_type}: {cond_stats[test_type]}')


import matplotlib.pyplot as plt
import numpy as np

BIAS_MAG_MAX = 0.010
BIAS_MARK_MAG_MAX = 0.01

""" group trials by serial difference, and check bias gtoup by group """
def raw_display_stats_as_tuning_func_of_sd_diff(
        results, stats_type, common_lmb=None, plot_name='',
        return_subj_stats=False, N_SD_BINS=6,
        item_weights_lmb=None, display_distrib_func=None,
        to_display=True):
    
    if to_display:
        fig, axs = plt.subplots(1, N_SD_BINS, figsize=(9*N_SD_BINS, 9))
    
    all_cond_stats = []
    for sd_bin_id in np.arange(N_SD_BINS):
        sd_bin_val = sd_bin_id / N_SD_BINS 
        sd1_lmb = lambda d: (
            (d['sd_diff_group_1'] == sd_bin_val)
        ).values
        sd2_lmb = lambda d: (
            (d['sd_diff_group_2'] == sd_bin_val)
        ).values
        sd_valid_lmb = None
        if common_lmb is not None:
            sd_valid_lmb = lambda d: (
                common_lmb(d) & (sd1_lmb(d) | sd2_lmb(d)))
        else:
            sd_valid_lmb = lambda d: (
                (sd1_lmb(d) | sd2_lmb(d)))
        plot_settings = {
            'stim 1': {
                'target': 'stim 1',
                'lmb': sd1_lmb,
                # 'to_plot': False,
            },
            'stim 2': {
                'target': 'stim 2',
                'lmb': sd2_lmb,
                # 'to_plot': False,
            },
            'combined': {
                'target': 'combined',
                'lmb': None,
            }
        }
        ax = axs[sd_bin_id] if to_display else None
        cond_stats = display_distrib_func(ax,
            results,
            stats_type=stats_type, 
            common_lmb=sd_valid_lmb, condition_lmbs=plot_settings,
            item_weights_lmb=item_weights_lmb,
            return_subj_stats=return_subj_stats)
        all_cond_stats.append(cond_stats)
        
        if to_display:
            ax.set_title(f'sd-diff={sd_bin_id+1}', fontsize=16)

    stat_short_name = {
        'accuracy': 'acc',
        'sd': 'sd',
    }[stats_type]
    plot_full_name = f'{plot_name}delay_decoded_{stat_short_name}_as_sd_diff_func.png'

    return all_cond_stats, plot_full_name

SD_PLOTS_LABEL_FONTSIZE = {
    'xtick': 22,
    'ytick': 22,
    'xaxis': 24,
    'yaxis': 24,
}

def raw_display_stats_as_tuning_func_of_sd_diff_compact(
        ax, all_cond_stats, stats_type, plot_color=None, label=None, 
        mark_sig=True, mark_sig_offset=0.05, return_results=False,
        N_SD_BINS=6):

    xtick_label_fontsize = SD_PLOTS_LABEL_FONTSIZE['xtick']
    ytick_label_fontsize = SD_PLOTS_LABEL_FONTSIZE['ytick']
    xaxis_label_fontsize = SD_PLOTS_LABEL_FONTSIZE['xaxis']
    yaxis_label_fontsize = SD_PLOTS_LABEL_FONTSIZE['yaxis']

    plot_xs, plot_ys, plot_xerrs, plot_is_sig = [], [], [], []
    stat_name = {
        'accuracy': 'accuracy',
        'sd': 'bias'
    }[stats_type]
    for i in range(N_SD_BINS):
        sd_diff_deg = int(90 / N_SD_BINS * (i+0.5))
        sd_m = all_cond_stats[i]['combined'][stat_name]['mean']
        sd_sem = all_cond_stats[i]['combined'][stat_name]['sem']
        sd_pval = all_cond_stats[i]['combined'][stat_name]['p_val']
        # print(sd_pval)
        plot_is_sig.append(sd_pval <= 0.05)
        plot_xs.append(sd_diff_deg)
        plot_ys.append(sd_m)
        plot_xerrs.append(sd_sem)
    ax.errorbar(
        plot_xs, plot_ys, yerr=plot_xerrs, 
        linewidth=4, elinewidth=2,
        fmt='o-', capsize=5, color=plot_color, label=label)

    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel(
        'Last Resp - Current Stim', 
        fontsize=xaxis_label_fontsize)
    # yname = r'Error ($r_{t-1}$ aligned, $\times 10^{-3}$)' if stats_type == 'sd' else 'Evidence'
    yname = r'Bias towards $r_{t-1}$  ($x10^{-3}$)' if stats_type == 'sd' else 'Evidence'
    ax.set_ylabel(yname, fontsize=yaxis_label_fontsize)
    ymin, ymax = {
        'accuracy': (-0.1, 1.2),
        'sd': (-BIAS_MAG_MAX, BIAS_MAG_MAX)
    }[stats_type]
    ax.set_ylim(ymin, ymax)

    # make the label ticks sparse
    yticks = {
        'accuracy': np.arange(6) * 0.2,
        'sd': np.linspace(-BIAS_MARK_MAG_MAX, BIAS_MARK_MAG_MAX, 5)
    }[stats_type]
    xticks = 15 * (np.arange(5)+1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=xtick_label_fontsize)
    ax.set_yticks(yticks)
    ytick_labels = [f'{int(np.round(ytick*1000))}' for ytick in yticks]
    ax.set_yticklabels(ytick_labels, fontsize=ytick_label_fontsize)

    # mark the significance
    if (stats_type == 'sd') and mark_sig:
        # only mark sig for serial bias
        for  i in range(N_SD_BINS):
            if plot_is_sig[i]:
                sig_plot_y = plot_ys[i] + (mark_sig_offset * (ymax - ymin) + plot_xerrs[i])
                ax.text(plot_xs[i], sig_plot_y, '*', fontsize=20, color=plot_color, ha='center', va='center')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if return_results:
        return plot_xs, plot_ys, plot_xerrs, ymin, ymax
    

""" check the development of bias """
stim1_valid_lmb = lambda d: ((d['stim_1_to_report']) | (d['trial_code'] == 1)).values
stim2_valid_lmb = lambda d: ((d['stim_2_to_report']) | (d['trial_code'] == 1)).values 
has_valid_prev_lmb = lambda d: (d['prev_last_response'].notna()).values

def raw_plot_all_combined_phase_mode_compare(
        iem_results, plot_over_phase_func,
        stat_type, stat_name, conds_included, 
        common_lmb=None, stim1_extra_lmb=None, stim2_extra_lmb=None, plot_extension='',
        axs=None, plot_alpha=1, label=None, plot_color=None):
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    stim1_lmb = stim1_valid_lmb
    if stim1_extra_lmb is not None:
        stim1_lmb = lambda d: (stim1_valid_lmb(d) & stim1_extra_lmb(d))

    stim2_lmb = stim2_valid_lmb
    if stim2_extra_lmb is not None:
        stim2_lmb = lambda d: (stim2_valid_lmb(d) & stim2_extra_lmb(d))

    plot_settings = {
        'stim 1': {
            'target': 'stim 1',
            'lmb': stim1_lmb,
            # 'to_plot': False,
        },
        'stim 2': {
            'target': 'stim 2',
            'lmb': stim2_lmb,
            # 'to_plot': False,
        },
        'combined': {
            'target': 'combined',
            'lmb': None,
        }
    }

    mode_phase_results, mode_lmbs, mode_color_dicts, mode_offset_dicts = {}, {}, {}, {}
    for result_name, result in iem_results.items():
        mode_phase_results[result_name] = result['result']
        mode_lmbs[result_name] = result['lmb']
        mode_color_dicts[result_name] = result['color']
        mode_offset_dicts[result_name] = result['offset']

    plot_ymin, plot_ymax = {
        'accuracy': (0.0, 1.0),
        'sd': (-BIAS_MARK_MAG_MAX, BIAS_MARK_MAG_MAX),
        'sur': (-BIAS_MARK_MAG_MAX, BIAS_MARK_MAG_MAX)
    }[stat_type]

    for phase_step in [0, 1]:
        ax = axs[phase_step]
        for mode in conds_included:
            selected_mode_lmb = mode_lmbs[mode]
            raw_selected_lmb = lambda d: (selected_mode_lmb(d) & has_valid_prev_lmb(d))
            selected_lmb = raw_selected_lmb
            if common_lmb is not None:
                selected_lmb = lambda d: (raw_selected_lmb(d) & common_lmb(d))
            x_offset = mode_offset_dicts[mode]
            to_show_sig = stat_type != 'accuracy'

            cond_plot_label = f'{mode}'
            if label is not None:
                if len(conds_included) > 1:
                    cond_plot_label = f'{mode}' + label
                else:
                    cond_plot_label = label

            cond_plot_color = plot_color
            if plot_color is None:
                cond_plot_color = mode_color_dicts[mode]

            plot_over_phase_func(
                ax, mode_phase_results[mode][phase_step], 
                stat_type, stat_name, phase_step,
                plot_settings, selected_lmb, 
                plot_ymin=plot_ymin, plot_ymax=plot_ymax, 
                label=cond_plot_label,
                x_offset=x_offset,
                color=cond_plot_color,
                alpha=plot_alpha, show_significance=to_show_sig)

        ylabel_name = {
            'sd': 'Serial Bias',
            'accuracy': 'Evidence',
            'sur': 'Surrounding Bias'        
        }[stat_type]
        ax.set_ylabel(ylabel_name, fontsize=20)
        ax.set_xlabel('Train -> Test Phases', fontsize=20)
        if len(conds_included) > 1:
            ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right')

        # MARK y ticks
        mark_ymin, mark_ymax = {
            'accuracy': (0.0, 1.0),
            'sd': (-BIAS_MARK_MAG_MAX, BIAS_MARK_MAG_MAX),
            'sur': (-BIAS_MARK_MAG_MAX, BIAS_MARK_MAG_MAX)
        }[stat_type]
        ax.set_yticks(np.linspace(mark_ymin, mark_ymax, 3))

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


    plt.tight_layout()
    
    # get the plot name
    plot_name = 'mode_compare' if len(conds_included) > 1 else conds_included[0]
    plot_name += plot_extension
    plot_name = f'delay_phase_{stat_type}_{stat_name}_{plot_name}.png'

    return plot_name

## to separate small and large sd
def raw_plot_sd_2groups_phase_mode_compare(
        iem_results, plot_over_phase_func,
        stat_type, stat_name, conds_included, 
        common_lmb=None, plot_extension='', plot_color=None,
        show_legend=True):
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    sd_diff_lmbs = {
        'small d-serial': {
            'stim1_extra_lmb': lambda d: (d['sd_diff_group_1'] < 0.5).values,
            'stim2_extra_lmb': lambda d: (d['sd_diff_group_2'] < 0.5).values,
        },
        'large d-serial': {
            'stim1_extra_lmb': lambda d: (d['sd_diff_group_1'] >= 0.5).values,
            'stim2_extra_lmb': lambda d: (d['sd_diff_group_2'] >= 0.5).values,
        }
    }

    for sd_diff_group_name in sd_diff_lmbs:
        lmbs = sd_diff_lmbs[sd_diff_group_name]
        plot_alpha = 1 if sd_diff_group_name.startswith('small') else 0.5
        raw_plot_all_combined_phase_mode_compare(
            iem_results, plot_over_phase_func,
            stat_type,stat_name,
            conds_included=conds_included,
            plot_extension='_small_sd_diff', **lmbs, common_lmb=common_lmb,
            axs=axs, plot_alpha=plot_alpha, label=' '+sd_diff_group_name,
            plot_color=plot_color)

    if show_legend:
        for ax in axs:
            # show legend
            ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right')

    # get plot name
    plot_name = 'mode_compare' if len(conds_included) > 1 else conds_included[0]
    plot_name += 'sd_2groups'
    plot_name += plot_extension
    plot_name = f'delay_{stat_type}_{stat_name}_{plot_name}.png'
    return plot_name
    
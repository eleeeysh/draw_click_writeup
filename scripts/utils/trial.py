import numpy as np
import pandas as pd
import warnings

""" functions to retrive trials and their N-back trials """
def get_prev_trial_ids(ids, N_back):
    # mapping of trial to row
    row_table = [None for _ in range(ids.max()+1)]
    for row_id, trial_idx in enumerate(ids):
        row_table[trial_idx] = row_id
    row_table = np.array(row_table)
    
    # fetch the row id of previous trials
    trial_table = np.zeros(ids.max()+1)
    trial_table[ids] = 1
    has_prev_rids = []
    prev_trial_rids = []
    for row_id, trial_idx in enumerate(ids):
        prev_trial_idx = trial_idx - N_back
        if prev_trial_idx >= 0 and trial_table[prev_trial_idx] > 0:
            # the 'previous' trial exist
            has_prev_rids.append(row_id)
            prev_trial_rid = int(row_table[prev_trial_idx])
            prev_trial_rids.append(prev_trial_rid)
    
    has_prev_rids = np.array(has_prev_rids)
    prev_trial_rids = np.array(prev_trial_rids)    
    return has_prev_rids, prev_trial_rids

def extract_pair_from_one_block(df, N_back):
    original_trial_ids = df['trial'].to_numpy()
    has_prev_rids, prev_trial_rids = get_prev_trial_ids(
        original_trial_ids, N_back)
    preceding = df.iloc[prev_trial_rids]
    succeeding =  df.iloc[has_prev_rids]
    return preceding, succeeding

def get_pair_rowid_N_back(df, N_back):
    preceding_list = []
    succeeding_list = []
    
    # group data into block
    df['dummy_id'] = np.arange(len(df)) # dummy position ids
    group_criteria = ['participant', 'sample_stage', 'block']
    grouped = df.groupby(group_criteria)

    for _, group in grouped:
        original_trial_ids = group['trial'].to_numpy(copy=True)
        suc_rids, pre_rids = get_prev_trial_ids(
            original_trial_ids, N_back)
        suc_real_rids = group.iloc[suc_rids].dummy_id.to_numpy(copy=True)
        pre_real_rids = group.iloc[pre_rids].dummy_id.to_numpy(copy=True)
        preceding_list.append(pre_real_rids)
        succeeding_list.append(suc_real_rids)
    
    preceding_ids = np.concatenate(preceding_list)
    succeeding_ids = np.concatenate(succeeding_list)
    
    # remove the dummy
    df.drop(columns=['dummy_id',])
    
    return preceding_ids, succeeding_ids

def get_pair_dataset_N_back(df, N_back):
    preceding_list = []
    succeeding_list = []
    
    # group data into block
    group_criteria = ['participant', 'sample_stage', 'block']
    grouped = df.groupby(group_criteria)

    for _, group in grouped:
        pre, suc = extract_pair_from_one_block(group, N_back)
        preceding_list.append(pre)
        succeeding_list.append(suc)
    
    preceding_df = pd.concat(preceding_list, ignore_index=True)
    succeeding_df = pd.concat(succeeding_list, ignore_index=True)
    
    return preceding_df, succeeding_df


""" functions to select data from two stimuli/responses """
def select_id_compare_func(arr, is_smaller):
    all_nans = np.all(np.isnan(arr), axis=1)
    selected = np.zeros(len(arr))
    selected[all_nans] = np.nan
    if is_smaller:
        selected[~all_nans] = np.nanargmin(arr[~all_nans, :], axis=1)
    else:
        selected[~all_nans] = np.nanargmax(arr[~all_nans, :], axis=1)
    return selected

def merge_two_row(df, des_df, col_1, col_2, selected_ids):
    new_col_name = col_1.replace('_1', '')
    col1_vals = df[col_1].to_numpy(copy=True)
    col2_vals = df[col_2].to_numpy(copy=True)
    is_nan_idx = np.isnan(selected_ids)
    merged_vals = np.zeros(len(col1_vals))
    merged_vals[is_nan_idx] = np.nan
    merged_vals[~is_nan_idx] = np.where(
        selected_ids[~is_nan_idx] == 0,
        col1_vals[~is_nan_idx],
        col2_vals[~is_nan_idx])
    des_df[new_col_name] = merged_vals
    
def get_id_by_order(df, is_earlier, by_end_time=True): 
    if is_earlier and ('early_ids' in df.columns):
        return df['early_ids'].to_numpy(copy=True)
    elif (not is_earlier) and ('early_ids' in df.columns):
        return df['late_ids'].to_numpy(copy=True)
    else:
        # decide whether to fetch response 1 or 2
        draw_time = np.array([])
        if by_end_time:
            draw_time = df[[
                'resp_1_last_drawing_tend', 
                'resp_2_last_drawing_tend']].to_numpy(copy=True)
        else:
            draw_time = df[[
                'resp_1_last_drawing_tstart', 
                'resp_2_last_drawing_tstart']].to_numpy(copy=True)

        # get the id of stim-resp pair selected 
        selected_ids = select_id_compare_func(draw_time, is_smaller=is_earlier)
    
    return selected_ids

def get_drawing_by_id_helper(df, selected_ids):
    # get the list of columns to copy
    columns = df.columns.tolist()
    columns_to_copy = []
    columns_to_merge = []
    for c in columns:
        if '1' in c:
            columns_to_merge.append((
                c, c.replace('1', '2')))
        elif '2' in c:
            pass
        else:
            columns_to_copy.append(c)
    
    # create the new df
    new_df = df[columns_to_copy].copy()
    for col_1, col_2 in columns_to_merge:
        merge_two_row(
            df, new_df, col_1, col_2, selected_ids)
        
    return new_df

def get_drawing_by_order(df, is_earlier, by_end_time=True, append_id=False):
    # get the id of stim-resp pair selected 
    selected_ids = get_id_by_order(df, is_earlier, by_end_time=by_end_time)
    new_df = get_drawing_by_id_helper(df, selected_ids)
    if append_id:
        new_df['stim_id'] = selected_ids
    return new_df

""" get the stimuli/response on left/right """
def get_drawing_by_left_right(df, is_left):
    # get the list of columns to copy
    select_side = 1 if is_left else 2
    non_select = 3 - select_side
    new_df = pd.DataFrame()
    columns = df.columns.tolist()
    for c in columns:
        if str(select_side) in c:
            new_c = c.replace(f'_{select_side}', '')
            new_df[new_c] = df[c].copy()
        elif str(non_select) in c:
            pass
        else:
            new_df[c] = df[c].copy()
    return new_df

""" function to align onset time of two response"""
def align_onset(df):
    # firstly, checked if this is aligned already
    # if yes, a 'resp_order' already exist
    df_copy = df.copy()
    if 'early_ids' in df.columns:
        warnings.warn(f'The df has already been aligned!')
    else:
        # first, record the first and last response id
        early_ids = get_id_by_order(df, is_earlier=True, by_end_time=True)
        late_ids = get_id_by_order(df, is_earlier=False, by_end_time=True)
        df_copy['early_ids'] = early_ids
        df_copy['late_ids'] = late_ids
        
        # get the onset and end time
        onset_left = df['resp_1_last_drawing_tstart'].to_numpy(copy=True)
        onset_right = df['resp_2_last_drawing_tstart'].to_numpy(copy=True)
        end_left = df['resp_1_last_drawing_tend'].to_numpy(copy=True)
        end_right = df['resp_2_last_drawing_tend'].to_numpy(copy=True)

        left_to_align = (~df['resp_1'].isna()).to_numpy(copy=True) & (end_right < onset_left)
        right_to_align = (~df['resp_2'].isna()).to_numpy(copy=True) & (end_left < onset_right)

        # align left
        onset_left[left_to_align] = onset_left[left_to_align] - end_right[left_to_align]
        end_left[left_to_align] = end_left[left_to_align] - end_right[left_to_align]

        # align right
        onset_right[right_to_align] = onset_right[right_to_align] - end_left[right_to_align]
        end_right[right_to_align] = end_right[right_to_align] - end_left[right_to_align]

        # for without delay, need to -0.5 for all onset
        no_delay_mask = (df['trial_code'] == 2).to_numpy(copy=True)
        left_no_delay_mask = no_delay_mask & (~df['resp_1'].isna()).to_numpy(copy=True) & (~left_to_align)
        right_no_delay_mask = no_delay_mask & (~df['resp_2'].isna()).to_numpy(copy=True) & (~right_to_align)
        no_delay_offset = 0.5
        onset_left[left_no_delay_mask] = onset_left[left_no_delay_mask] - no_delay_offset
        onset_right[right_no_delay_mask] = onset_right[right_no_delay_mask] - no_delay_offset
        end_left[left_no_delay_mask] = end_left[left_no_delay_mask] - no_delay_offset
        end_right[right_no_delay_mask] = end_right[right_no_delay_mask] - no_delay_offset

        # finally create a copy
        df_copy['resp_1_last_drawing_tstart'] = onset_left
        df_copy['resp_2_last_drawing_tstart'] = onset_right
        df_copy['resp_1_last_drawing_tend'] = end_left
        df_copy['resp_2_last_drawing_tend'] = end_right
    
    return df_copy

def df_pair_collapse(df):
    # original df: has data for both stims
    left_df = get_drawing_by_left_right(df, is_left=True)
    right_df = get_drawing_by_left_right(df, is_left=False)
    new_df = pd.concat([left_df, right_df], ignore_index=True)
    return new_df

import os
import shutil
import zipfile
import argparse
import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm

import sys

def get_n_dir_up(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path

CUR_PATH= os.path.abspath(__file__)
# include utils
sys.path.append(
    os.path.join(get_n_dir_up(CUR_PATH, 2)))
from utils.data import get_all_files

""" prefilter psychopy output """
def prefilter(df, exp_version='24eyetrack'):
    """ collect additional information """
    additional_info = {}
    
    """ create df and rename some attributes """
    attrs_to_collect = [
        'participant', 'session', 'mode', 'date', 'is_real_trial', 
        'sample_stage', 'stim_sample_method', 'block', 'block.thisRepN', 
        'stim_1', 'stim_2', 'stim_region_1', 'stim_region_2', 
        'trial_code',
        'stim_1_loc_x', 'stim_1_loc_y', 'stim_2_loc_x', 'stim_2_loc_y',
        'ITI',
        'display_stimuli.started', 'display_stimuli.stopped', 
        'response.started', 'response.stopped',
        'resp_1_x', 'resp_1_y', 'resp_1_time', 'resp_1_invalid',
        'resp_2_x', 'resp_2_y', 'resp_2_time', 'resp_2_invalid',
    ]
    to_rename_dict = {'block.thisRepN': 'trial'}
    if exp_version == '24eyetrack':
        attrs_to_collect.append("mode")
        attrs_to_collect += ['stim_1_to_report', 'stim_2_to_report', 'delay']
        attrs_to_collect += ['delay_post_stim', ]
        if 'TRIALID' in df:
            attrs_to_collect += ['TRIALID', ]
    else:
        raise NotImplementedError(f"unknown experiment version {exp_version}")
    
    df = df[attrs_to_collect]
    if len(to_rename_dict) > 0:
        df = df.rename(columns=to_rename_dict)
        
    """ filter out only real trials """
    df = df[df['is_real_trial'] == True]
        
    """ other modification """
    if exp_version in ['24eyetrack', ]:
        # rewrite the block id
        trial_ids = df['trial'].to_numpy()
        block_start = trial_ids == 0
        block_ids = np.cumsum(block_start) - 1
        df['block'] = block_ids
        
    """ convert data type of some """
    df['participant'] = df['participant'].astype('Int32')
    df['sample_stage'] = df['sample_stage'].astype('Int32')
    df['block'] = df['block'].astype('Int32')
    df['trial'] = df['trial'].astype('Int32')
    df['trial_code'] = df['trial_code'].astype('Int32')
    
    """ generate TRIALID if it's not avialable """
    if (exp_version == '24eyetrack') and ('TRIALID' not in df):
        df['TRIALID'] = df.apply(lambda row: f"('{row['participant']}', {row['trial']}, {row['block']})", axis=1)
    
    return df, additional_info

""" prefilter mouse data """
def prefilter_mouse(df, exp_version='24eyetrack'):
    # print(df.columns.values)
    
    """ filter out only real trials """
    df = df[df['is_real_trial'] == True]
    
    """ collect additional information """
    attrs_to_collect = ['participant', 'block', 'block.thisRepN',]
    attrs_to_collect += [
        'mouseStim.x_1', 'mouseStim.y_1', # 'stim0Start', 'stim0End',
        'mousePostStim.x_1', 'mousePostStim.y_1', 
        'mouseStim.x_2', 'mouseStim.y_2', # 'stim1End', 'stim1End',
        'mousePostStim.x_2', 'mousePostStim.y_2',
        'mouseDelay.x', 'mouseDelay.y', # 'delayStart', 'delayEnd',
    ]
    to_rename_dict = {'block.thisRepN': 'trial'}
    if exp_version == '24eyetrack':
        if 'TRIALID' in df:
            attrs_to_collect += ['TRIALID',]
    else:
        raise ValueError(f'{exp_version} unknown')
    
    # print(df.columns.values)
    
    df = df[attrs_to_collect]
    if len(to_rename_dict) > 0:
        df = df.rename(columns=to_rename_dict)
        
    """ convert data type of some """
    df['participant'] = df['participant'].astype('Int32')
    df['block'] = df['block'].astype('Int32')
    df['trial'] = df['trial'].astype('Int32')
    """ generate TRIALID if it's not avialable """
    if (exp_version == '24eyetrack') and ('TRIALID' not in df):
        df['TRIALID'] = df.apply(lambda row: f"('{row['participant']}', {row['trial']}, {row['block']})", axis=1)
    
    return df

def prescan_dataset(src_path, target_path, target_mouse_path):    
    # copy the files satisfy the requirement to target folder
    min_kb = 60
    all_files = get_all_files(src_path)
    for f in tqdm(all_files):
        # should be a csv file
        if not f.endswith('.csv'):
            continue
        # should be at least 40 kb
        f_size_bytes = os.path.getsize(f)
        if f_size_bytes / 1024 < min_kb:
            continue
    
        # otherwise, copy the file to target
        # first read the file and simplify it
        df = pd.read_csv(f)
        behavior_df, additional_info = prefilter(df, exp_version)
        mouse_df = prefilter_mouse(df, exp_version)
        base_name = os.path.basename(f)
        fdes_path = os.path.join(target_path, base_name)
        behavior_df.to_csv(fdes_path)
        mouse_fdes_path = os.path.join(target_mouse_path, base_name)
        mouse_df.to_csv(mouse_fdes_path)
        
        # save additional info
        if additional_info:
            addinfo_path = os.path.join(target_path, f'{base_name}.pkl')
            with open(addinfo_path, 'wb') as fp:
                pickle.dump(additional_info, fp)

DEFAULT_DATA_SRC_PATH = os.path.join(
    get_n_dir_up(CUR_PATH, 3), 'data', 'psychopy_raw', 'original')
DEFAULT_DATA_TARGET_PATH = os.path.join(
    get_n_dir_up(CUR_PATH, 3), 'data', 'psychopy_raw', 'filtered')
DEFAULT_DATA_MOUSE_TARGET_PATH = os.path.join(
    get_n_dir_up(CUR_PATH, 3), 'data', 'psychopy_raw', 'filtered_mouse')


if __name__ == '__main__':
    # read input and output
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', default=DEFAULT_DATA_SRC_PATH)
    parser.add_argument('--target_path', default=DEFAULT_DATA_TARGET_PATH)
    parser.add_argument('--target_mouse_path', default=DEFAULT_DATA_MOUSE_TARGET_PATH)
    parser.add_argument('--version', default='24eyetrack')
    args = parser.parse_args()
    src_path = args.src_path
    target_path = args.target_path
    target_mouse_path = args.target_mouse_path
    exp_version = args.version
    prescan_dataset(src_path, target_path, target_mouse_path)

import os,shutil
import argparse
import numpy as np
import pandas as pd

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

""" process filtered data """
from utils.drawing_analysis import (
    # for drawing
    parse_drawing_list_string,
    keep_last_drawing, join_drawing, filter_start_end,
    compute_ori_start_to_end, 
    compute_curvature,
    # for clicking
    parse_click_data,
    compute_click_ori,
)

""" extract the stimulus and response data for drawing """
def extract_resp_draw(df, resp_id, draw_process_method=compute_ori_start_to_end):
    """ read raw data """
    x_raw = df[f'resp_{resp_id}_x'].apply(parse_drawing_list_string).to_list()
    y_raw = df[f'resp_{resp_id}_y'].apply(parse_drawing_list_string).to_list()
    t_raw = df[f'resp_{resp_id}_time'].apply(parse_drawing_list_string).to_list()
    
    """ filterng """
    last_drawings = [keep_last_drawing(x, y, t) for x, y, t in zip(x_raw, y_raw, t_raw)]
    last_x, last_y, last_t = zip(*last_drawings)
    start_p, end_p = 0.1, 0.9
    join_x, join_y, join_t = zip(*[join_drawing(x, y, t) for x, y, t in zip(
        last_x, last_y, last_t)])
    filtered_drawings = [filter_start_end(x, y, t, start_p, end_p) for x, y, t in zip(
        join_x, join_y, join_t)]
    filtered_x, filtered_y, filtered_t = zip(*filtered_drawings)
    
    """ compute orientations """
    oris = [draw_process_method(x, y) for x, y in zip(filtered_x, filtered_y)]
    
    """ compute curvature """
    curvs = [compute_curvature(x, y, t) for x, y, t in zip(filtered_x, filtered_y, filtered_t)]
    
    """ compute other t information """
    n_attempts = [len(d) for d in t_raw]
    lambda_draw_tstart = lambda ts: (ts[0] if len(ts) > 0 else None)
    lambda_draw_tend = lambda ts: (ts[-1] if len(ts) > 0 else None)
    last_drawing_tstart = [lambda_draw_tstart(ts) for ts in join_t]
    last_drawing_tend = [lambda_draw_tend(ts) for ts in join_t]
    last_drawing_t = [
        a - b if a is not None and b is not None else None 
        for a, b in zip(last_drawing_tend, last_drawing_tstart)]
    
    """ if response is nan, set time-info as nan  """
    for i in range(len(oris)):
        if oris[i] is None or np.sum(np.isnan(oris[i])) > 0:
            last_drawing_tstart[i] = None
            last_drawing_tend[i] = None
            last_drawing_t[i] = None
    
    draw_time_info = {
        'n_attempts': n_attempts,
        'last_drawing_tstart': last_drawing_tstart,
        'last_drawing_tend': last_drawing_tend,
        'last_drawing_t': last_drawing_t,
        'last_drawing_curv': curvs,
    }
    return oris, draw_time_info

""" extract the stimulus and response data for clicking """
def extract_resp_click(df, resp_id, click_process_method=compute_click_ori):
    # xy_to_ori(x1, x2, y1, y2)
    """ read raw data """
    x_raw = df[f'resp_{resp_id}_x'].apply(parse_click_data).to_list()
    y_raw = df[f'resp_{resp_id}_y'].apply(parse_click_data).to_list()
    t_raw = df[f'resp_{resp_id}_time'].apply(parse_click_data).to_list()
    
    lambda_get_first = lambda xs: (xs[0] if len(xs) > 0 else None)
    lambda_get_last = lambda xs: (xs[-1] if len(xs) > 0 else None)
    
    """ analyze only the last click """
    filtered_x = [lambda_get_last(x) for x in x_raw]
    filtered_y = [lambda_get_last(y) for y in y_raw]
    filtered_t = [lambda_get_last(t) for t in t_raw]
    
    """ compute orientations """
    oris = [click_process_method(x, y) for x, y in zip(filtered_x, filtered_y)]
    
    """ compute other t information """
    n_attempts = [len(d) for d in t_raw]
    last_drawing_tstart = [lambda_get_first(ts) for ts in t_raw]
    last_drawing_tend = [lambda_get_last(ts) for ts in t_raw]
    last_drawing_t = [
        a - b if a is not None and b is not None else None 
        for a, b in zip(last_drawing_tend, last_drawing_tstart)]
    
    """ if response is nan, set time-info as nan  """
    for i in range(len(oris)):
        if oris[i] is None or np.sum(np.isnan(oris[i])) > 0:
            last_drawing_tstart[i] = None
            last_drawing_tend[i] = None
            last_drawing_t[i] = None
    
    draw_time_info = {
        'n_attempts': n_attempts,
        'last_drawing_tstart': last_drawing_tstart,
        'last_drawing_tend': last_drawing_tend,
        'last_drawing_t': last_drawing_t,
    }
    return oris, draw_time_info


def compute_display_region(df):
    # 1: displayed at the upper half
    # -1: displayed at the lower half
    y_loc = df[f'stim_1_loc_y'].to_list()
    y_loc_sign = np.sign(y_loc).tolist()
    return y_loc_sign

def compute_RT(df):
    trial_code = df['trial_code'].to_numpy()
    simult = trial_code == 2
    display_t = (df['display_stimuli.stopped'] - df['display_stimuli.started']).to_numpy()
    response_t = (df['response.stopped'] - df['response.started']).to_numpy()
    rt = np.where(simult, display_t, response_t).tolist()
    return rt


""" specifically for errors """
def smart_diff_ori(x1, x2, r=180):
    d = x1 - x2
    d = (d + r/2) % r - r/2
    return d

def compute_error(df, resp_id):
    stims = df[f'stim_{resp_id}'].to_numpy()
    resps = df[f'resp_{resp_id}'].to_numpy()
    valid = (~np.isnan(stims.astype(float))) & (~np.isnan(resps.astype(float)))
    errs = np.full_like(stims, np.nan)
    errs[valid] = smart_diff_ori(resps[valid], stims[valid])
    errs = errs.tolist()
    return errs

""" convert and compress raw psychopy outputs to more readable data """
def process_raw_data(df, version):
    """ read trial information """
    to_copy_columns = [
        'participant', 'mode',
        'sample_stage', 'stim_sample_method', 'block', 'trial', 
        'trial_code', 'ITI',
    ]
    result_df = df[to_copy_columns].copy()
    
    for i in [1, 2]:
        stim_columns = [f'stim_{i}', f'stim_region_{i}', f'stim_{i}_to_report']
        result_df[stim_columns] = df[stim_columns].copy()
    # there is only one delay
    result_df['delay'] = df['delay'].copy() 
    
    # but for eyetracking, record the delay after first stim
    if version == '24eyetrack':
        result_df['delay_post_stim'] = df['delay_post_stim'].copy()
        result_df['TRIALID'] = df['TRIALID'].copy()
    else:
        raise ValueError(f"unknown version {version}")
    
    """ compute display region """
    display_region = compute_display_region(df)
    result_df['display_region'] = display_region
    
    """ compute RT """
    RTs = compute_RT(df)
    result_df['RT'] = RTs
    
    extract_method_mapping = {
        'draw': extract_resp_draw,
        'click': extract_resp_click,
    }
    
    for mode in extract_method_mapping:
        mode_mask = (df['mode'] == mode).to_numpy()
        if len(df[mode_mask]) > 0:
            extract_method = extract_method_mapping[mode]
            all_resp_idx = [0, 1, 2] if version == '24may' else [1, 2]
            """ compute response """
            for resp_idx in all_resp_idx:
                # convert drawing to response
                resps, resp_time_info = extract_method(df[mode_mask], resp_idx)
                result_df.loc[mode_mask, f'resp_{resp_idx}'] = resps
                for kname in resp_time_info:
                    kname_converted = f'resp_{resp_idx}_{kname}'
                    if kname_converted not in result_df:
                        result_df[kname_converted] = None # np.nan
                    result_df.loc[mode_mask, kname_converted] = resp_time_info[kname]
                
                # compute error 
                errors = np.array(compute_error(result_df[mode_mask], resp_idx))
                result_df.loc[mode_mask, f'err_{resp_idx}'] = errors
    
    return result_df


""" convert the whole dataset """
def preprocess_dataset(src_folder, des_folder):
    if os.path.exists(des_folder):
        shutil.rmtree(des_folder)
    os.makedirs(des_folder)
    
    for l in tqdm(os.listdir(src_folder)):
        if l.endswith('.csv'):
            # only process files in csv format
            src_path = os.path.join(src_folder, l)
            df = pd.read_csv(src_path)
            df = process_raw_data(df, exp_version)
            participant_id = df['participant'].iloc[0]
            des_path = os.path.join(des_folder, f'{participant_id}.csv')
            df.to_csv(des_path)

def get_n_dir_up(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path

DEFAULT_DATA_SRC_PATH = os.path.join(
    get_n_dir_up(CUR_PATH, 3), 'data', 'psychopy_raw', 'filtered')
DEFAULT_DATA_TARGET_PATH = os.path.join(
    get_n_dir_up(CUR_PATH, 3), 'data', 'behavior', 'subjects')

if __name__ == '__main__':
    # read input and output
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_folder', default=DEFAULT_DATA_SRC_PATH)
    parser.add_argument('--des_folder', default=DEFAULT_DATA_TARGET_PATH)
    parser.add_argument('--version', default='24eyetrack')
    args = parser.parse_args()
    src_folder = args.src_folder
    des_folder = args.des_folder
    exp_version = args.version
    preprocess_dataset(src_folder, des_folder)

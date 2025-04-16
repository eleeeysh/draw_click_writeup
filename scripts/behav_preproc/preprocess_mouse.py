import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm

""" process mouse data list """
def parse_list(s):
    if len(s) == 0:
        return []
    s = s.replace("[", "").replace("]", "")
    stroke = s.split(',')
    stroke = [float(x) for x in stroke]
    return stroke

def list_to_pixel(data, axis):
    RADIUS = 540
    CENTER = 540 if axis == 'y' else 960
    data = np.array(data) * RADIUS * 2 + CENTER
    data = data.astype(int)
    return data

def time_align(lst, length, frequency):
    # actual timing of recording
    n_lst_timing = len(lst)
    lst_timing = np.arange(n_lst_timing) / n_lst_timing
    # timing to align with
    n_actual_timing = length * frequency
    actual_timing = np.arange(n_actual_timing) / n_actual_timing
    # interpolation
    est_pos = np.interp(actual_timing, lst_timing, lst)
    return est_pos

EXP_SETTINGS = [
    {
        'phase': 'display_1',
        'x_name': 'mouseStim.x_1',
        'y_name': 'mouseStim.y_1',
        'length': 0.75,
    },
    {
        'phase': 'post_stim_1',
        'x_name': 'mousePostStim.x_1',
        'y_name': 'mousePostStim.y_1',
        'length': 1.5,
    },
    {
        'phase': 'display_2',
        'x_name': 'mouseStim.x_2',
        'y_name': 'mouseStim.y_2',
        'length': 0.75,
    },
    {
        'phase': 'post_stim_2',
        'x_name': 'mousePostStim.x_2',
        'y_name': 'mousePostStim.y_2',
        'length': 1.5,
    },
    {
        'phase': 'delay',
        'x_name': 'mouseDelay.x',
        'y_name': 'mouseDelay.y',
        'length': 4.0,
    },
    
]
FREQUENCY = 60

def convert_csv_to_motion_npy(df, settings, result_path):
    results = {}
    for ss in settings:
        xdata = df[ss['x_name']].apply(
            parse_list).apply(
            list_to_pixel, args=('x',)).apply(
            time_align, args=(ss['length'], FREQUENCY))
        xdata = np.array(xdata.to_list())
        ydata = df[ss['y_name']].apply(
            parse_list).apply(
            list_to_pixel, args=('y',)).apply(
            time_align, args=(ss['length'], FREQUENCY))
        ydata = np.array(ydata.to_list())
        results[f"{ss['phase']}.x"] = xdata
        results[f"{ss['phase']}.y"] = ydata
        
    # also save the frequency
    results['frequency'] = FREQUENCY
    np.savez_compressed(result_path, **results)


def convert_all_mouse_data(source_folder, result_folder):
    os.makedirs(result_folder, exist_ok=True)
    for f in tqdm(os.listdir(source_folder)):
        if f.endswith('.csv'):
            source_csv = os.path.join(source_folder, f)
            source_df = pd.read_csv(source_csv)
            subj = os.path.splitext(f)[0]
            subj = subj.split('_')[0]
            result_path = os.path.join(result_folder, f'{subj}.npz')
            convert_csv_to_motion_npy(source_df, EXP_SETTINGS, result_path)

""" default paths """
def get_n_dir_up(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path

CUR_PATH= os.path.abspath(__file__)

DEFAULT_DATA_SRC_PATH = os.path.join(
    get_n_dir_up(CUR_PATH, 3), 'data', 'psychopy_raw', 'filtered_mouse')
DEFAULT_DATA_TARGET_PATH = os.path.join(
    get_n_dir_up(CUR_PATH, 3), 'data', 'mouse', 'compressed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_folder', default=DEFAULT_DATA_SRC_PATH)
    parser.add_argument('--des_folder', default=DEFAULT_DATA_TARGET_PATH)
    args = parser.parse_args()
    src_folder = args.src_folder
    des_folder = args.des_folder
    convert_all_mouse_data(src_folder, des_folder)

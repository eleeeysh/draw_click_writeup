import os
import pandas as pd

""" load all files within a directory """
def get_all_files(p):
    if os.path.isfile(p):
        return [p,]
    else:
        files = []
        for d in os.listdir(p):
            new_d = os.path.join(p, d)
            new_d_files = get_all_files(new_d)
            files += new_d_files
        return files
    
""" combine data from one or multiple directory """
def load_dataset(ds_names, subfolder='preprocessed'):
    super_df_list = []
    for ds_path in ds_names:
        ffolder = os.path.join(ds_path, subfolder)
        all_fs = os.listdir(ffolder)
        all_fs.sort()
        for l in all_fs:
            fpath = os.path.join(ffolder, l)
            df = pd.read_csv(fpath, index_col=0)
            super_df_list.append(df)
    super_df = pd.concat(super_df_list, ignore_index=True)
    super_df = super_df.reset_index(drop=True)
    return super_df
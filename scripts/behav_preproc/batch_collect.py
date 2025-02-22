import os
import json
import pandas as pd

CUR_PATH= os.path.abspath(__file__)
def get_n_dir_up(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path

DATA_PATH = os.path.join(get_n_dir_up(CUR_PATH, 3), 'data')
SRC_PATH = os.path.join(DATA_PATH, 'behavior', 'subjects')
TARGET_PATH = os.path.join(DATA_PATH, 'behavior', 'batches')
BATCH_PATH = os.path.join(DATA_PATH, 'batch.json')

def main():
    with open(BATCH_PATH, 'r') as f:
        batch = json.load(f)
        for batch_name, batch_subjs in batch.items():
            batch_collected = []
            for subj in batch_subjs:
                subj_path = os.path.join(SRC_PATH, f'{subj}.csv')
                subj_df = pd.read_csv(subj_path, index_col=0)
                batch_collected.append(subj_df)
            batch_df = pd.concat(batch_collected, ignore_index=True)
            batch_df = batch_df.reset_index(drop=True)
            batch_des_path = os.path.join(TARGET_PATH, f'{batch_name}.csv')
            batch_df.to_csv(batch_des_path)

if __name__ == '__main__':
    main()

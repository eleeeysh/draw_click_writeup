import os
import numpy as np
import pandas as pd

""" read data in the form of participant, trialid + npdata """
class TrialData:
    def __init__(self, data, trial_ids):
        self.trial_ids = np.array(trial_ids)
        self.trial_index_map = {trial_id: index for index, trial_id in enumerate(self.trial_ids)}
        self.data = data
        
        # some meta data
        self.n_trials = len(self.trial_ids)
        
    def read(self, trial_ids=None, get_trial_mask=False):
        # fetch the trials
        fetched_data = self.data[:]
        if trial_ids is None:
            trial_ids = self.trial_ids
        
        # Note that some eyetracking data has been included, so no guarantee all 
        trial_mask = np.array([tname in self.trial_index_map for tname in trial_ids])
        trial_idx_mapping = [self.trial_index_map[tname] for tname in np.array(trial_ids)[trial_mask]]
        fetched_data = fetched_data[trial_idx_mapping]
        
        if get_trial_mask:
            return fetched_data, trial_mask
        else:
            return fetched_data
    
    def groupby(self, base_df, by):
        results = []
        for group, groupdf in base_df.groupby(by=by):
            group_trial_ids = groupdf['TRIALID'].to_numpy()
            group_loaded, id_mask = self.read(group_trial_ids, get_trial_mask=True)
            results.append((
                group, group_trial_ids[id_mask], group_loaded, 
            ))
        return results

""" readin 2D data """
class XYData(TrialData):
    def __init__(self, x_data, y_data, trial_ids):
        stacked = np.stack((x_data, y_data), axis=-1)
        super().__init__(stacked, trial_ids)
        
        # some meta data
        self.n_time = x_data.shape[1]
        
    def read(self, trial_ids=None, min_time=None, max_time=None, get_trial_mask=False):
        # fetch the trials
        fetched, fetched_id_mask = super().read(trial_ids=trial_ids, get_trial_mask=True)
        fetched_x, fetched_y = fetched[:, :, 0], fetched[:, :, 1]
            
        # fetch the data within a time range
        min_time = 0 if min_time is None else min_time
        max_time = self.n_time if max_time is None else max_time
        if len(fetched_x) > 0:
            # should be non-empty
            fetched_x = fetched_x[:, min_time:max_time]
            fetched_y = fetched_y[:, min_time:max_time]

        if get_trial_mask:
            return fetched_x, fetched_y, fetched_id_mask
        else:
            return fetched_x, fetched_y
    
    def read_phase(self, trial_ids=None, phases=[(None, None)], get_trial_mask=False):
        result_xs, result_ys = [], []
        mask = None
        for phase in phases:
            new_xs, new_ys, mask = self.read(
                trial_ids=trial_ids, min_time=phase[0], max_time=phase[1], get_trial_mask=True)  
            result_xs.append(new_xs)
            result_ys.append(new_ys)
        result_xs = np.concatenate(result_xs, axis=1)
        result_ys = np.concatenate(result_ys, axis=1)
        
        if get_trial_mask:
            return result_xs, result_ys, mask
        else:
            return result_xs, result_ys
    
    def groupby(self, base_df, by, phases=[(None, None)]):
        results = []
        for group, groupdf in base_df.groupby(by=by):
            group_trial_ids = groupdf['TRIALID'].to_numpy()
            group_xs, group_ys, id_mask = self.read_phase(group_trial_ids, phases=phases, get_trial_mask=True)
            results.append((
                group, group_trial_ids[id_mask], (group_xs, group_ys), 
            ))
        return results

def create_xydata_from_df(df):
    x_data = df[df['axis']=='x']
    y_data = df[df['axis']=='y']

    # resort
    x_data = x_data.sort_values(by='TRIALID')
    x_data.reset_index(drop=True, inplace=True)
    y_data = y_data.sort_values(by='TRIALID')
    y_data.reset_index(drop=True, inplace=True)

    # stor the data
    trial_ids = x_data['TRIALID'].to_list()
    trial_index_map = {trial_id: index for index, trial_id in enumerate(trial_ids)}

    numeric_columns = [col for col in df.columns if col.isdigit()]

    x_data = x_data[numeric_columns].to_numpy()
    y_data = y_data[numeric_columns].to_numpy()
    
    return XYData(x_data, y_data, trial_ids)


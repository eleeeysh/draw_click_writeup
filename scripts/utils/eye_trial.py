""" process experiment setting related data """

from collections import OrderedDict

""" settings """
TRIAL_START_TIME = 1000
CUE_ONSET = 250
STIM_DURATION = 750
DELAY_MASK_DURATION = 500
SHORT_DELAY = 1000
LONG_DELAY = 5000

def generate_events():
    s1_onset = TRIAL_START_TIME
    s1_cue_onset = s1_onset + CUE_ONSET
    s1_delay_mask_onset = s1_onset + STIM_DURATION
    s1_delay_onset = s1_delay_mask_onset + DELAY_MASK_DURATION
    s2_onset = s1_delay_onset + SHORT_DELAY
    s2_cue_onset = s2_onset + CUE_ONSET
    s2_delay_mask_onset = s2_onset + STIM_DURATION
    s2_delay_onset = s2_delay_mask_onset + DELAY_MASK_DURATION
    resp_onset = s2_delay_onset + LONG_DELAY
    
    events = OrderedDict()
    events['s1 onset'] = s1_onset
    events['s1 cue onset'] = s1_cue_onset
    events['s1 delay mask onset'] = s1_delay_mask_onset
    events['s1 delay onset'] = s1_delay_onset
    events['s2 onset'] = s2_onset
    events['s2 cue onset'] = s2_cue_onset
    events['s2 delay mask onset'] = s2_delay_mask_onset
    events['s2 delay onset'] = s2_delay_onset
    events['response'] = resp_onset
    
    return events

def generate_stim_phases(events):
    all_stim_phases = []
    for sid in [1, 2]:
        phases = OrderedDict()
        end_name = 's2 onset' if sid == 1 else 'response'
        phases[f'display'] = (events[f's{sid} onset'], events[f's{sid} delay mask onset'])
        phases[f'display with cue'] = (events[f's{sid} cue onset'], events[f's{sid} delay mask onset'])
        phases[f'delay'] = (events[f's{sid} delay mask onset'], events[end_name])
        phases[f'delay no mask'] = (events[f's{sid} delay onset'], events[end_name])

        all_stim_phases.append(phases)
    return all_stim_phases

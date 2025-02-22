import numpy as np
from .eye_data import XYData

""" movement data to vector """
def inspect_movement_data(angle, mag):
    # distribution of magnitude
    print(f"[magnitude] mean: {np.mean(mag):.2f}, median: {np.median(mag, 0.9):.2f}")
    print(f"[magnitude] 0.1: {np.quantile(mag, 0.1):.2f}, 0.9: {np.quantile(mag, 0.9):.2f}")

    # TODO: distribution of angle
    pass

def convert_movement_to_angle_only(xs, ys, epoch=360, stim_align=True):
    # convert to angles
    angle = np.rad2deg(np.arctan2(ys, xs))
    angle = angle.astype(int)
    
    # note that we want it to align with stimuli input angle format
    if stim_align:
        angle = 90 - angle
        angle = angle % epoch
    
    return angle
    

def convert_movement_to_angle(xs, ys, epoch=360, compute_mag=False, stim_align=True):
    # convert to angles
    angle = convert_movement_to_angle_only(xs, ys, epoch=epoch, stim_align=stim_align)
    
    if compute_mag:
        # compute magnitude
        mag = np.linalg.norm(np.array([xs, ys]), axis=0)
        return angle, mag
    else:
        return angle

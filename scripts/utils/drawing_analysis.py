import numpy as np
from itertools import chain
from scipy.signal import gaussian
from scipy.ndimage import convolve1d

""" parse raw drawing data """


def parse_one_stroke_data(s):
    if len(s) == 0:
        return []
    # format: x1,x2,x3...xk
    stroke_data = s.split(',')
    stroke_data = [float(x) for x in stroke_data]
    return stroke_data


def parse_one_drawing(s):
    # format [stroke1],[stroke2]
    if len(s) == 0:
        return []
    # s = s.strip('[]')
    s = s[1:-1]
    s = s.replace('],[', '#')
    s = s.replace('], [', '#') # version diff?
    strokes = s.split('#')
    strokes = [parse_one_stroke_data(x) for x in strokes]
    return strokes


def parse_drawing_list_string(s):
    # for invalid inputs
    if (not isinstance(s, str)) or (len(s) == 0):
        return []
    
    drawings = []
    n_left_minus_right_bracket = 0
    processing_drawing = False
    temp_s = ''
    for x in s:
        if x == '[':
            n_left_minus_right_bracket += 1
            if not processing_drawing:
                if n_left_minus_right_bracket == 2:
                    # start aggregatin the string for a new drawing
                    temp_s = ''
                    processing_drawing = True
            else:
                temp_s = temp_s + x
        elif x == ']':
            n_left_minus_right_bracket -= 1
            if processing_drawing:
                if n_left_minus_right_bracket == 1:
                    # process the string of drawing
                    drawing = parse_one_drawing(temp_s)
                    drawings.append(drawing)
                    processing_drawing = False
                else:
                    temp_s = temp_s + x
        else:
            if processing_drawing:
                temp_s = temp_s + x
    return drawings

def parse_click_data(s):
    # for invalid inputs
    if (not isinstance(s, str)) or (len(s) == 0):
        return []
    return parse_one_stroke_data(s[1:-1])

""" data filtering """


def keep_last_drawing(xs, ys, ts):
    if len(ts) == 0:
        return [], [], []
    else:
        return xs[-1], ys[-1], ts[-1]


def join_drawing(draw_x, draw_y, draw_time):
    all_time = np.array(list(chain(*draw_time)))
    all_xs = np.array(list(chain(*draw_x)))
    all_ys = np.array(list(chain(*draw_y)))
    return all_xs, all_ys, all_time


def filter_start_end(all_xs, all_ys, all_time, start_perc, end_perc):
    # start_perc: % time to remove at the start
    # end_perc: % time to removew from the end
    # return: the filtered
    
    if len(all_time) < 2:
        # no all only one points are considered invalid
        return [], [], []
    
    # compute the total time
    total_t = all_time[-1] - all_time[0]
    start_t = all_time[0] + total_t * start_perc
    end_t = all_time[0] + total_t * end_perc
    valid_mask = (all_time >= start_t) & (all_time <= end_t)
    
    filtered_x = []
    filtered_y = []
    filtered_t = []
    if np.sum(valid_mask) >= 2:
        filtered_x = all_xs[valid_mask]
        filtered_y = all_ys[valid_mask]
        filtered_t = all_time[valid_mask]
    return filtered_x, filtered_y, filtered_t


""" convert drawing to response """

# +
""" convert to angle (scalar) """
def round_degree(deg):
    deg = np.round(deg).astype(int)
    deg = deg % 180
    return deg

def compute_vector_ori(x_resp, y_resp):
    # compute all the rest
    rad = np.arctan2(y_resp, x_resp)
    deg = np.rad2deg(rad)
    deg = round_degree(deg)
    deg = 90 - deg
    deg = round_degree(deg)
    return deg

def xy_to_ori(x1, x2, y1, y2):
    x_resp = x2 - x1
    y_resp = y2 - y1
    return compute_vector_ori(x_resp, y_resp)

def compute_ori_start_to_end(xs, ys):
    # convert response to orientation
    if len(xs) >= 2:    
        # the easiest way: start to end
        deg = xy_to_ori(xs[0], xs[-1], ys[0], ys[-1])
        return deg
    else:
        return None
    
def compute_click_ori(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return compute_vector_ori(x, y)

""" convert to angle (distribution) """
def compute_distrib_of_ori(xs, ys):
    distrib = np.zeros(180)
    if len(xs) >= 2:
        start_x, start_y = xs[:-1], ys[:-1]
        end_x, end_y = xs[1:], ys[1:]
        oris = xy_to_ori(start_x, end_x, start_y, end_y)
        dists = np.sqrt((end_x-start_x)**2+(end_y-start_y)**2)
        weights = dists / np.sum(dists)
        one_enc = np.eye(180)[oris]
        
        if len(xs) == 2:
            distrib = one_enc
        else:
            distrib = np.dot(weights, one_enc) 
        distrib = distrib / np.sum(distrib)

    return distrib


""" convert to trajectory """
def compute_approx_trajectory(xs, ys, n_points=10):
    results = np.zeros(n_points)
    if len(xs) >= 2:
        sample_indices = np.linspace(
            0, len(xs)-1, n_points+1, endpoint=True)
        start_indices = np.floor(sample_indices).astype(int)[:-1]
        end_indices = np.ceil(sample_indices).astype(int)[1:]
        start_x, start_y = xs[start_indices], ys[start_indices]
        end_x, end_y = xs[end_indices], ys[end_indices]
        oris = xy_to_ori(start_x, end_x, start_y, end_y)
        results = np.array(oris)
    return results

""" approximate the curvature of trajectory """
def compute_curvature(xs, ys, ts):
    min_len = 3
    if len(xs) < min_len:
        return None
    else:
        # Compute first derivatives
        dx = np.gradient(xs, ts)
        dy = np.gradient(ys, ts)
    
        # Compute second derivatives
        ddx = np.gradient(dx, ts)
        ddy = np.gradient(dy, ts)
    
        # Calculate curvature using the formula
        curvature = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**(3/2) + 1e-5)

        # compute the average curvature
        cur_weights = np.ones(len(curvature))
        cur_weights[0] = 0.5
        cur_weights[-1] = 0.5
        avg_cur = np.sum(cur_weights * curvature) / np.sum(cur_weights)
        
        # Convert to a list if needed
        return avg_cur
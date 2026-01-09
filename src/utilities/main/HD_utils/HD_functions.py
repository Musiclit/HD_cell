'''
Functions used to analyze zebrafish data
'''
import numpy as np
import os
import matplotlib.pyplot as plt
# import lotr.plotting as pltltr
from lotr.behavior import get_bouts_props_array
from lotr.utils import convolve_with_tau
from lotr.default_vals import REGRESSOR_TAU_S, TURN_BIAS
# COLS = pltltr.COLS

from tqdm import tqdm
import pandas as pd


def subtract_preferred_HD_activity(data_df, stat_df):
    """
    Subtract the part of the activity modulated by preferred HD from the activity.
    Returns a DataFrame with a new column 'activity_subtract'.
    """
    beta3_lookup = stat_df.set_index(['fish', 'cell'])['beta3'].to_dict()

    data_df['beta3'] = data_df.set_index(['fish', 'cell']).index.map(beta3_lookup).values
    data_df['activity_subtract'] = data_df['activity'] - data_df['beta3'] * data_df['cos_diff']
    
    data_df.drop(columns=['beta3'], inplace=True)
    return data_df


def select_cell(lm_noHD, min_dist, max_dist, data_noHD_acv):
    lm_noHD_s = lm_noHD.loc[(lm_noHD.dist2HD <= max_dist) & (lm_noHD.dist2HD > min_dist), ['fish', 'cell', 'ring', 'x', 'y', 'z', 'dist2HD']]

    fish_ids = data_noHD_acv['fish'].unique()
    df_list = []
    for fishi in tqdm(fish_ids):
        cell_ids = lm_noHD_s.loc[lm_noHD_s['fish'] == fishi, 'cell'].unique()
        data_noHD_acv_fish = data_noHD_acv.loc[(data_noHD_acv['fish'] == fishi)]
        for celli in cell_ids:
            data_noHD_acv_cell = data_noHD_acv_fish.loc[(data_noHD_acv_fish['cell'] == celli)]
            if len(data_noHD_acv_cell) > 0:
                df_list.append(data_noHD_acv_cell)
                
    data_noHD_acv_s = pd.concat(df_list, ignore_index=True)
    return data_noHD_acv_s

def add_angvel__ring_to_acv_df(data_noHD_acv_s, lm_noHD_s, data_noHD_vel):
    data_noHD_all = data_noHD_acv_s.merge(lm_noHD_s, on=['fish', 'cell'], how='inner')

    fish_ids = data_noHD_all['fish'].unique()
    df_list = []
    for fishi in tqdm(fish_ids):
        data_fish = data_noHD_all.loc[data_noHD_all['fish'] == fishi]
        vel_fish = data_noHD_vel.loc[data_noHD_vel['fish'] == fishi]
        cell_ids = data_fish['cell'].unique()
        for celli in cell_ids:
            data_cell = data_fish.loc[data_fish['cell'] == celli]
            data_cell2 = pd.concat([data_cell.reset_index(drop=True), vel_fish.drop(columns='fish').reset_index(drop=True)], axis=1)
            data_cell2 = data_cell2.dropna()
            if len(data_cell2) > 0:
                df_list.append(data_cell2)
                
    data_noHD = pd.concat(df_list, ignore_index=True)
    
    return data_noHD

def cal_mean_acv_per_bin(data_noHD, nbins):
    data_sum = data_noHD.groupby(['fish', 'ring', 'angVel_bin']).agg({'activity': 'mean'}).reset_index()
    data_sum2 = data_noHD.loc[data_noHD.angVel_bin == (nbins+1)//2].groupby(['fish']).agg({'activity': 'mean'}).reset_index()
    data_sum2 = data_sum2[['fish', 'activity']]

    data_sum = pd.merge(data_sum, data_sum2, on=['fish'], how='left', suffixes=('', '_mean'))
    data_sum['activity_norm'] = data_sum.activity / data_sum.activity_mean
    
    return data_sum

def separate_data_into_each_AHV_condition(exp):
    # Preparation
    theta_turned = get_bouts_props_array(exp.n_pts, exp.bouts_df, min_bias=TURN_BIAS, selection="all", value="bias")
    theta_forward = get_bouts_props_array(exp.n_pts, exp.bouts_df, min_bias=0, selection="forward", value="bias")
    theta_all = get_bouts_props_array(exp.n_pts, exp.bouts_df, min_bias=0, selection="all", value="bias")
    fictive_head = convolve_with_tau(np.cumsum(theta_turned), REGRESSOR_TAU_S * exp.fn)
    velocity = convolve_with_tau(theta_turned, REGRESSOR_TAU_S * exp.fn)
    movement = convolve_with_tau(theta_all, REGRESSOR_TAU_S * exp.fn)
    movement_forward = convolve_with_tau(theta_forward, REGRESSOR_TAU_S * exp.fn)
    
    ## Indexs for each condition: Crude separation
    threshold = 0.0001
    index_CW = np.where(velocity > threshold)[0]
    index_CCW = np.where(velocity < -threshold)[0]
    median_CW = np.median(velocity[index_CW])
    median_CCW = np.median(velocity[index_CCW])
    
    index_stable = np.where( np.isclose(movement, 0, atol=threshold) )[0]
    index_forawrd = np.where( np.isclose(movement_forward, 0, atol=threshold) )[0]
    
    # Stable duration
    index2th2_begining_of_stable = np.concatenate([[0], np.where( np.diff(index_stable) > 1 )[0] + 1])
    index2th2_end_of_stable = np.concatenate([np.where( np.diff(index_stable) > 1 )[0], [len(index_stable)-1]])
    stable_duration = np.zeros(len(fictive_head))
    for i, start in enumerate(index2th2_begining_of_stable):
        index_start_stable_t = index_stable[start]
        index_end_stable_t = index_stable[index2th2_end_of_stable[i]]
        stable_duration[index_start_stable_t:index_end_stable_t+1] = np.arange(index_end_stable_t - index_start_stable_t + 1)
    median_stable_duration = np.median(stable_duration[index_stable])
    ## Separate for each condition
    index_CW_fast = np.where(velocity > median_CW)[0]
    index_CW_slow = np.where( (velocity > 0.0001) & (velocity < median_CW) )[0]
    index_CCW_fast = np.where(velocity < median_CCW)[0]
    index_CCW_slow = np.where( (velocity < -0.0001) & (velocity > median_CCW) )[0]
    index_stable_early = np.where( (stable_duration > 0) & (stable_duration < median_stable_duration))[0]
    index_stable_late = np.where( stable_duration > median_stable_duration)[0]

    cond_id_crude = [index_CW, index_CCW, index_stable, index_forawrd] # CW, CCW, Stable
    cond_id_fine = [index_CW_fast, index_CW_slow, index_CCW_fast, index_CCW_slow, index_stable_early, index_stable_late]
    cond_median = [median_CW, median_CCW, median_stable_duration]

    return cond_id_crude, cond_id_fine, cond_median, fictive_head, velocity, stable_duration

def separate_data_into_each_AHV_condition2(exp):
    # Preparation
    theta_turned = get_bouts_props_array(exp.n_pts, exp.bouts_df, min_bias=TURN_BIAS, selection="all", value="bias")
    theta_forward = get_bouts_props_array(exp.n_pts, exp.bouts_df, min_bias=0, selection="forward", value="bias")
    theta_all = get_bouts_props_array(exp.n_pts, exp.bouts_df, min_bias=0, selection="all", value="bias")

    fictive_head = convolve_with_tau(np.cumsum(theta_turned), REGRESSOR_TAU_S * exp.fn)
    velocity = convolve_with_tau(theta_turned, REGRESSOR_TAU_S * exp.fn)
    movement = convolve_with_tau(theta_all, REGRESSOR_TAU_S * exp.fn)
    
    ## Indexs for each condition: Crude separation
    threshold = 0.0001
    perc_lower = 50
    perc_higher = 100
    index_CW = np.where(velocity > threshold)[0]
    index_CCW = np.where(velocity < -threshold)[0]
    index_rotation = np.concatenate([index_CW, index_CCW])
    rotation_perc_lower = np.percentile(np.abs(velocity[index_rotation]), perc_lower)
    rotation_perc_higher = np.percentile(np.abs(velocity[index_rotation]), perc_higher)
    index_stable = np.where( np.isclose(movement, 0, atol=threshold) )[0]
    
    # Stable duration
    index2th2_begining_of_stable = np.concatenate([[0], np.where( np.diff(index_stable) > 1 )[0] + 1])
    index2th2_end_of_stable = np.concatenate([np.where( np.diff(index_stable) > 1 )[0], [len(index_stable)-1]])
    stable_duration = np.zeros(len(fictive_head))
    for i, start in enumerate(index2th2_begining_of_stable):
        index_start_stable_t = index_stable[start]
        index_end_stable_t = index_stable[index2th2_end_of_stable[i]]
        stable_duration[index_start_stable_t:index_end_stable_t+1] = np.arange(index_end_stable_t - index_start_stable_t + 1)
    median_stable_duration = np.median(stable_duration[index_stable])
    ## Separate for each condition
    index_CW_middle = np.where( (velocity > rotation_perc_lower) & (velocity < rotation_perc_higher) )[0]
    index_CCW_middle = np.where( (velocity < -rotation_perc_lower) & (velocity > -rotation_perc_higher) )[0]
    index_stable_early = np.where( (stable_duration > 0) & (stable_duration < median_stable_duration))[0]
    index_stable_late = np.where( stable_duration > median_stable_duration)[0]

    median_middle_CW = np.median(velocity[index_CW_middle])
    median_middle_CCW = np.median(velocity[index_CCW_middle])
    mean_middle_CW = np.mean(velocity[index_CW_middle])
    mean_middle_CCW = np.mean(velocity[index_CCW_middle])

    cond_id_crude = [index_CW, index_CCW, index_stable] # CW, CCW, Stable
    cond_id_fine = [index_CW_middle, index_CCW_middle, index_stable_early, index_stable_late]
    cond_stat = [median_middle_CW, median_middle_CCW, mean_middle_CW, mean_middle_CCW, median_stable_duration]
    rotation_perc = [rotation_perc_lower, rotation_perc_higher]

    return cond_id_crude, cond_id_fine, cond_stat, rotation_perc, stable_duration, velocity

def separate_data_into_AHVs(exp):
    # Preparation
    theta_rotation = get_bouts_props_array(exp.n_pts, exp.bouts_df, min_bias=TURN_BIAS, selection="all", value="bias")
    theta_forward = get_bouts_props_array(exp.n_pts, exp.bouts_df, min_bias=0, selection="forward", value="bias")
    theta_all = get_bouts_props_array(exp.n_pts, exp.bouts_df, min_bias=0, selection="all", value="bias")
    fictive_head = convolve_with_tau(np.cumsum(theta_rotation), REGRESSOR_TAU_S * exp.fn)
    velocity = convolve_with_tau(theta_rotation, REGRESSOR_TAU_S * exp.fn)
    movement = convolve_with_tau(theta_all, REGRESSOR_TAU_S * exp.fn)
    movement_forward = convolve_with_tau(theta_forward, REGRESSOR_TAU_S * exp.fn)
    
    ## Indexs for each condition: Crude separation
    threshold = 0.0001
    index_CW = np.where(velocity > threshold)[0]
    index_CCW = np.where(velocity < -threshold)[0]
    median_CW = np.median(velocity[index_CW])
    median_CCW = np.median(velocity[index_CCW])
    
    index_stable = np.where( np.isclose(movement, 0, atol=threshold) )[0]
    index_forawrd = np.where( np.isclose(movement_forward, 0, atol=threshold) )[0]
    
    # Stable duration
    index2th2_begining_of_stable = np.concatenate([[0], np.where( np.diff(index_stable) > 1 )[0] + 1])
    index2th2_end_of_stable = np.concatenate([np.where( np.diff(index_stable) > 1 )[0], [len(index_stable)-1]])
    stable_duration = np.zeros(len(fictive_head))
    for i, start in enumerate(index2th2_begining_of_stable):
        index_start_stable_t = index_stable[start]
        index_end_stable_t = index_stable[index2th2_end_of_stable[i]]
        stable_duration[index_start_stable_t:index_end_stable_t+1] = np.arange(index_end_stable_t - index_start_stable_t + 1)
    median_stable_duration = np.median(stable_duration[index_stable])
    ## Separate for each condition
    index_CW_fast = np.where(velocity > median_CW)[0]
    index_CW_slow = np.where( (velocity > 0.0001) & (velocity < median_CW) )[0]
    index_CCW_fast = np.where(velocity < median_CCW)[0]
    index_CCW_slow = np.where( (velocity < -0.0001) & (velocity > median_CCW) )[0]
    index_stable_early = np.where( (stable_duration > 0) & (stable_duration < median_stable_duration))[0]
    index_stable_late = np.where( stable_duration > median_stable_duration)[0]

    cond_id_crude = [index_CW, index_CCW, index_stable, index_forawrd] # CW, CCW, Stable
    cond_id_fine = [index_CW_fast, index_CW_slow, index_CCW_fast, index_CCW_slow, index_stable_early, index_stable_late]
    cond_median = [median_CW, median_CCW, median_stable_duration]

    return cond_id_crude, cond_id_fine, cond_median, fictive_head, velocity, stable_duration

# Find the range of the direction average
def direc_averg_range(arr, current_id, condition_value=np.pi/36, point_limit=5):
    x = condition_value
    x0 = arr[current_id]
    forward_search = arr.copy()[current_id:]
    backward_search = arr.copy()[:current_id][::-1]
    forward_find = False
    backward_find = False

    forward_endid = current_id + len(forward_search)
    for num, point in enumerate(forward_search):
        if point > x0 + x:
            forward_endid = current_id + num
            forward_find = True
            break
        elif point < x0 - x:
            forward_endid = current_id + num
            forward_find = True
            break
        elif num == point_limit:
            forward_endid = current_id + point_limit
            break

    backward_endid = current_id - len(backward_search)
    for num, point in enumerate(backward_search):
        if point > x0 + x:
            backward_endid = current_id - num
            backward_find = True
            break
        elif point < x0 - x:
            backward_endid = current_id - num
            backward_find = True
            break
        elif num == point_limit:
            backward_endid = current_id - point_limit
            break

    return forward_endid, backward_endid, forward_find, backward_find

# Find the closest direction shift point
def shift_direc_id_np(arr, current_id, shift_value=np.pi/2, point_limit=300): 
    # the input array range [no]: -pi - pi ---> [yes] -pi -> inf

    x = shift_value
    x0 = arr[current_id]
    forward_search = arr.copy()[current_id:]
    backward_search = arr.copy()[:current_id][::-1]

    forward_endid = None
    try:
        num = np.where( ( (forward_search > x0 + x) & (x > 0) ) | ( (forward_search < x0 + x) & (x < 0) ) )[0][0]
    except:
        num = np.Inf
    if num <= point_limit:
        forward_endid = current_id + num

    backward_endid = None
    try:
        num = np.where( ( (backward_search > x0 + x) & (x > 0) ) | ( (backward_search < x0 + x) & (x < 0) ) )[0][0]
    except:
        num = np.Inf
    if num <= point_limit:
        backward_endid = current_id - num
    
    if forward_endid is None:
        shiftid = backward_endid
    elif backward_endid is None:
        shiftid = forward_endid
    elif (forward_endid != None) & (backward_endid != None):
        if abs(forward_endid - current_id) < abs(backward_endid - current_id):
            shiftid = forward_endid
        else:
            shiftid = backward_endid
    else:
        shiftid = None

    return shiftid

def plot_tuning_curve(phase, cell_id, traces):
    phase = phase
    row_num = len(cell_id)//5+1
    fig = plt.figure(figsize=(10, row_num*2))
    for i, a_cell_id in enumerate(cell_id):
        ax = fig.add_subplot(row_num,5,i+1)
        ax.scatter(phase, traces[:,a_cell_id], alpha=0.1)
        ax.set_title(f'cell {a_cell_id}')
    plt.tight_layout()
    plt.show()

def plot_PCA_2comp(cell_id, pca_scores_t, cell_num, circle_params):
    all_cell_id = np.arange(cell_num)
    other_cell_id = np.setdiff1d(all_cell_id, cell_id)

    f, ax = plt.subplots(figsize=(5, 5))
    s = ax.scatter(
        pca_scores_t[cell_id, 0],
        pca_scores_t[cell_id, 1],
        color=COLS["ring"],
        label="r1π ROIs",
        lw=0.2,
    )
    s = ax.scatter(
        pca_scores_t[other_cell_id, 0],
        pca_scores_t[other_cell_id, 1],
        label="cnt ROIs",
        fc="none",
        ec="k",
        lw=0.2,
        zorder=-100,
    )

    s = ax.scatter(
        circle_params[0],
        circle_params[1],
        label="Circle center",
        lw=0.5,
        zorder=100,
    )

    plt.legend(frameon=True)

    ax.set_aspect("equal", "box")
    # pltltr.add_scalebar(ax, xlabel="PC1", ylabel="PC2", xlen=30, ylen=30)
    # ax.axis("equal")
    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()

def replace_consecutive_element(lst, x, element):
    '''
    if the element in lst are consecutively repeated more than x times, ignore the extra repeated elements
    '''

    result = []
    count = 0

    for i in range(len(lst)):
        last_element = lst[i]
        if lst[i] == element:
            count += 1
        else:
            if count > x:
                result.extend([element] * x)
            else:
                result.extend([last_element])
            count = 0

    # Handle the last group of elements
    if count > x:
        result.extend([element] * x)
    else:
        result.extend([last_element])

    return result
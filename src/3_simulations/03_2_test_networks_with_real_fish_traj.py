'''
## Overview
This notebook loads all the valid networks from 
12_2I1R_zebrafish_network_bincrease.ipynb and use the trajectories of
all fish (excluding fish with too few rotations (<3 either side)) as inputs
to examine the performance of the networks in tracking the true head direction.

changing input section perform the main calculations

## Author
Siyuan Mei (mei@bio.lmu.de)


## Last update
2025-9-14
'''

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from scipy.integrate import solve_ivp
from matplotlib import colors
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import pandas as pd
from sklearn.metrics import mean_squared_error

import HD_utils.circular_stats as cstat
from HD_utils.network import *
from HD_utils.matrix import *
from HD_utils.adap_sim_move import *
from HD_utils.adap_sim_stable import *
from HD_utils.IO import *
from HD_utils.plot import *
from HD_utils.comput_property import *
from HD_utils.exam import *

# Load Model

# Simulation theta precision
theta_num = 50
dtheta = (2*np.pi)/theta_num
theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) # must use np.arange(-np.pi+dtheta/2, np.pi, dtheta) to make it symmetry
# Changeable parameters
ring_num = 3
actfun = max0x
weight_fun = vonmises_weight_2i1r_2
search_pars = {'JI': np.linspace(-50,0,6), 'JE': np.linspace(0,50,6), 'K0': np.linspace(-50,0,6), 'kappa': np.logspace(-0.2,1,6)}
file_pre_name = 'new_12'
# Default parameters
inputs = np.array([-1, -0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6, 1])
net_diff_equa = net_diff_equa_f_in
phi = -np.pi * 8/9
tau = 20 # ms
b0 = 1
bc = 1
# Generated parameters
par_num = len(search_pars)
search_num = len(ParameterGrid(search_pars))
zeroid = np.where(inputs == 0)[0][0]
par_names = list(search_pars.keys())
input_num = len(inputs)


network_evals, network_evaldes, network_acvs, network_pars, network_ts = load_pickle(
    ['evals', 'eval_des', 'acvs', 'pars', 'ts'], weight_fun, actfun, file_pre_name)
valid_index_s = np.where(network_evals == 'valid')[0]
valids_num = len(valid_index_s)

Vels, network_eval_moving, network_vvcor, network_acvs_moving, network_ts_moving, network_eval_moving_sum = load_pickle(
    ['moving_slope', 'moving_eval', 'moving_eval_des', 'moving_acvs', 'moving_ts', 'moving_eval_sum'], weight_fun, actfun, file_pre_name)

stable_mov_range, stable_mov_range_id, linear_mov_range, linear_mov_range_id = \
    cal_linear_range(network_eval_moving, Vels, inputs, valid_index_s)
valid_index_linear_move = np.where(np.diff(linear_mov_range_id, axis=1) == 8)[0]


# Load Fish Trajectories

from HD_utils.HD_functions import *
from HD_utils.defaults import DATA_FOLDERS
from lotr import LotrExperiment
from lotr.default_vals import TURN_BIAS
from lotr.behavior import get_bouts_props_array
directory = FISH_DATA_PATH # Should be / rather than \ for linux
# You can change to the directory where you store the data
fish_num = len(DATA_FOLDERS)


traj_fish = []
for fish in tqdm(range(fish_num)):
    exp = LotrExperiment(DATA_FOLDERS[fish])
    fs = exp.fn
    cell_num = len(exp.hdn_indexes)
    theta_rotation = get_bouts_props_array(exp.n_pts, exp.bouts_df, min_bias=TURN_BIAS, selection="all", value="bias")
    theta_rotation = replace_consecutive_element(theta_rotation, 10, 0) # replace any consecutive 0s which repeated more than x times (x/5s) by 0s repeating x times
    angle = np.concatenate([[0], np.cumsum(theta_rotation)])
    angVel = np.concatenate([[0], theta_rotation]) * fs
    t = np.arange(0, len(angle) / fs, 1/fs)
    df_temp = pd.DataFrame({'t': t, 'angle_expand': angle, 'angVel': angVel})
    traj_fish.append(df_temp)
    
    
# Perform Simulations

net_id_list = []
traj_i_list = []
ratio_list = []
r_list = []
net_phase_list = []
phase_raw_list = []
mse_list = []

for i_num, i in enumerate(tqdm(valid_index_linear_move)):
    w = weight_fun(*network_pars[i], phi, theta_num)
    vel = Vels[i].mean(axis=1)
    s1 = np.concatenate( [network_acvs[i][0,:,-1], network_acvs[i][1,:,-1], network_acvs[i][2,:,-1]] )
    slope, _, _, _ = np.linalg.lstsq(inputs.reshape(-1,1), vel * np.pi * 2, rcond=None)
    slope = slope[0]
    
    ratio_list_temp = []
    for j in tqdm(range(len(traj_fish))):
        traj_sample = traj_fish[j]
        t0 = traj_sample.t[0]
        t_spans, ratiovs, t_evals = cal_sim_t_angVel(traj_sample, 'fish', slope)
        for si in range(len(ratiovs)):
            t_span = t_spans[si]
            t_eval = t_evals[si]
            ratiov = ratiovs[si]
            b_array = steady_inputb_2rr_b_increase(bc, b0, ratiov, theta_num)  # input
            ## Compute network dynamics 
            y0 = s1 if si == 0 else net_dynamics.y[:,-1].copy()

            # compute atol
            rdtol = 1e-6
            rtol = 1e-6
            y0list = sep_slist(y0, ring_num)
            atollist = [0] * ring_num
            for ri in range(ring_num):
                atollist[ri] = rdtol*np.ptp(y0list[ri])
            atol = np.min(atollist) + 1e-8 # !!! Caution, atol is not computed for each solve_ivp step, but for initial value only !!!
            
            # Solve ODE
            funargs = [w, tau, b_array, dtheta, actfun]
            net_dynamics = solve_ivp(net_diff_equa, t_span, y0, t_eval=t_eval, 
                                        args=funargs, events=explode_event, rtol=rtol, atol=atol) # when explode, stop simulation
            if si == 0:
                y, t = net_dynamics.y, net_dynamics.t
                yt, tt = y, t
            else:
                yt, tt = net_dynamics.y, net_dynamics.t
                y, t = net_var_append([y, t], [yt, tt])
        phase_raw = cal_peak_loc_auto(y, theta_range)
        net_phase = cstat.rerange_expand(phase_raw.mean(axis=0)) if ring_num == 2 else cstat.rerange_expand(phase_raw[0])
        # if 2 ring: average of the left and right; if 3 ring: select the central ring; if 1 ring: the only ring
        true_HD = (traj_sample.angle_expand.values)
        r = np.corrcoef(true_HD, net_phase)[0,1]
        tp_ratio, _, _, _ = np.linalg.lstsq(net_phase.reshape(-1,1), true_HD, rcond=None) # zero intercept
        tp_ratio = tp_ratio[0]

        net_id_list.append(i)
        traj_i_list.append(j)
        r_list.append(r)
        ratio_list.append(tp_ratio)
        ratio_list_temp.append(tp_ratio)
        net_phase_list.append(net_phase)
        phase_raw_list.append(phase_raw)

    mean_ratio = np.mean(ratio_list_temp)
    for j_enu, j in enumerate(range(len(traj_fish))):
        traj_sample = traj_fish[j]
        true_HD = (traj_sample.angle_expand.values)
        net_phase = net_phase_list[j_enu + len(traj_fish) * i_num]
        mse = mean_squared_error(true_HD, net_phase * mean_ratio)
        mse_list.append(mse)

net_performance = pd.DataFrame({'net_id': net_id_list, 'traj_i': traj_i_list, 'r': r_list, 'ratio': ratio_list, 'mse': mse_list})
sim_prefix = 'new13_'
pd.to_pickle(net_performance, SIM_RESULT_PATH / sim_prefix + 'net_performance.pkl')

with open(SIM_RESULT_PATH / sim_prefix + 'net_phase_list.pkl', 'wb') as f:
    pickle.dump(net_phase_list, f)
with open(SIM_RESULT_PATH / sim_prefix + 'phase_raw_list.pkl', 'wb') as f:
    pickle.dump(phase_raw_list, f)
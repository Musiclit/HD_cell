'''Not used?'''
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from HD_utils.IO import *
from HD_utils import network
from scipy.stats import rankdata

network_settings = pd.read_pickle(SIM_RESULT_PATH + '/network_v1_settings.pkl')

def bools_valid_no_flat_mult_ring(net_filei, restrict_phi, tanh_thres=3, rel_thres=1e-3, thres_num=3):

    b_value = network_settings.loc[net_filei, 'b']
    ring_num = network_settings.loc[net_filei, 'Ring num']
    actfun = getattr(network, network_settings.loc[net_filei, 'actfun_'])
    weight_fun = getattr(network, network_settings.loc[net_filei, 'weightfun_'])
    search_pars = network_settings.loc[net_filei, 'search_pars']
    file_pre_name = network_settings.loc[net_filei, 'file_pre_name']
    act_fun_name = network_settings.loc[net_filei, 'Act Fun']
    ## Generated parameters
    search_num = len(ParameterGrid(search_pars))
    if ring_num == 2:
        bs = [b_value, b_value]
    elif ring_num == 3:
        bs = [b_value[0], b_value[1], b_value[1]]
    phi_num = len(restrict_phi)
    
    cond1 = np.zeros((phi_num, search_num), dtype=bool) # valid stable
    cond2 = np.ones((phi_num, search_num), dtype=bool) # no flat bump

    for phi_enu, phi_i in enumerate(restrict_phi):
        if phi_i == 1:
            network_evals, network_acvs, = load_pickle(['evals', 'acvs', ], weight_fun, actfun, file_pre_name)
        else:
            network_evals, network_acvs, = load_pickle(['evals', 'acvs', ], weight_fun, actfun, 
                                                       '_RefractionV_' + file_pre_name + f'_phi{str(abs(int(phi_i / 18 * 360)))}degrees')
        network_acvs = np.array(network_acvs) # to supress errors
        cond1[phi_enu] = (network_evals == 'valid')
        
        for i in range(search_num):
            for r in range(ring_num):
                acv = network_acvs[i][r,:,-1]
                acv_max = np.max(acv)
                flat_num1 = np.sum(acv > (1 - rel_thres) * acv_max)
                flat_num2 = 0
                if act_fun_name == 'tanh1':
                    flat_num2 = np.sum(acv + bs[r] > tanh_thres)
                if (flat_num1 >= thres_num) | (flat_num2 >= thres_num):
                    cond2[phi_enu,i] = False
                    break

    # print(f'{np.sum(cond1)}/')
    return cond1&cond2

# def order_valid_net_82(net_filei, refrac_v_nets_all, vv_slope_nets_all, net_b_v_nets_all, desired_slope, v_thres, restrict_phi=np.arange(1,9)):

#     '''restrict phi: int ranges from 1 ~ 8, represent 1/9 pi ~ 8/9 pi'''
    
#     if net_filei >= 4:
#         net4_id = net_filei - 4
#         refrac_v = refrac_v_nets_all[net4_id][restrict_phi-1]
#         vv_slope = np.abs(vv_slope_nets_all[net4_id][restrict_phi-1])
#         slope_dist = vv_slope / desired_slope
#         slope_dist[slope_dist > 1] = 1 / slope_dist[slope_dist > 1]
#         net_r = np.array(np.abs(net_b_v_nets_all[net4_id][restrict_phi-1,:,2]).copy()).astype(float)

#         maxv_cond = refrac_v > v_thres
#         # slope_cond = (vv_slope > slope_range[0]) & (vv_slope < slope_range[1])
#         no_flat_cond = bools_valid_no_flat_mult_ring(net_filei, restrict_phi)
#         cond = maxv_cond & no_flat_cond

#         rank_slope = rankdata(-slope_dist, nan_policy='omit') # 0: highest slope, end np.nan or zero
#         rank_r = rankdata(-net_r, nan_policy='omit') # 0: highest r, end np.nan or zero
#         rank = rank_slope + rank_r
#         rank_sort = np.argsort(rank) # 0: smallest rank

#         net_r[~cond] = np.nan
#         max_order = net_r.flatten()[rank_sort]
#     elif net_filei < 4:
#         actfun = getattr(network, network_settings.loc[net_filei, 'actfun_'])
#         weight_fun = getattr(network, network_settings.loc[net_filei, 'weightfun_'])
#         file_pre_name = network_settings.loc[net_filei, 'file_pre_name']
#         network_vvr, network_eval_moving_sum = load_pickle(['moving_eval_des', 'moving_eval_sum'], weight_fun, actfun, file_pre_name)
#         network_vvr = np.array(network_vvr) # to supress errors
#         net_r = np.abs(network_vvr[:,0,0])
#         cond_valid = (network_eval_moving_sum == 'linear moving')
#         max_order = np.sort(net_r[cond_valid])[::-1]

#     return max_order, net_r

def order_valid_net_82(net_filei, refrac_v_nets_all, vv_slope_nets_all, net_b_v_nets_all, desired_slope, v_thres, restrict_phi=np.arange(1,9),
                         r_thres=0.99):

    '''restrict phi: int ranges from 1 ~ 8, represent 1/9 pi ~ 8/9 pi'''
    
    if net_filei >= 4:
        net4_id = net_filei - 4
        refrac_v = refrac_v_nets_all[net4_id][restrict_phi-1]
        vv_slope = np.abs(vv_slope_nets_all[net4_id][restrict_phi-1])
        
        net_r = np.array(np.abs(net_b_v_nets_all[net4_id][restrict_phi-1,:,2]).copy()).astype(float)

        maxv_cond = refrac_v > v_thres
        # slope_cond = (vv_slope > slope_range[0]) & (vv_slope < slope_range[1])
        no_flat_cond = bools_valid_no_flat_mult_ring(net_filei, restrict_phi)
        cond = maxv_cond & no_flat_cond

        slope_penalty = vv_slope / desired_slope
        slope_penalty[slope_penalty < 1] = 1 / slope_penalty[slope_penalty < 1]
        cor_penelty = (1 - net_r) / (1 - r_thres)
        total_penalty = slope_penalty + cor_penelty
        rank_sort = np.argsort(total_penalty, axis=None) # 0: smallest rank

        net_r[~cond] = np.nan
        max_order = net_r.flatten()[rank_sort]
    elif net_filei < 4:
        actfun = getattr(network, network_settings.loc[net_filei, 'actfun_'])
        weight_fun = getattr(network, network_settings.loc[net_filei, 'weightfun_'])
        file_pre_name = network_settings.loc[net_filei, 'file_pre_name']
        network_vvr, network_eval_moving_sum = load_pickle(['moving_eval_des', 'moving_eval_sum'], weight_fun, actfun, file_pre_name)
        network_vvr = np.array(network_vvr) # to supress errors
        net_r = np.abs(network_vvr[:,0,0])
        cond_valid = (network_eval_moving_sum == 'linear moving')
        max_order = np.sort(net_r[cond_valid])[::-1]

    return max_order, net_r

def get_nth_id_mult_phi(max_order, net_r, nth): # there is no phi for 1 ring model
    if nth >= len(max_order):
        return False, False
    elif np.isnan(max_order[nth]):
        return 'next', 'next'
    else:
        ids = np.where(net_r == max_order[nth])
        return ids[0][0], ids[1][0]
    
def get_nth_id_1phi(max_order, net_r, nth, ring_num):
    if nth >= len(max_order):
        return False
    elif np.isnan(max_order[nth]):
        return 'next'
    elif ring_num == 1:
        return np.where(net_r == max_order[nth])[0][0]
    elif ring_num > 1:
        return np.where(net_r == max_order[nth])[1][0]
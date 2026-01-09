import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.model_selection import ParameterGrid

from HD_utils.network import *
from HD_utils import network
from HD_utils.exam import *
from HD_utils.adap_sim_move import *
from HD_utils.IO import *
from HD_utils.adap_sim_stable import *
from HD_utils.dataclass import SimulationConfig


def cal_vel_given_input(ratiov, bs_1net, vels_1net, b_value, theta_num, ring_num, net_diff_equa, s1, w, tau, actfun, append=True):
    '''
    Perform integration for a given input (i.e. ratiov) and return the results
    also save the input and corresponding velocity to 
    bs_1net and vels_1net if append is True
    '''
    b_array = produce_inputb(b_value, ratiov, theta_num, ring_num)  # input
    y, t, eval, vel = inte_check_move(
        net_diff_equa, s1, w, tau, b_array, theta_num, actfun)
    if append:
        bs_1net.append(ratiov)
        vels_1net.append(vel.mean())
    return y, t, eval, vel


def save_max_v_and_other_vars(refrac_v, net_b_v, vv_slope, neti, bs_1net, vels_1net, bs_1net_remianing, vels_1net_remianing, inputs=None, VelsM=None):
    '''
    Store the the max velocity
    the velocities and inputs b with a correlation > 0.99,
    the slope from b to v,
    the correlation between b and v (> 0.99 part),
    and all simulated b and v
    '''
    sort_id = np.argsort(bs_1net)
    bs_1net = np.array(bs_1net)[sort_id]
    vels_1net = np.array(vels_1net)[sort_id]

    net_b_v[neti, 0] = bs_1net
    net_b_v[neti, 1] = vels_1net

    refrac_v[neti] = net_b_v[neti, 1][0] 

    slope, _, _, _ = np.linalg.lstsq(net_b_v[neti, 0].reshape(-1,1), net_b_v[neti, 1], rcond=None) # zero intercept
    vv_slope[neti] = slope[0]

    r = pearsonr(net_b_v[neti, 0], net_b_v[neti, 1])[0]
    net_b_v[neti, 2] = r

    bs_min = bs_1net[0]
    if VelsM is None:
        net_b_v[neti, 3] = np.concatenate((bs_1net, bs_1net_remianing))
        net_b_v[neti, 4] = np.concatenate((vels_1net, vels_1net_remianing))
    else:
        other_ids = np.where(inputs < bs_min)[0]
        net_b_v[neti, 3] = np.concatenate((inputs[other_ids], bs_1net, bs_1net_remianing))
        net_b_v[neti, 4] = np.concatenate((VelsM[neti, other_ids], vels_1net, vels_1net_remianing))

    sort_id = np.argsort(net_b_v[neti, 3])
    net_b_v[neti, 3] = net_b_v[neti, 3][sort_id]
    net_b_v[neti, 4] = net_b_v[neti, 4][sort_id]


def pop_and_store(bs_1net, vels_1net, bs_1net_remianing, vels_1net_remianing):
    '''
    Remove the last element from bs_1net and vels_1net
    and store them in bs_1net_remianing and vels_1net_remianing
    '''
    bs_1net_remianing.append(bs_1net[-1])
    vels_1net_remianing.append(vels_1net[-1])
    bs_1net.pop()
    vels_1net.pop()


def get_valid_stable_net(net_filei, tau, phi, alpha, theta_num, theta_range, net_diff_equa):
    '''
    Outdated function, have been replaced by 03_1_valid_stationary_network_search.py
    '''
    network_settings = pd.read_pickle(SIM_RESULT_PATH + '/network_v1_settings.pkl')
        # Get parameters
    b_value = network_settings.loc[net_filei, 'b']
    ring_num = network_settings.loc[net_filei, 'Ring num']
    actfun = getattr(network, network_settings.loc[net_filei, 'actfun_'])
    weight_fun = getattr(network, network_settings.loc[net_filei, 'weightfun_'])
    search_pars = network_settings.loc[net_filei, 'search_pars']
    file_pre_name = network_settings.loc[net_filei, 'file_pre_name']
    ## Generated parameters
    par_num = len(search_pars)
    search_num = len(ParameterGrid(search_pars))
    par_names = list(search_pars.keys())

    # Valid Stable
    t_max1_stable = tau * 10 # value must be bigger than 200, if t > t_max1 & network has a bad shape, end simulation
    t_max2_stable = tau * 50 # if t > t_max2, stop simulation, even though network has a good shape
    # Routine
    network_acvs = np.zeros(search_num, dtype='object')
    network_evals= np.ones(search_num, dtype='U30')
    network_evaldes = np.zeros(search_num, dtype='object')
    network_pars = np.zeros((search_num, par_num))
    network_ts = np.zeros(search_num, dtype='object')

    b_array_stable = produce_inputb(b_value, 0, theta_num, ring_num)

    for i, pars in enumerate(tqdm(  list(ParameterGrid(search_pars))  )):
        # --- Need change
        par_list = [pars[par_names[j]] for j in range(par_num)]
        
        par_list = [pars[par_names[j]] for j in range(par_num)]
        par1 = phi if ring_num == 2 else alpha
        par2 = theta_num if ring_num == 2 else theta_range
        w = weight_fun(*par_list, par1, par2)
        ## Routine
        s0 = net_ini_v2(theta_range, ring_num, offset=0)
        y, t, network_evals[i], network_evaldes[i] = \
            inte_check_sta(net_diff_equa, t_max1_stable, t_max2_stable, s0, w, tau, b_array_stable, theta_num, actfun)
        # Store values
        network_pars[i] = par_list
        if ring_num == 2:
            network_acvs[i] = np.array([y[:theta_num], y[theta_num:]])
        elif ring_num == 3:
            network_acvs[i] = np.array([y[:theta_num], y[theta_num:2*theta_num], y[2*theta_num:3*theta_num]])
        network_ts[i] = t
    # save simulation result
    store_pickle([network_evals, network_evaldes, network_acvs, network_pars, network_ts], 
    ['evals', 'eval_des', 'acvs', 'pars', 'ts'], weight_fun, actfun, '_RefractionV_' + file_pre_name + f'_phi{str(abs(int(alpha / (2*np.pi) * 360)))}degrees')

    valid_index_s = np.where(network_evals == 'valid')[0]
    return valid_index_s, network_pars, network_acvs


def determine_max_vel(config: SimulationConfig, 
                        net_stationary: GridSearchResultStationary,
                        net_moving: GridSearchResultMoving) -> List[np.ndarray]:
    '''
    Determine the maximum velocity for a given network type
    and store the results
    
    Note that refract_v, Rv, and max_velocity are used interchangeably

    Returns:
        refrac_v: np.1darray[search_num]
            Maximum velocity for each valid network configuration
        vv_slope: np.1darray[search_num]
            Slope of the from the input to velocity
        net_b_v: np.2darray[search_num, 5]
            Detailed input strengths and corresponding velocities for each valid network configuration
            2nd dimension: 0: b, 1: vel, 2: correlation between b and v, 3: b including all simulated b, 4: vel including all simulated v
    '''
    # Unpack config to accommodate previous function signature
    valid_index_s = net_stationary.valid_id
    network_pars = net_stationary.par
    network_acvs = net_stationary.activity
    phi = config.phi
    alpha = config.alpha
    theta_num = config.theta_num
    theta_range = config.theta_range
    net_diff_equa = config.net_diff_equa
    tau = config.tau
    b_value = config.bs
    ring_num = config.ring_num
    actfun = config.actfun
    weight_fun = config.weight_fun
    search_num = config.search_num
    
    maxV_thres = config.maxV_thres
    base_v = config.base_v
    base_v_tol = config.base_v_tol
    v_precision = config.v_precision
    vvr_thres = config.vvr_thres
    search_mult = config.search_mult
    ini_b_stop_count = config.ini_b_stop_count
    inputs = config.inputs
    VelsM = net_moving.velocity.mean(axis=2)
    
    # Variable storage
    refrac_v = np.zeros(search_num)
    refrac_v[:] = np.nan
    refrac_v[valid_index_s] = 0
    
    vv_slope = np.copy(refrac_v)
    
    net_b_v = np.zeros((search_num, 5), dtype=object) # 2dim: 0: b, 1: vel, 2: r, 3: b complete, 4: vel complete
    net_b_v[:] = np.nan
    net_b_v[valid_index_s, :] = 0

    for neti in (valid_index_s):
        # Initialize storage variables
        base_ratiov = 0  # Default value before it's properly set
        # store b and vel used within which corr > 0.99
        bs_1net = [0]
        vels_1net = [0]
        # store remaining b and vel
        bs_1net_remianing = []
        vels_1net_remianing = []
        
        # weights of the network
        pars = network_pars[neti]
        par1 = phi if ring_num == 2 else alpha
        par2 = theta_num if ring_num == 2 else theta_range
        w = weight_fun(*pars, par1, par2)
        # Initial state is the stable state in the stationary case
        s1 = network_acvs[neti][:,:,-1].flatten() 

        # select the ini b that produce velocity = 0.1
        b_for_01v_found = False # if found that b?
        Rv_small = False # if the max velocity is too small
        ratiov = -0.1 # initial input
        count = 0
        while not b_for_01v_found:
            count += 1

            if (count > ini_b_stop_count): # if reach the max count, break
                break
            
            y, t, eval, vel = cal_vel_given_input(ratiov, bs_1net, vels_1net, b_value, theta_num, ring_num, net_diff_equa, s1, w, tau, actfun, append=False)
            bs_1net_remianing.append(ratiov)
            vels_1net_remianing.append(vel.mean())

            # if too small (== 0), increase ratiov
            if vel.mean() == 0:
                ratiov = ratiov * search_mult
            
            # If too large or too small, adjust ratiov
            elif (vel.mean() > base_v + base_v_tol) | (vel.mean() < base_v - base_v_tol):
                ratiov = ratiov / (vel.mean() / base_v)
            
            # If == 0.1 within a tolerance
            elif (vel.mean() >= base_v - base_v_tol) & (vel.mean() <= base_v + base_v_tol):
                base_ratiov = ratiov
                b_for_01v_found = True

                # double ratiov and check the linearity in the range [0,0.1,0.2]
                bs_1net_remianing.pop()
                vels_1net_remianing.pop()
                bs_1net.append(ratiov)
                vels_1net.append(vel.mean())
                ratiov = ratiov * 2
                y, t, eval, vel = cal_vel_given_input(ratiov, bs_1net, vels_1net, b_value, theta_num, ring_num, net_diff_equa, s1, w, tau, actfun)
                
                if eval == 'stable moving': # bifrication point 4 OP2 condition 1
                    r = np.abs(pearsonr(bs_1net, vels_1net)[0])
                    Rv_small = True if r < vvr_thres else False
                else:
                    Rv_small = True
            
            # if nan, maybe the ratiov is too large
            else:
                ratiov = ratiov / search_mult
        
        if Rv_small | (not b_for_01v_found): 
            # ill set max_velocity (i.e. RV) = 0 because 
            # it is not linear in the range [0,0.1,0.2]
            continue 

        # If the network is linear in the range [0,0.1,0.2]
        # continue to search for max velocity
        result_found = False
        while not result_found:
            step = ratiov * (search_mult - 1)
            ratiov = ratiov * search_mult
            prec_improve = True

            y, t, eval, vel = cal_vel_given_input(ratiov, bs_1net, vels_1net, b_value, theta_num, ring_num, net_diff_equa, s1, w, tau, actfun)

            if eval == 'stable moving': # bifrication point 4 OP2 condition 1
                r = np.abs(pearsonr(bs_1net, vels_1net)[0])
                # If have exceeded the max velocity
                # perform a finer search (prec_improve = True) 
                # to narrow down the max velocity
                prec_improve = False if r > vvr_thres else True # bifrication point 4 OP2 condition 2
            # ELSE: back to while

            # bifrication point 4: OP1, search between (ratiov, ratiov - step)
            if prec_improve: 
                ratiov = ratiov - step
                
                # Determine the upper and lower bounds of the max velocity
                v_upper = vels_1net[-1]
                b_upper = bs_1net[-1]
                # remove the last one which makes r < 0.99
                pop_and_store(bs_1net, vels_1net, bs_1net_remianing, vels_1net_remianing) 
                v_lower = vels_1net[-1]
                b_lower = bs_1net[-1]
                search_neger = 1

                count_nan = 0
                count_nan_max = int(np.ceil(np.log2((ratiov / base_ratiov) * (base_v + base_v_tol) / v_precision))) + 1
                while True:
                    count_nan += 1
                    # Bifrication point 5 OP1: find the refract vel
                    # If the upper and lower bounds are close enough or
                    # too many iterations, stop searching
                    if (np.abs(v_upper - v_lower) < v_precision) | (count_nan > count_nan_max): # > 7, because  
                        if v_lower > maxV_thres:
                            if b_lower not in bs_1net:
                                bs_1net.append(b_lower)
                                vels_1net.append(v_lower)
                            save_max_v_and_other_vars(refrac_v, net_b_v, vv_slope, neti, bs_1net, vels_1net, bs_1net_remianing, vels_1net_remianing, inputs, VelsM)
                        # Else: Rv = 0
                        result_found = True
                        break
                    
                    # ELSE: bifrication point 5 OP2: increase precision
                    step /= 2
                    ratiov = ratiov + step * search_neger
                    y, t, eval, vel = cal_vel_given_input(ratiov, bs_1net, vels_1net, b_value, theta_num, ring_num, net_diff_equa, s1, w, tau, actfun)
                    
                    if eval == 'stable moving':
                        r = np.abs(pearsonr(bs_1net, vels_1net)[0])
                        search_neger = 1 if r > vvr_thres else -1
                    else:
                        search_neger = -1

                    if search_neger == 1:
                        v_lower = vels_1net[-1]
                        b_lower = bs_1net[-1]
                    elif search_neger == -1:
                        v_upper = vels_1net[-1]
                        b_upper = bs_1net[-1]
                    
                    pop_and_store(bs_1net, vels_1net, bs_1net_remianing, vels_1net_remianing)
                    
    # Store results
    arraylist = [refrac_v, vv_slope, net_b_v]
    arraynames = ['max_vel', 'b2v_slope', 'b_and_v_and_corr']
    additional_text = f'{config.file_pre_name}_phi_{str(abs(int(alpha / (2*np.pi) * 360)))}_deg'
    
    for i, var in enumerate(arraylist):
        pathstr = SIM_RESULT_PATH / "max_velocity" / f"{additional_text}_{arraynames[i]}.p"
        with open(pathstr, 'wb') as f:
            pickle.dump(var, f)

    print(f"Maximum velocity {np.nanmax(refrac_v):.2f} turns/s found for Network type {config.file_pre_name} completed.")
    
    return refrac_v, vv_slope, net_b_v


    
    
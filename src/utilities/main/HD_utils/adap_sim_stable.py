'''
A module that simulate a ring attractor network's behavior given zero input. 
It adaptively stops simulation based on coded criterion: if it is deem to be 
stable or unstable.

Siyuan Mei (mei@bio.lmu.de)
2024
'''
# pyright: reportUnboundVariable = false
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

from HD_utils.network import *
from HD_utils.exam import *
from HD_utils.IO import store_pickle, save_grid_search_results
from HD_utils.dataclass import *


def inte_check_sta(net_diff_equa, t_max1, t_max2, s0, w, tau, b, theta_num, actfun, 
                        rtol=1e-6, rdtol=1e-6, rctol=0.01, rdctol=0.01, rtolflat=0.01, 
                        rtol_bumpyflat=0.1, peak_s_time=None, maxdistol=5,
                        exam_interval_len=None, rec_t_gap=None, dtheta=None):
    '''
    Do the numerical integration and check the validity of the bump shape in a ring
    attractor network for the stationary case (i.e., the bump is not moving).

    Parameters
    ----------
    net_diff_equa: function
        Differential equation of the network.
    t_max1: int or float    
        Time (ms) after which bump shape and stability are examined.
    t_max2: int or float
        Maximum simulation time (ms) to reach a stable state.
    s0: 1darray
        Initial state of the network (size: theta_num * ring_num).
    w: 2darray
        Weight matrix of the network (size: [theta_num * ring_num, theta_num * ring_num]).
    tau: int or float
        Time constant of the network. Mostly use 20.
    b: 1darray
        External inputs to the network (length = number of neurons).
    theta_num: int
        Number of neurons in each ring.
    actfun: function
        Activation function of the network.
    rtol: float, optional
        Relative tolerance for numerical integration.
    rdtol: float, optional
        Relative tolerance for absolute tolerance in integration (atol = rdtol * np.ptp(y)).
    rctol: float, optional
        Relative tolerance for bump shape stability check.
    rdctol: float, optional
        Relative tolerance for bump shape stability (atol = rdctol * np.ptp(y)).
    rtolflat: float, optional
        Relative tolerance for checking if the bump is flat.
    rtol_bumpyflat: float, optional
        Relative tolerance for checking if the bump is bumpy flat.
    peak_s_time: int or float, optional
        Time interval for peak stability check. If None, defaults to tau//20*200.
    maxdistol: int or float, optional
        Tolerance for peak stability check. Decreases linearly from 
        t_max1 (maxdistol) to t_max2 (1).
    exam_interval_len: int or float, optional
        Length of each shape and stability examination interval. 
        If None, defaults to tau//20*100
    rec_t_gap: int or float, optional
        Time gap between recorded points in each interval. 
        If None, defaults to tau//20*50.
    dtheta: float, optional
        Distance between adjacent neurons theta. 
        If None, defaults to 2*pi/theta_num.

    Returns
    -------
    y: 2darray (theta_num * ring_num, t_num)
        Two ring: (left|right, t_num). Three ring: (central|left|right, t_num).
        t_num is the number of all the recorded time points.
    t: 1darray (t_num)
        Recorded time points, whose length is determined by when the numerical 
        integration terminates.
    evals: str
        Evaluation of the network, includes: valid, flat, explode, unstable, 
        multiple peaks, unstable bumpy flat, singularity. The default value is 
        1 and has no meaning.
    evaldes: list of str
        Description of the evaluation for debugging

    '''
    # produce examination intervals and recording intervals
    trange, amplf, exam_interval_len, rec_t_gap = _produce_trange_s(
        tau, t_max1, t_max2, exam_interval_len, rec_t_gap) 
    
    # Set default values if not specified otherwise
    dtheta = (2*np.pi)/theta_num if dtheta is None else dtheta
    peak_s_time = 200 * amplf if peak_s_time is None else peak_s_time
    # Determin ring numbers
    ring_num = s0.shape[0] // theta_num

    # Examine
    evals = '1' # initial value
    evaldes = []
    for ti, trangev in enumerate(trange):
        # Initial state for this interval,
        # Almost zero if at the beginning, 
        # else the last state of the last interval
        y0 = s0 if ti == 0 else net_dynamics.y[:,-1].copy()
        # separate activity to a list of each ring's activity
        y0list = sep_slist(y0, ring_num) 
        
        # Set the absolute tolerance of error during numerical simulation
        # as the minimum of each ring's "peak to bottom" * rdtol
        atollist = [0] * ring_num
        for i in range(ring_num):
            atollist[i] = rdtol*np.ptp(y0list[i])
        atol = np.min(atollist)

        # Solve OED in the time range [min(trangev), max(trangev)]
        # The values are recorded for each time point within trangev
        # explode_event: when activity becomes very large, stop simulation
        # the error tolerance of the numerical integration is set as y*rtol+atol
        net_dynamics = solve_ivp(net_diff_equa, (min(trangev),max(trangev)), y0, t_eval=trangev, 
                                 args=(w, tau, b, dtheta, actfun), 
                             events=explode_event, rtol=rtol, atol=atol) 
        
        # Store the result
        if ti == 0:
            y, t = net_dynamics.y, net_dynamics.t
        else:
            yt, tt = net_dynamics.y, net_dynamics.t
            y, t = net_var_append([y, t], [yt, tt])
        
        # Return if the ODE solver reaches singularity
        if net_dynamics.success == False:
            return y, t, 'singularity', ['ODE solver reaches singularity']

        # Return if the integration stops due to network explosion
        if len(net_dynamics.t_events[0]) > 0:
            evals = 'explode'
            return y, t, evals, evaldes

        # After t_max1, start examination
        if (min(trangev) >= t_max1):

            # separate activity to a list of each ring's activity
            slist = sep_slist(y, ring_num)
            # Prepare variables for the evaluation
            sptp_list, se_list, mean_acv_list = prep_exam_var(slist, ring_num)

            # Stop integration if the shape is bad
            ## First, check whether all activity is close to zero
            evals, evaldes = zeroflat_check(mean_acv_list, ring_num, evaldes)
            if evals == 'flat':
                return y, t, evals, evaldes
            
            ## Second, Check whether all activity is close to the mean activity
            evals, evaldes, _, medium_rvar_num = rel_flat_check(
                se_list, mean_acv_list, ring_num, rtolflat, rtol_bumpyflat, theta_num, evaldes)
            if evals == 'flat':
                return y, t, evals, evaldes

            ## Third, check whether there are multiple peaks
            evals, evaldes = exam_peak_num(slist, evaldes)
            if evals == 'multiple peaks':
                return y, t, evals, evaldes
            
            ## Fourth, check whether the peak location difference
            ## between two consecutive time points <= distol
            ## distol decreases linearly from maxdistol at t_max1 to 1 at t_max2
            ## If network pass the examination, it means it is not too 
            ## unstable, but to determine is it is stable enough, I use another
            ## examination below by checking if peak stable within the 
            ## peak_s_time interval (default 200ms)
            distol = maxdistol - \
                (maxdistol - 1) * (max(trangev) - t_max1) / (t_max2 - t_max1)
            if not(if_peak_loc_stable2(rec_t_gap, rec_t_gap, slist, distol)):
                evals = 'unstable'
                evaldes.append('peak moves after t_max1')
                return y, t, evals, evaldes
            
            
            # Check if the population profile remains the same shape
            shape_match_list = [False] * ring_num
            for i in range(ring_num):
                # Determine the tolerance for shape matching
                ptptol = rdctol * sptp_list[i]
                meantol = rctol * np.mean(np.abs(slist[i][:,-1]))
                atol = min(ptptol, meantol)
                
                # Check whether the shape matches between two consecutive time points
                shape_match_list[i] = np.all( np.isclose(slist[i][:,-2], slist[i][:,-1], atol=atol, rtol=1e-16) )
            
            # Check if all rings activity shape remains the same
            shape_match = np.all(shape_match_list)

            # Check whether the peak location is stable
            peak_loc_stable = if_peak_loc_stable2(peak_s_time, rec_t_gap, slist)

            # Return result if the activity profile's shape remains the same
            # and the peak location is stable
            if shape_match:
                if peak_loc_stable:         
                    if (medium_rvar_num == theta_num * ring_num):
                        evals = 'flat'
                        evaldes.append('relative bumpy flat')
                        return y, t, evals, evaldes
                    else:
                        evals = 'valid'
                        evaldes.append(atol)
                        return y, t, evals, evaldes
    
    # Do not converge after t_max2
    if (medium_rvar_num == theta_num * ring_num):
        evals = 'unstable bumpy flat'
    else:
        evals = 'unstable'
    if (not shape_match):
        evaldes.append('shape unstable')
    elif (not peak_loc_stable):
        evaldes.append('peak location unstable')
    return y, t, evals, evaldes


def _produce_trange_s(tau, t_max1, t_max2, exam_interval_len, rec_t_gap):
    '''
    To produce time points at which shape and stability was examined 
    and activity was recorded.
    It also check if t_max1 >= 10*tau
    
    Parameters:
    tau: int (ms)
        time constant of the network, default to 20
    t_max1: int (ms)
        bump should have a good shape before t_max1, 
        but it is allowed to change its shape and peak location after t_max1.
    t_max2: int (ms)
        bump should not change its shape and peak location after t_max2.
    exam_interval_len: int (ms) or None
        length of each shape and stability examination interval.
        If None, default to tau * 5, if tau default to 20 then it is 100.
    rec_t_gap: int (ms) or None
        time gap between recorded points in each interval.
        If None, default to tau * 2.5, if tau default to 20 then it is 50.
        
    Returns:
    trange: list of 1darray
        Each 1darray includes all time points where the values of y shall be recorded.
        example: [array([0,50,100]), array([100,150,200]), ...]
    amplf: int
        tau // 20, used to determine default values of exam_interval_len and rec_t_gap
        because tau is default to 20, amplf is default to 1
    exam_interval_len: int (ms)
        max - min of each array within the list of trange
    rec_t_gap: int (ms)
        gap between each two consecutive time points within each array
    '''

    # determine the default value according to tau
    # tau is default to 20
    amplf = tau // 20 
    
    if t_max1 < 200 * amplf:
        raise ValueError('t_max must be bigger than 10 * tau')
    
    if exam_interval_len is None:
        exam_interval_len = 100 * amplf # 5 * tau, default 100
    if rec_t_gap is None:
        rec_t_gap = exam_interval_len // 2 # 2.5 * tau, default 50
    
    # produce trange, list of time points to exam and record
    # default: [array([0,50,100]), array([100,150,200]), ...]
    # exam_interval_len is the max - min of each array within the list
    # rec_t_gap is the gap between each two consecutive time points within each array
    trange = []
    for tv in range(0, t_max2-exam_interval_len+1, exam_interval_len):
        trange.append(np.arange(tv, tv+exam_interval_len+1, rec_t_gap))
    
    return trange, amplf, exam_interval_len, rec_t_gap


def grid_search_stationary(config: SimulationConfig, n_jobs=-1, save_results=True) -> GridSearchResultStationary:
    """
    Perform grid search for a ring attractor network in the stationary case
    """
    # Parameter used to adaptively end simulations
    t_max1 = config.tau * 10 
    t_max2 = config.tau * 50 
    
    # Get parameter combinations
    param_combinations = list(ParameterGrid(config.search_pars))
    
    def process_single_params(pars):
        """Process a single parameter set"""
        par_list = [pars[config.par_names[j]] for j in range(config.par_num)]
        weight = config.get_weight(par_list)
        s0 = config.init_activity() # initial state
        
        y, t, network_eval, network_evaldes = \
            inte_check_sta(config.net_diff_equa, 
                            t_max1, 
                            t_max2, 
                            s0, 
                            weight, 
                            config.tau, 
                            config.get_barray(),
                            config.theta_num, 
                            config.actfun)
        
        return par_list, y, t, network_eval, network_evaldes
    
    # Run parallel computation
    print(f"Running parallel grid search with {len(param_combinations)} parameter combinations")
    results = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(
        delayed(process_single_params)(pars) for pars in param_combinations
    )
    
    # Initialize result arrays
    network_acvs = np.zeros(config.search_num, dtype='object')
    network_evals = np.ones(config.search_num, dtype='U30')
    network_evaldes = np.zeros(config.search_num, dtype='object')
    network_pars = np.zeros((config.search_num, config.par_num))
    network_ts = np.zeros(config.search_num, dtype='object')
    
    # Collect results
    for i, (par_list, y, t, eval, eval_des) in enumerate(results):
        network_pars[i] = par_list
        network_acvs[i] = np.array(sep_slist(y, config.ring_num))
        network_ts[i] = t
        network_evals[i] = eval
        network_evaldes[i] = eval_des
    
    # Create result dataclass
    net_sta = GridSearchResultStationary(
        eval=network_evals,
        eval_des=network_evaldes,
        activity=network_acvs,
        par=network_pars,
        time=network_ts
    )
    if save_results:
        save_grid_search_results('stationary', net_sta, config)
    
    return net_sta
'''
A module that simulate a ring attractor network's behavior given fixed input. 
It adaptively stops simulation based on coded criterion: if it is deem to be 
stable or unstable.

Siyuan Mei (mei@bio.lmu.de)
2024
'''
# pyright: reportUnboundVariable = false
import numpy as np
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed

from HD_utils.network import *
from HD_utils.exam import *
from HD_utils.comput_property import *
from HD_utils.dataclass import *
from HD_utils.IO import save_grid_search_results


def prepare_exam_time_points(tau, t_max1, t_max2, rec_t_gap, exam_interval, store_mag):
    '''
    Prepare time points for examination intervals.
    
    Parameters:
        tau: int
            Time constant of the network. default: 20 ms
        t_max1: int or None    
            Start examination after t_max1 (ms).
            if None, default to tau * 5 (100 ms)
        t_max2: int or None
            The maximum time for the numerical integration (ms).
            if None, default to tau * 50 (1000 ms)
        rec_t_gap: int or None
            the activity is recorded every rec_t_gap (ms).
            if None, default to tau (20 ms)
        exam_interval: int or None
            Perform examination after every exam_interval (ms).
            if None, default to tau * 5 (100 ms).
    Returns:
        trange: list of 1darray
            Each 1darray contains time points to record activity
            in one exam interval. 
        store_mag: int
            The activity is stored every store_mag time points.
    '''
    t_max1 = tau * 5 if t_max1 is None else t_max1
    t_max2 = tau * 50 if t_max2 is None else t_max2
    rec_t_gap = tau if rec_t_gap is None else rec_t_gap
    exam_interval = tau * 5 if exam_interval is None else exam_interval
    store_mag = int(np.ceil(tau / rec_t_gap)) if store_mag is None else store_mag
    
    trange = []
    for ti, tv in enumerate(range(0, t_max2-exam_interval+1, exam_interval)):
        trange.append(np.arange(tv, tv+exam_interval+1, rec_t_gap))
        
    if (t_max1 < rec_t_gap) | (t_max2 < rec_t_gap):
        raise ValueError('t_max1 or t_max2 must be bigger than exam interval (default 100)')

    return trange, store_mag, exam_interval, rec_t_gap, t_max1, t_max2

def inte_check_move(net_diff_equa, s0, w, tau, b, theta_num, actfun, 
                    t_max1=None, t_max2=None, rec_t_gap=None, exam_interval=None, 
                    rtol=1e-6, rdtol=1e-6, rtolflat=0.01, rtol_bumpyflat=0.1, store_mag=None,
                    theta_range=None, dtheta=None):
    '''
    Do the numerical integration and check the validity of the bump shape in a ring
    attractor network for the moving case (i.e., the bump is rotating).


    Parameters
    ----------
    net_diff_equa : function
        Differential equation of the network.
    s0: 1darray (theta_num * ring_num)
        Initial state of the network.
    w: 2darray (theta_num * ring_num, theta_num * ring_num)
        Weight matrix of the network.
    tau: int
        Time constant of the network. default: 20 ms
    b: int or float
        Constant external input of the network.
    theta_num: int
        Number of neurons in each ring.
    actfun: function
        Activation function of the network.
    t_max1: int or None    
        Start examination after t_max1 (ms).
        if None, default to tau * 5 (100 ms)
    t_max2: int or None
        The maximum time for the numerical integration (ms).
        if None, default to tau * 50 (1000 ms)
    rec_t_gap: int or None
        the activity is recorded every rec_t_gap (ms).
        if None, default to tau (20 ms)
    exam_interval: int or None
        Perform examination after every exam_interval (ms).
        if None, default to tau * 5 (100 ms).
    rtol: float, optional
        Relative tolerance for the numerical integration. The default is 1e-6.
    rdtol: float, optional
        Relative tolerance for calculating absolute tolerance 
        in integration (atol = rdtol * np.ptp(y)).
    rtolflat: float, optional
        Relative tolerance for checking whether the bump is flat.
    rtol_bumpyflat: float, optional
        Relative tolerance for checking whether the bump is bumpy flat.
    store_mag: int or None
        The activity is returned every store_mag time points to save memory.
        if None, default to ceil(tau / rec_t_gap).
    theta_range: 1darray or None
        The range of thetas of the network.
        if None, equally spaced between 0 and 2pi with theta_num points.
    dtheta: float or None
        The difference between two adjacent thetas of the network.
        if None, dtheta = 2pi/theta_num.

    Returns
    -------
    y: 2darray (theta_num * ring_num, t_num)
        Two ring: (left|right, t_num). Three ring: (symmetric|left|right, t_num).
        t_num is the number of all the recorded time points.
    t: 1darray (t_num)
        Recorded time points, whose length is determined by when the numerical 
        integration terminates.
    evals: str
        Evaluation of the network used for further analysis.
    v: 1darray (ring_num)
        Velocity of the bump. If the bump is not stably moving, the velocity is nan.
    '''
    trange, store_mag, exam_interval, rec_t_gap, t_max1, t_max2 = \
        prepare_exam_time_points(tau, t_max1, t_max2, rec_t_gap, exam_interval, store_mag)
    
    dtheta = (2*np.pi)/theta_num if dtheta is None else dtheta
    funargs = [w, tau, b, dtheta, actfun]
    ring_num = s0.shape[0] // theta_num

    # Initial values of variables
    evals = '1' # initial value of examination
    v = np.array([np.nan] * ring_num)
        
    # Numerical integration and examination
    for ti, trangev in enumerate(trange):
        # initial activity for this exam interval
        y0 = s0 if ti == 0 else net_dynamics.y[:,-1].copy()

        # separate y0 into a list of each ring's activity
        y0list = sep_slist(y0, ring_num) 
        # compute atol
        atollist = [0] * ring_num
        for i in range(ring_num):
            atollist[i] = rdtol*np.ptp(y0list[i])
        atol = np.min(atollist)
        # Note that atol is not computed for each solve_ivp step
        # but for initial value of the exam interval only
        
        # Solve ODE
        net_dynamics = solve_ivp(net_diff_equa, (min(trangev),max(trangev)), y0, t_eval=trangev, 
                                    args=funargs, events=explode_event, rtol=rtol, atol=atol)
        # store results
        if ti == 0:
            y, t = net_dynamics.y, net_dynamics.t
            yt, tt = y, t
        else:
            yt, tt = net_dynamics.y, net_dynamics.t
            y, t = net_var_append([y, t], [yt, tt])

        # If the integration fails, return 'singularity'
        if net_dynamics.success == False:
            return y[:,::store_mag], t[::store_mag], 'singularity', v
        
        
        # whether network explode
        if len(net_dynamics.t_events[0]) > 0:
            evals = 'explode'
            return y[:,::store_mag], t[::store_mag], evals, v
        
        # Examine the shape of the activity profile after t_max1
        
        if min(trangev) >= t_max1:
            # Separate activity into a list of each ring's activity
            slist = sep_slist(y, ring_num)
            
            _, se_list, mean_acv_list = prep_exam_var(slist, ring_num)

            # Zero flat check
            evals, _ = zeroflat_check(mean_acv_list, ring_num, [])
            if evals == 'flat':
                return y[:,::store_mag], t[::store_mag], evals, v
            
            # Relative flat check
            evals, _, _, _ = rel_flat_check(
                se_list, mean_acv_list, ring_num, rtolflat, rtol_bumpyflat, theta_num, [])
            if evals == 'flat':
                return y[:,::store_mag], t[::store_mag], evals, v
            
            # Examine Whether the bump only have one single peak
            evals, _ = exam_peak_num(slist, [])
            if evals == 'multiple peaks':
                return y[:,::store_mag], t[::store_mag], evals, v
            
            # compute velocity
            # only use the activity from the last exam interval to compute velocity
            st_list = sep_slist(yt, ring_num) 
            evals, v = compute_velocity(st_list, theta_num, rec_t_gap, theta_range=theta_range)
            
            # return the activity if bump reaches a stable state
            if (evals in ('stable moving', 'stationary')):
                return y[:,::store_mag], t[::store_mag], evals, v
        
    # Do not converge after t_max2
    if evals == '1':
        evals = 'unstable'
    
    return y[:,::store_mag], t[::store_mag], evals, v


def grid_search_moving_old(config: SimulationConfig, net_sta: GridSearchResultStationary):
    
    # Initialize storage variables
    network_eval_moving = np.zeros(( config.search_num, len(config.inputs) ), dtype='U30')
    network_eval_moving_sum = np.copy(net_sta.eval)
    network_vvcor = np.zeros((config.search_num, config.ring_num, 2))
    network_acvs_moving = np.zeros((config.search_num, len(config.inputs)), dtype='object')
    network_ts_moving = np.zeros((config.search_num, len(config.inputs)), dtype='object')
    Vels = np.zeros((config.search_num, len(config.inputs), config.ring_num))

    # Loop over all valid stationary networks
    for i in tqdm(net_sta.valid_id): 
        # Initial state is the stable state in the stationary case
        s1 = net_sta.get_final_activity(i)
        for ratioi, ratiov in enumerate(config.inputs):
            w = config.get_weight(net_sta.par[i], ratiov)

            y, t, network_eval_moving[i,ratioi], Vels[i,ratioi] = inte_check_move(
                config.net_diff_equa, s1, w, 
                config.tau, config.get_barray(ratiov), 
                config.theta_num, config.actfun)
            
            # Store values
            if config.ring_num == 1:
                network_acvs_moving[i,ratioi] = y
            elif config.ring_num in (2, 3):
                network_acvs_moving[i,ratioi] = np.array(sep_slist(y, config.ring_num))
            network_ts_moving[i,ratioi] = t
        
        ## Check if the network moves stably for all inputs
        # If yes, check if the correlation between input and velocity > 0.99
        network_eval_moving[i][config.zeroid] = 'stable moving'
        network_eval_moving_sum[i] = 'valid stationary shape'
        
        if np.all(network_eval_moving[i] == 'stable moving'):
            network_vvcor[i] = cal_correlation(config.inputs, Vels[i])
            network_eval_moving_sum[i] = exam_vv_linearity(network_vvcor[i])

    # Create dataclass to store results
    result = GridSearchResultMoving(
        velocity=Vels,
        eval=network_eval_moving,
        correlation=network_vvcor,
        activity=network_acvs_moving,
        time=network_ts_moving,
        eval_sum=network_eval_moving_sum
    )
    
    save_grid_search_results('moving', result, config)
    
    return result


def grid_search_moving(config: SimulationConfig, net_sta: GridSearchResultStationary, n_jobs=-1) -> GridSearchResultMoving:
    """
    Parallel version of grid_search_moving
    """
    
    def process_single_network(i):
        """Process a single network configuration for all inputs"""
        # Initialize storage for this network
        network_eval_moving_i = np.zeros(len(config.inputs), dtype='U30')
        network_acvs_moving_i = np.zeros(len(config.inputs), dtype='object')
        network_ts_moving_i = np.zeros(len(config.inputs), dtype='object')
        Vels_i = np.zeros((len(config.inputs), config.ring_num))
        
        # Initial state is the stable state in the stationary case
        s1 = net_sta.get_final_activity(i)
        
        for ratioi, ratiov in enumerate(config.inputs):
            weight = config.get_weight(net_sta.par[i], ratiov)

            y, t, network_eval_moving_i[ratioi], Vels_i[ratioi] = inte_check_move(
                config.net_diff_equa, s1, weight, 
                config.tau, config.get_barray(ratiov), 
                config.theta_num, config.actfun)
            
            # Store values
            if config.ring_num == 1:
                network_acvs_moving_i[ratioi] = y
            elif config.ring_num in (2, 3):
                network_acvs_moving_i[ratioi] = np.array(sep_slist(y, config.ring_num))
            network_ts_moving_i[ratioi] = t
        
        # Check if the network moves stably for all inputs
        network_eval_moving_i[config.zeroid] = 'stable moving'
        network_eval_moving_sum_i = 'valid stationary shape'
        network_vvcor_i = np.zeros((config.ring_num, 2))
        
        if np.all(network_eval_moving_i == 'stable moving'):
            network_vvcor_i = cal_correlation(config.inputs, Vels_i)
            network_eval_moving_sum_i = exam_vv_linearity(network_vvcor_i)
        
        return network_eval_moving_i, network_acvs_moving_i, network_ts_moving_i, Vels_i, network_vvcor_i, network_eval_moving_sum_i
    
    # Run parallel computation
    print(f"Running parallel grid search moving with {len(net_sta.valid_id)} valid networks")
    results = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(
        delayed(process_single_network)(i) for i in net_sta.valid_id
    )
    
    # Initialize result arrays
    network_eval_moving = np.zeros((config.search_num, len(config.inputs)), dtype='U30')
    network_eval_moving_sum = np.copy(net_sta.eval)
    network_vvcor = np.zeros((config.search_num, config.ring_num, 2))
    network_acvs_moving = np.zeros((config.search_num, len(config.inputs)), dtype='object')
    network_ts_moving = np.zeros((config.search_num, len(config.inputs)), dtype='object')
    Vels = np.zeros((config.search_num, len(config.inputs), config.ring_num))
    
    # Collect results
    for idx, i in enumerate(net_sta.valid_id):
        eval_i, acvs_i, ts_i, vels_i, vvcor_i, eval_sum_i = results[idx]
        network_eval_moving[i] = eval_i
        network_acvs_moving[i] = acvs_i
        network_ts_moving[i] = ts_i
        Vels[i] = vels_i
        network_vvcor[i] = vvcor_i
        network_eval_moving_sum[i] = eval_sum_i
    
    # Create dataclass to store results
    result = GridSearchResultMoving(
        velocity=Vels,
        eval=network_eval_moving,
        correlation=network_vvcor,
        activity=network_acvs_moving,
        time=network_ts_moving,
        eval_sum=network_eval_moving_sum
    )
    
    save_grid_search_results('moving', result, config)
    
    return result




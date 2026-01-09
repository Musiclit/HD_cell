import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import HD_utils.circular_stats as cstat
import matplotlib.pyplot as plt


def if_peak_loc_stable2(peak_s_time, rec_t_gap, slist, distol=1):
    '''
    Check if the peak location of any ring is unstable, if yes, return False.
    
    Parameters:
    peak_s_time: float
        The peak location should be stable within the peak_s_time interval.
    rec_t_gap: float
        The time gap between two recorded time points.
    slist: list of np.array (theta_num, time_num)
        The activity of each ring, order: [symmetric, left, right]
    distol: float
        The tolerance of the peak location change, in the unit of neuron number.
        Default 1, means the peak location should not change more than 1 neuron.
        
    Returns:
    is_stable: bool
        True if the peak location is stable for all rings, else False.
    '''
    theta_num = len(slist[0])
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta)
    
    # Convert distol in neuron number to radian
    distol = distol * dtheta

    # Determine the index to select the time intervals to check
    # peak location stability. example: peak_s_time = rec_t_gap, 
    # then smp_id = (-1,-2)
    smp_id_c = peak_s_time // rec_t_gap + 2
    smp_id = -np.arange(1, smp_id_c)
    
    # convert theta_range to 2d array for cstat.mean calculation
    theta_range_2d = np.tile(theta_range, (len(smp_id), 1)).T
    
    is_stable = True
    
    # Check the stability ring by ring
    for s in slist:
        se = s - np.min(s, axis=0) # zero-offset activity
        
        # Calculate the peak location with in the peak_s_time interval
        peakloc = cstat.mean(theta_range_2d, se[:,smp_id], axis=0)
        
        # Check the peak location stability
        peakloc_same = np.isclose(
            np.max(
                np.abs(
                    cstat.cdiff(peakloc[:1], peakloc)
                    )
                )
            , 0, atol=distol) 
        # Peakloc[:1]: the first peak location within the interval
        # use [:1] rather than [0] to keep it as an array
        
        # If one ring is unstable, return False
        # without checking other rings
        if not peakloc_same:
            is_stable = False
            break
    
    return is_stable


def exam_stbump_shape(slist, network_evals):
    # It only examine whether the activity profile has multiple peaks, while the exam of flat and explode is elsewhere
    network_evaldes = []
    # If have examined while doing ODE --- Need change if events order in ODE solver change
    if network_evals in ('explode', 'equal flat', 'unequal flat', 'unstable', 'flat'):
        return network_evals, network_evaldes
    # Examine big problems
    for si, s in enumerate(slist):
        # Examine whether it has multiple prominent peaks
        max_vert_dis = np.ptp(s[:,-1])
        peaks1, _ = find_peaks(s[:,-1], prominence=0.1*max_vert_dis)     #---------------------------- had tune
        peaks2, _ = find_peaks(s[:,-1], height=0.3*np.max(s[:,-1]) + np.min(s[:,-1]))     #---------------------------- had tune
        peak_num = len(peaks1) + len(peaks2)
        if peak_num > 2:
            network_evals = 'bad shape'
            network_evaldes.append(f'multiple peaks s{si}')
#         # Examine whether activation is flat
#         rel_acc = np.abs( ( np.mean(s[:,-1]) - s[:,-1] )/s[:,-1] )
#         pred_correct_num = np.sum(rel_acc < 1e-2)              # ----------------------hand tune
#         if pred_correct_num == theta_num:
#             network_evals = 'bad shape'
#             network_evaldes.append(f'flat s{si}')
    # if not( np.isclose( np.mean(slist[0][:,-1]), np.mean(slist[1][:,-1]), rtol=0.1 ) ):              # ----------------------hand tune
    #     network_evals = 'bad shape'
    #     network_evaldes.append(f'two ring asymmetry')
    # Minor problems
    if network_evals == '1':
        for i, s in enumerate(slist):
            # Examine whether it has multiple small peaks
            minheight = 0.1*np.max(s[:,-1]) + np.min(s[:,-1]) #---------------------------- had tune
            peaks, props = find_peaks(s[:,-1], prominence=0.01*max_vert_dis, height=minheight) #---------------------------- had tune
            peak_num = len(peaks)
            if peak_num > 1:
                network_evals = 'multiple small peaks'
                break
    # Network activation as expected
    if network_evals == '1':
        network_evals = 'valid'
    return network_evals, network_evaldes


def exam_peak_num(slist, network_evaldes):
    '''
    Examine the number of peaks in the final activity profile.
    
    Parameters:
    slist: list of np.array (theta_num, time_num)
        The activity of each ring, order:
        [one_ring], [left_ring, right_ring], [symmetric_ring, left_ring, right_ring]
    network_evaldes: list of str
        The description of all the evaluations, to be appended if
        one/some rings have more than one peak.
        
    Returns:
    network_evals: str
        '1' if only one peak, 'multiple peaks' if multiple prominent peaks.
        If there are multiple small peaks, still return '1' but will add
        "multiple small peaks" in network_evaldes.
    network_evaldes: list of str
        The description of all the evaluations, to be appended if
        one/some rings have more than one peak.
    '''
    network_evals = '1'
    # Examine prominent peaks ring by ring
    for si, s in enumerate(slist):
        max_vert_dis = np.ptp(s[:,-1]) # peak to bottom value of the final activity
        # The prominence of a peak measures how much a peak stands out from the 
        # surrounding baseline of the signal and is defined as the vertical 
        # distance between the peak and its lowest contour line.
        peaks1, _ = find_peaks(s[:,-1], prominence=0.1*max_vert_dis)
        peaks2, _ = find_peaks(s[:,-1], height=0.3*np.max(s[:,-1]) + np.min(s[:,-1]))
        peak_num = len(peaks1) + len(peaks2)
        
        if peak_num > 2:
            network_evals = 'multiple peaks'
            network_evaldes.append(f'multiple peaks s{si}')
    
    # Examine small peaks ring by ring only if no multiple prominent peaks
    if network_evals == '1':
        for si, s in enumerate(slist):

            minheight = 0.1*np.max(s[:,-1]) + np.min(s[:,-1])
            # Small peaks should satisfy both prominence and height criteria
            peaks, _ = find_peaks(s[:,-1], prominence=0.01*max_vert_dis, height=minheight) 
            peak_num = len(peaks)
            
            if peak_num > 1:
                network_evaldes.append(f'multiple small peaks s{si}')
                
    return network_evals, network_evaldes


def exam_vv_linear_relation(inputs, vs, criterionw=0.99, criterionm=0.99, midrange=None):
    if midrange is None:
        midrange = len(inputs) // 2
    rs = np.zeros(4) # all r left, all r right, mid r left, mid r right
    for vi in range(2):
        rs[vi+0] = pearsonr(inputs, vs[:,vi])[0]
        rs[vi+2] = pearsonr(inputs[:midrange], vs[:midrange,vi])[0]
    if np.any(rs[:2] > criterionw):
        evals = 'linear'
    elif np.any(rs[2:] > criterionm):
        evals = 'mid linear'
    else:
        evals = 'unlinear'
    return evals, rs


def exam_vv_linear_relation_3ring(inputs, vs, criterionw=0.99, criterionm=0.99, midrange=None):
    if midrange is None:
        midrange = len(inputs) // 2
    rs = np.zeros((3,2)) # 3 ring X whole, mid
    for i in range(3):
        rs[i,0] = pearsonr(inputs, vs[:,i])[0]
        rs[i,1] = pearsonr(inputs[:midrange], vs[:midrange,i])[0]
    if np.any(np.abs(rs[:,0]) > criterionw):
        evals = 'linear'
    elif np.any(np.abs(rs[1]) > criterionm):
        evals = 'mid linear'
    else:
        evals = 'unlinear'
    return evals, rs


def exam_vv_linear_relation_1ring(inputs, vs, criterionw=0.99, criterionm=0.99, midrange=None):
    if midrange is None:
        midrange = len(inputs) // 2
    rs = np.zeros(2) # all r left, all r right, mid r left, mid r right
    rs[0] = pearsonr(inputs, vs)[0]
    rs[1] = pearsonr(inputs[:midrange], vs[:midrange])[0]
    if rs[0] > criterionw:
        evals = 'linear'
    elif rs[1] > criterionm:
        evals = 'mid linear'
    else:
        evals = 'unlinear'
    return evals, rs


def exam_vv_linearity(rs, criterionw=0.99, criterionm=0.99):

    '''

    Parameter
    ---------
    rs: 2darray(ring_num, 2)
        Dimension 1: correlation for different rings.
        Dimension 2: (1) The correlation between inputs and velocity,
        (2) The correlation between inputs and velocity in the first half of the inputs.
    '''

    if np.any(np.abs(rs[:,0]) > criterionw):
        evals = 'linear moving'
    elif np.any(np.abs(rs[:,1]) > criterionm):
        evals = 'mid-linear moving'
    else:
        evals = 'nonlinear moving'

    return evals


def prep_exam_var(slist, ring_num):
    '''
    Calculate the peak to bottom value of the final activity, 
    zero-offset final activity,
    and the mean of the zero-offset final activity
    
    Parameters:
    slist: list of np.array (theta_num, time_num)
        The activity of each ring, order:
        [one_ring], [left_ring, right_ring], [symmetric_ring, left_ring, right_ring]
    ring_num: int
        Number of rings in the network
        
    Returns:
    sptp_list: list of float
        The peak to bottom value of the final activity of each ring
    se_list: list of np.array (theta_num,)
        The zero-offset final activity of each ring
    mean_acv_list: list of float
        The mean of the zero-offset final activity of each ring
    '''
    # Initialize storage variables
    sptp_list = [0] * ring_num
    se_list = [0] * ring_num
    mean_acv_list = [0] * ring_num
    
    # Calculate the return variables ring by ring
    for i in range(ring_num):
        s_end = slist[i][:,-1] # final activity
        sptp_list[i] = np.ptp(s_end)
        se_list[i] = s_end - np.min(s_end)
        mean_acv_list[i] = np.mean(se_list[i])

    return sptp_list, se_list, mean_acv_list


def zeroflat_check(mean_acv_list, ring_num, evaldes, zerotol=1e-3):
    '''
    Check whether the mean of the zero-offset final activity is close to zero
    for all rings. If so, return 'flat', else return '1'
    
    Parameters:
    mean_acv_list: list of float
        The mean of the zero-offset final activity of each ring
    ring_num: int
        Number of rings in the network.
    evaldes: list of str
        The description of all the evaluations, to be appended if 
        one/some rings are flat.
    zerotol: float
        If the mean of the zero-offset final activity is
        smaller than zerotol, it is considered as zero.
        
    Returns:
    evals: str
        'flat' or '1'
    evaldes: list of str
        The description of all the evaluations, appended if
        one/some rings are flat.
    '''
    evals = '1' # If not flat, return '1'
    flat_temp = [False] * ring_num # If flat for each ring
    
    for i in range(ring_num):
        if np.isclose(mean_acv_list[i], 0, atol=zerotol): 
            flat_temp[i] = True
            evaldes.append(f'mean s{i} == 0')
    if np.all(flat_temp):
        evals = 'flat'

    return evals, evaldes


def rel_flat_check(se_list, mean_acv_list, ring_num, rtolflat, rtol_bumpyflat, 
                   theta_num, evaldes):
    '''
    Check the relative flatness of the zero-offset final activity.
    
    Parameters:
    se_list: list of np.array (theta_num,)
        The zero-offset final activity of each ring
    mean_acv_list: list of float
        The mean of the zero-offset final activity of each ring
    ring_num: int
        Number of rings in the network.
    rtolflat: float
        The relative tolerance to consider the activity as flat.
        default 0.01, according to the function that calls this function.
    rtol_bumpyflat: float
        The relative tolerance to consider the activity as bumpy flat.
        default 0.1, according to the function that calls this function.
    theta_num: int
        Number of neurons in each ring.
    evaldes: list of str
        The description of all the evaluations, to be appended if
        one/some rings are flat.
    
    Returns:
    evals: str
        'flat' or '1'
    evaldes: list of str
        The description of all the evaluations, appended if
        one/some rings are flat.
    small_rvar_num: int
        The number of neurons (including all rings) with relative variation 
        smaller than rtolflat (default 0.01).
    medium_rvar_num: int
        The number of neurons (including all rings) with relative variation
        smaller than rtol_bumpyflat (default 0.1).
    '''
    evals = '1'
    small_rvar_num = 0
    medium_rvar_num = 0
    rvar = [0] * ring_num
    
    for i in range(ring_num):
        # relative variation = |mean - s(theta)| / |mean|
        rvar[i] = np.abs(mean_acv_list[i] - se_list[i] ) / np.abs(mean_acv_list[i])
        small_rvar_num += np.sum(rvar[i] < rtolflat)
        medium_rvar_num += np.sum(rvar[i] < rtol_bumpyflat)

    if (small_rvar_num == theta_num * ring_num): 
        evals = 'flat'
        evaldes.append('relative flat')
    
    return evals, evaldes, small_rvar_num, medium_rvar_num


def check_stamov_range(network_eval_moving, index_check):
    '''
    Intend to check the network with valid stationary shape, so
    omit stamov range that is full
    '''
    input_num = network_eval_moving.shape[1]
    search_num = network_eval_moving.shape[0]

    linear_range = np.zeros((search_num, 2), dtype=int)
    for i in index_check:
        eval_moving = network_eval_moving[i]
        for j in range(1, input_num//2):
            check_range = np.arange(j, input_num-j)
            if np.all(network_eval_moving[i, check_range] == 'stable moving'):
                linear_range[i] = [j, input_num-j]
                break
    return linear_range


def check_vv_linearity2(inputs, Vels, linear_range, index_check, network_vvcor, network_eval_moving_sum, criterionm=0.99):
    '''
    Check the linearity of the velocity vs input for the case that
    the stable moving range is not full

    Parameters:
    network_vvcor: np.array(search_num, ring_num, 2) Dim3: 0: fullrange correlation; 1: midrange correlation
    '''
    id_l = (np.abs(inputs - min(inputs)/2)).argmin()
    id_u = (np.abs(inputs - max(inputs)/2)).argmin() + 1
    search_num = Vels.shape[0]
    ring_num = Vels.shape[2]
    
    for i in index_check:
        lrl = linear_range[i, 0]
        lru = linear_range[i, 1]
        for r in range(ring_num):
            if (lrl < id_l) & (lru > id_u):
                network_vvcor[i, r, 0] = pearsonr(inputs[lrl:lru], Vels[i, lrl:lru, r])[0]
                network_vvcor[i, r, 1] = pearsonr(inputs[id_l:id_u], Vels[i, id_l:id_u, r])[0]
            elif (lrl == id_l) & (lru == id_u):
                network_vvcor[i, r, 1] = pearsonr(inputs[id_l:id_u], Vels[i, id_l:id_u, r])[0]
        if (lrl <= id_l) & (lru >= id_u):
            network_eval_moving_sum[i] = 'nonlinear moving'
            if np.any(np.abs(network_vvcor[i, :, 1]) > criterionm):
                network_eval_moving_sum[i] = 'mid-linear moving'
            
    return network_vvcor, network_eval_moving_sum
'''
Used to compute properties of netowrks

Siyuan Mei (mei@bio.lmu.de)
2024
'''
# pyright: reportUnboundVariable =false
import numpy as np
import HD_utils.circular_stats as cstat
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from HD_utils.network import max0x
import warnings
import matplotlib.pyplot as plt
from scipy.stats import linregress

def cal_move_bump_property(valid_index_s, network_eval_moving, network_eval_moving_des, inputs, Vels, search_num, network_acvs, network_acvs_moving, theta_range, b0, acvf=max0x):
    '''Calculate properties of a moving activity bump, deprecated'''
    valid_index_stable_move = valid_index_s[np.where( np.sum(network_eval_moving[valid_index_s] == 'stable moving', axis=1) == len(inputs) )[0]]
    linear_move_bol = np.mean(network_eval_moving_des[:,:2], axis=1) > 0.99
    valid_index_linear_move = np.where(linear_move_bol)[0]
    mid_linear_move_bol = np.mean(network_eval_moving_des[:,2:], axis=1) > 0.99
    valid_index_mid_linear_move = np.where(mid_linear_move_bol)[0]
    move_bols = [mid_linear_move_bol, linear_move_bol]
    print('Stable still, Stable moving, Half Linear, Whole Linear')
    print(len(valid_index_s), len(valid_index_stable_move), len(valid_index_mid_linear_move), len(valid_index_linear_move))
    net_mov_lin, max_v, lin_range, lin_range_id = exam_linearlity(inputs, Vels, search_num, valid_index_stable_move, lin_thres = 0.99)
    # net_mov_lin: the correlation between network velocity and inputs until a specific input value. 3rd dim: (correlation, if > lin_thres)
    # max_v: the maximum velocity of the network in the linear range
    # lin_range: the input range where the network velocity is linearly related to the input
    max_amplitude = compute_max_ampl(search_num, inputs, valid_index_stable_move, network_acvs, network_acvs_moving)
    skewness, skewness_mean = cal_skewness(search_num, inputs, valid_index_stable_move, network_acvs, network_acvs_moving, theta_range)
    slopes, slopes_mean =  cal_slope(search_num, inputs, valid_index_linear_move, Vels)
    vvcor = np.mean(network_eval_moving_des[:,:2], axis=1)
    vvmidcor = np.mean(network_eval_moving_des[:,2:], axis=1)
    ampl_vel_cor = cal_ampl_vel(search_num, valid_index_stable_move, max_amplitude, Vels, lin_range_id)
    bump_height, max_firate = cal_height_a_firate(search_num, inputs, valid_index_stable_move, network_acvs_moving, b0, acvf)
    vel_fir_cor = cal_ampl_vel(search_num, valid_index_stable_move, max_firate, Vels, lin_range_id)
    return valid_index_stable_move, valid_index_linear_move, valid_index_mid_linear_move, move_bols, net_mov_lin, max_v, lin_range, lin_range_id, \
    max_amplitude, skewness, skewness_mean, slopes, slopes_mean, vvcor, vvmidcor, ampl_vel_cor, bump_height, max_firate, vel_fir_cor

  
def cal_move_bump_property2(valid_index_s, network_eval_moving, network_eval_moving_des, inputs, Vels, search_num, network_acvs, network_acvs_moving, theta_range, b0, acvf=max0x):
    '''Calculate properties of a moving activity bump v2, deprecated'''
    valid_index_stable_move = valid_index_s[np.where( np.sum(network_eval_moving[valid_index_s] == 'stable moving', axis=1) == len(inputs) )[0]]
    linear_move_bol = np.mean(network_eval_moving_des[:,:2], axis=1) > 0.99
    valid_index_linear_move = np.where(linear_move_bol)[0]
    mid_linear_move_bol = np.mean(network_eval_moving_des[:,2:], axis=1) > 0.99
    valid_index_mid_linear_move = np.where(mid_linear_move_bol)[0]
    move_bols = [mid_linear_move_bol, linear_move_bol]
    print('Stable still, Stable moving, Half Linear, Whole Linear')
    print(len(valid_index_s), len(valid_index_stable_move), len(valid_index_mid_linear_move), len(valid_index_linear_move))
    net_mov_lin, max_v, lin_range, lin_range_id = exam_linearlity(inputs, Vels, search_num, valid_index_stable_move, lin_thres = 0.99)
    # net_mov_lin: the correlation between network velocity and inputs until a specific input value. 3rd dim: (correlation, if > lin_thres)
    # max_v: the maximum velocity of the network in the linear range
    # lin_range: the input range where the network velocity is linearly related to the input
    amplitude = cal_avg_acv(search_num, inputs, valid_index_stable_move, network_acvs_moving, b0)
    skewness, skewness_mean = cal_skewness(search_num, inputs, valid_index_stable_move, network_acvs, network_acvs_moving, theta_range)
    slopes, slopes_mean =  cal_slope(search_num, inputs, valid_index_linear_move, Vels)
    vvcor = np.mean(network_eval_moving_des[:,:2], axis=1)
    vvmidcor = np.mean(network_eval_moving_des[:,2:], axis=1)
    ampl_vel_cor = cal_ampl_vel(search_num, valid_index_stable_move, amplitude, Vels, lin_range_id)
    bump_height, max_firate = cal_height_a_firate(search_num, inputs, valid_index_stable_move, network_acvs_moving, b0, acvf)
    vel_fir_cor = cal_ampl_vel(search_num, valid_index_stable_move, max_firate, Vels, lin_range_id)
    return valid_index_stable_move, valid_index_linear_move, valid_index_mid_linear_move, move_bols, net_mov_lin, max_v, lin_range, lin_range_id, \
    amplitude, skewness, skewness_mean, slopes, slopes_mean, vvcor, vvmidcor, ampl_vel_cor, bump_height, max_firate, vel_fir_cor


def compute_velocity(netacvs, theta_num, rect_t_gap, loc_dif_tol_mild=1, loc_dif_tol_strict=1e-2, vdif_atol=1, theta_range=None):

    '''
    Compute the velocity of a moving activity bump
    
    Parameters:
        netacvs: List[np.ndarray[float, (theta_num, time_num)]]
    theta_num: int
        Number of neurons in each ring.
    rect_t_gap: int
        The activity is recorded every rect_t_gap (ms).
    loc_dif_tol_mild: float, optional
        The mild absolute tolerance for checking whether the velocity is zero. The default is 1.
        If peak location difference in one exam interval < loc_dif_tol_mild AND
        the moving direction is not consistent, then the bump is considered stationary.
    loc_dif_tol_strict: float, optional
        The strict absolute tolerance for checking whether the velocity is zero. The default is 1e-2.
        If peak location difference in one exam interval < loc_dif_tol_strict, 
        then the bump is considered stationary.
    vdif_atol: float, optional
        The absolute tolerance for checking whether the velocity is stable. The default is 1.
    theta_range: 1darray or None
        The thetas of the network.

    Returns:
        evals: str
            '1': not evaluated
            'stationary': bump is stationary
            'stable moving': bump is stably moving
            'shape&vel unstable': bump shape and velocity are unstable
            'shape unstable': bump shape is unstable
            'vel unstable': bump velocity is unstable
        v_mean: 1darray (ring_num,)
            Mean velocity of each ring (turn/s)
    '''
    # convert the tolerances from neuron to radian
    vdif_atol = vdif_atol / theta_num * 2 * np.pi
    loc_dif_tol_strict = loc_dif_tol_strict / theta_num * 2 * np.pi
    loc_dif_tol_mild = loc_dif_tol_mild / theta_num * 2 * np.pi

    # Determine the dimension
    ring_num = len(netacvs)
    vnum = netacvs[0].shape[1] - 1 # calculable velocity = time points - 1

    # Initialize evaluation storage variables
    evals = '1'
    eval_array = np.array(['1'] * ring_num, dtype='U30')
    # Initialize velocity storage variables
    v_mean = np.array([np.nan] * ring_num)
    v = np.zeros((ring_num, vnum))
    
    # Prepare theta variables
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) if theta_range is None else theta_range
    
    
    for j, s in enumerate(netacvs): # For each ring
        # Prepare theta_range_2d for circular mean calculation
        # check if theta_range is different for different rings
        if isinstance(theta_range, list):
            theta_range_2d = np.tile(theta_range[j], (vnum+1, 1)).T
        else:
            theta_range_2d = np.tile(theta_range, (vnum+1, 1)).T
            
        # Determine the peak location (HD representation)
        se = s - np.min(s, axis=0)
        peakloc = cstat.mean(theta_range_2d, se, axis=0)
        
        # Check if the the peak location is stationary
        # i.e. the activity bump is not moving
        peak_loc_max_diff = np.max(np.abs(cstat.cdiff(peakloc[:1], peakloc)))
        
        peakloc_same_mild = np.isclose(peak_loc_max_diff, 0, atol=loc_dif_tol_mild)
        
        peakloc_same_strict= np.isclose(peak_loc_max_diff, 0, atol=loc_dif_tol_strict)
        move_directions = np.sign(np.diff(peakloc))
        mov_direction_irregular = np.abs(np.sum(move_directions)) < len(move_directions)
        
        # If either (1) peak location difference < 0.02 (default) neurons or
        # (2) peak location difference < 1 (default) neuron and
        # the moving direction is not consistent, 
        # Then the activity bump is considered stationary
        if ((peakloc_same_mild & mov_direction_irregular) | (peakloc_same_strict)): 
            evals = 'stationary'
            return evals, np.zeros(ring_num)
        
        # If it is moving, calculate the velocity
        peakloc = cstat.rerange_expand(peakloc)  # expand to [-inf, inf] 
        v[j] = np.diff(peakloc)

        # Check if the activity keeps the same shape
        ## atol = maximum activity difference between two consecutive neurons
        atol = np.max(np.abs(np.diff(s, axis=0))) 
        ## roll the activity to align the peak to the first neuron
        roll_shifts = np.cumsum(v[j] / dtheta)[::-1]
        roll_int_shifts = np.rint(np.concatenate((roll_shifts, [0]))).astype(int)  # add 0 for the last time point
        s_roll = np.array([np.roll(col, x) for col,x in zip(s.T, roll_int_shifts)]).T
        ## Whether the activity matches after rolling
        shape_match = np.all(np.isclose(np.max(s_roll, axis=1), np.min(s_roll, axis=1), atol=atol, rtol=1e-16))

        ## Whether velocity matches
        vel_match = np.isclose(max(v[j]), min(v[j]), atol=vdif_atol)
        
        # evaluate of both shape and velocity are stable
        if vel_match & shape_match:
            eval_array[j] = 'stable moving'
        else:
            eval_array[j] = f'shape match {int(shape_match)}; vel match {int(vel_match)}'
    
    # Final evaluation combining all rings
    if np.all(eval_array == 'stable moving'):
        evals = 'stable moving'
        v_mean = np.mean(v, axis=1) / (2*np.pi) * 1000 / rect_t_gap # turn/s
    elif np.any(eval_array == 'shape match 0; vel match 0'):
        evals = 'shape&vel unstable'
    elif np.any(eval_array == 'shape match 0; vel match 1'):
        evals = 'shape unstable'
    elif np.any(eval_array == 'shape match 1; vel match 0'):
        evals = 'vel unstable'
            
    return evals, v_mean

def readout_direction(s, theta_range):
    '''readout the direction represented by the ring attractor network'''
    # 2 ring separate, max value of each ring
    temp = theta_range[np.argmax(s, axis=1)]
    ## Left Ring Max
    pos_left_max = cstat.rerange( temp[0] )
    ## Right Ring Max
    pos_right_max = cstat.rerange( temp[1] )
    ## average the two max values
    pos_2_max_mean = cstat.rerange( cstat.mean(temp, axis=0) )
    
    # 2 ring separate, weighted average value of each ring
    temp = np.zeros((2,s.shape[2]))
    for ring in range(2):
        for time in range(s.shape[2]):
            temp[ring, time] = cstat.mean(theta_range, s[ring,:,time])
    # Left Ring Weighted Average
    pos_left_weightmean = cstat.rerange( temp[0] )
    # Right Ring Weighted Average
    pos_right_weightmean = cstat.rerange( temp[1] )
    ## average the weighted average
    pos_2_weightmean_mean = cstat.rerange( cstat.mean(temp, axis=0) )

    # max value as the final ring
    s_max = np.max(s, axis=0)
    ## max value of the maxed final ring
    pos_max_max = theta_range[np.argmax(s_max, axis=0)]
    ## Weight average value of the maxed final ring
    pos_max_weightmean = np.zeros(s_max.shape[1])
    for time in range(s_max.shape[1]):
        pos_max_weightmean[time] = cstat.mean(theta_range, s_max[:,time])
    pos_max_weightmean = cstat.rerange(pos_max_weightmean)

    # sum value as the final ring
    s_sum = np.sum(s, axis=0)
    ## max value of the sum final ring
    pos_sum_max = theta_range[np.argmax(s_sum, axis=0)]
    ## Weight average value of the sum final ring
    pos_sum_weightmean = np.zeros(s_sum.shape[1])
    for time in range(s_sum.shape[1]):
        pos_sum_weightmean[time] = cstat.mean(theta_range, s_sum[:,time])
    pos_sum_weightmean = cstat.rerange(pos_sum_weightmean)
    
    pos = [pos_2_max_mean, pos_2_weightmean_mean, pos_max_max, pos_max_weightmean, pos_sum_max, pos_sum_weightmean, 
          pos_left_max, pos_right_max, pos_left_weightmean, pos_right_weightmean]
    pos_names = ['two_max_mean', 'two_weight_mean', 'maxone_max', 'maxone_weight', 'sumone_max', 'sumone_weight',
                'left_max', 'right_max', 'left_weight', 'right_weight']
    return pos, pos_names


def compute_amplitude(search_num, inputs, valid_index_linear_move, network_acvs, network_acvs_moving):
    '''Calculate the max membrane potential before receiving inputs, outdated'''
    max_amplitude = np.zeros((search_num, len(inputs),2))
    for neti in valid_index_linear_move:
        for i in range(len(inputs)):
            for j in range(2):
                if i == 0:
                    max_amplitude[neti,i,j] = max(network_acvs[neti][j,:,-1])
                else:
                    max_amplitude[neti,i,j] = max(network_acvs_moving[neti,i][j,:,-1])
    return max_amplitude


def compute_skewness(search_num, inputs, valid_index_linear_move, network_acvs, network_acvs_moving, theta_range):
    skewness = np.zeros((search_num,len(inputs),2))
    '''Calculate the skewness of the activity profile'''
    for i in valid_index_linear_move:
        for inputi in range(len(inputs)):
            for j in range(2):
                if inputi == 0:
                    skewness[i,inputi,j] = cstat.skewness(theta_range, w=network_acvs[i][j,:,-1])
                else:
                    skewness[i,inputi,j] = cstat.skewness(theta_range, w=network_acvs_moving[i,inputi][j,:,-1])
    return skewness


def compute_max_ampl(search_num, inputs, valid_index_s, network_acvs, network_acvs_moving):
    '''calculate the peak to bottom differen of the membran potential before receiving inputs, rarely used'''
    max_amplitude = np.zeros((search_num, len(inputs),2))
    for neti in valid_index_s:
        for i in range(len(inputs)):
            for j in range(2):
                if i == 0:
                    max_amplitude[neti,i,j] = max(network_acvs[neti][j,:,-1]) - min(network_acvs[neti][j,:,-1])
                else:
                    max_amplitude[neti,i,j] = max(network_acvs_moving[neti,i][j,:,-1]) - min(network_acvs_moving[neti,i][j,:,-1])
    return max_amplitude


def cal_avg_acv(search_num, inputs, valid_index_s, network_acvs_moving, b0, addb=1, subb=1):
    '''Calculate the average membrane potential'''
    avg_acv = np.zeros((search_num, len(inputs), 2))
    for neti in valid_index_s:
        for i in range(len(inputs)):
            b = [b0 + addb * inputs[i], b0 - subb * inputs[i]]
            for j in range(2):
                avg_acv[neti,i,j] = np.mean(network_acvs_moving[neti,i][j,:,-1] + b[j])
    return avg_acv


def scalr2vec(bE, inputs):
    '''convert the inputs from a scaler to a vector to fit data format of simulation'''
    if (not isinstance(bE, (list, np.ndarray))):
        bE = [bE] * len(inputs)
    return bE


def cal_avg_acv_3ring(search_num, inputs, valid_index_s, network_acvs_moving, bE, bI, addb=1, subb=1):
    # the add b and sub is not correct
    avg_acv = np.zeros((search_num, len(inputs), 3))
    bE = scalr2vec(bE, inputs)
    bI = scalr2vec(bI, inputs)
    for neti in valid_index_s:
        for i in range(len(inputs)):
            b = [bE[i], bI[i] + addb *
                  inputs[i], bI[i] - subb * inputs[i]]
            for j in range(3):
                avg_acv[neti,i,j] = np.mean(network_acvs_moving[neti,i][j,:,-1] + b[j])
    return avg_acv


def exam_linearlity(inputs, Vels, search_num, valid_index_s, lin_thres = 0.99):
    exam_num = len(inputs) - 1
    net_mov_lin = np.zeros((search_num, exam_num, 2)) # 3rd dim: (correlation, if > lin_thres)
    max_v = np.zeros((search_num))
    lin_range = np.zeros((search_num))
    lin_range_id = np.zeros((search_num), dtype=int)
    for i in valid_index_s:
        for j in range(exam_num):
            net_mov_lin[i,j,0] = pearsonr(Vels[i][:j+2,0], inputs[:j+2])[0]
            net_mov_lin[i,j,1] = net_mov_lin[i,j,0] > lin_thres
            if (net_mov_lin[i,j,1] == True):
                max_v[i] = np.mean(Vels[i][j+1])
                lin_range[i] = inputs[j+1]
                lin_range_id[i] = j+1

    return net_mov_lin, max_v, lin_range, lin_range_id


def cal_skewness(search_num, inputs, valid_index_s, network_acvs, network_acvs_moving, theta_range):
    skewness = np.zeros((search_num,len(inputs),2))
    for i in valid_index_s:
        for inputi in range(len(inputs)):
            for j in range(2):
                if inputi == 0:
                    skewness[i,inputi,j] = cstat.skewness(theta_range, w=network_acvs[i][j,:,-1] - np.min(network_acvs[i][j,:,-1]))
                else:
                    skewness[i,inputi,j] = cstat.skewness(theta_range, w=network_acvs_moving[i,inputi][j,:,-1] - np.min(network_acvs_moving[i,inputi][j,:,-1]))
    skewness_mean = np.mean(skewness, axis=2)
    return skewness, skewness_mean


def cal_slope(search_num, inputs, valid_index_linear_move, Vels):
    slopes = np.zeros((search_num,2))
    for i in valid_index_linear_move:
        for j in range(2):
            reg = LinearRegression(fit_intercept=False).fit(inputs.reshape(-1, 1), Vels[i,:,j].reshape(-1, 1))
            slopes[i,j] = reg.coef_[0]
    slopes_mean = np.mean(slopes, axis=1)
    return slopes, slopes_mean


def cal_width_stable(search_num, valid_index_s, network_acvs):
    widths = np.zeros((search_num, 2))
    for neti in valid_index_s:
        for j in range(2):
            maxacv = np.max(network_acvs[neti][j,:,-1])
            minacv = np.min(network_acvs[neti][j,:,-1])
            widths[neti,j] = 100 - np.sum(network_acvs[neti][j,:,-1] < (maxacv + minacv)/2)
    widths_mean = np.mean(widths, axis=1)
    return widths, widths_mean


def cal_ampl_vel(search_num, valid_index_stable_move, max_amplitude, Vels, lin_range_id):
    ampl_vel_cor = np.zeros((search_num))
    for i in valid_index_stable_move:
        x = max_amplitude[i,:lin_range_id[i]+1,0] + max_amplitude[i,:lin_range_id[i]+1,1]
        y = Vels[i,:lin_range_id[i]+1,0]
        if len(x) > 1:
            ampl_vel_cor[i] = pearsonr(x, y)[0]
    return ampl_vel_cor


def cal_height_a_firate(search_num, inputs, valid_index_s, network_acvs_moving, b0, acvf=max0x, addb=1, subb=1):
    height = np.zeros((search_num, len(inputs),2))
    firate_max = np.zeros((search_num, len(inputs),2))
    for neti in valid_index_s:
        for i in range(len(inputs)):
            b = [b0 + addb * inputs[i], b0 - subb * inputs[i]]
            for j in range(2):
                height[neti,i,j] = np.max(network_acvs_moving[neti,i][j,:,-1]) + b0
                firate_max[neti,i,j] = np.max(acvf(network_acvs_moving[neti,i][j,:,-1] + b[j]))
    return height, firate_max


def cal_max_firate_3ring(search_num, inputs, valid_index_s, network_acvs_moving, bE, bI, acvf=max0x, addb=1, subb=1):
    firate_max = np.zeros((search_num, len(inputs), 3))
    bE = scalr2vec(bE, inputs)
    bI = scalr2vec(bI, inputs)
    for neti in valid_index_s:
        for i in range(len(inputs)):
            b = [bE[i], bI[i] + addb * inputs[i], bI[i] - subb * inputs[i]]
            for j in range(3):
                firate_max[neti,i,j] = np.max(acvf(network_acvs_moving[neti,i][j,:,-1] + b[j]))
    return firate_max


def cal_max_firate_3ring_asyb(search_num, inputs, valid_index_s, network_acvs_moving, bE, bl, br, acvf=max0x):
    firate_max = np.zeros((search_num, len(inputs), 3))
    bE = scalr2vec(bE, inputs)
    bl = scalr2vec(bl, inputs)
    br = scalr2vec(br, inputs)
    for neti in valid_index_s:
        for i in range(len(inputs)):
            b = [bE[i], bl[i], br[i]]
            for j in range(3):
                firate_max[neti,i,j] = np.max(acvf(network_acvs_moving[neti,i][j,:,-1] + b[j]))
    return firate_max


def mean_firate(search_num, inputs, valid_index_s, network_acvs_moving, b0, acvf=max0x, addb=1, subb=1):
    firate_mean = np.zeros((search_num, len(inputs),2))
    for neti in valid_index_s:
        for i in range(len(inputs)):
            b = [b0 + addb * inputs[i], b0 - subb * inputs[i]]
            for j in range(2):
                firate_mean[neti,i,j] = np.mean(acvf(network_acvs_moving[neti,i][j,:,-1] + b[j]))
    return firate_mean


def cal_mean_firate_3ring(search_num, inputs, valid_index_s, network_acvs_moving, bE, bI, acvf=max0x, addb=1, subb=1):
    firate_mean = np.zeros((search_num, len(inputs), 3))
    bE = scalr2vec(bE, inputs)
    bI = scalr2vec(bI, inputs)
    for neti in valid_index_s:
        for i in range(len(inputs)):
            b = [bE[i], bI[i] + addb * inputs[i], bI[i] - subb * inputs[i]]
            for j in range(3):
                firate_mean[neti,i,j] = np.mean(acvf(network_acvs_moving[neti,i][j,:,-1] + b[j]))
    return firate_mean


def cal_mean_firate_3ring_asyb(search_num, inputs, valid_index_s, network_acvs_moving, bE, bl, br, acvf=max0x):
    firate_mean = np.zeros((search_num, len(inputs), 3))
    bE = scalr2vec(bE, inputs)
    bl = scalr2vec(bl, inputs)
    br = scalr2vec(br, inputs)
    for neti in valid_index_s:
        for i in range(len(inputs)):
            b = [bE[i], bl[i], br[i]]
            for j in range(3):
                firate_mean[neti,i,j] = np.mean(acvf(network_acvs_moving[neti,i][j,:,-1] + b[j]))
    return firate_mean


def cal_FWHM_3ring(search_num, inputs, valid_index_s, network_acvs_moving, bE, bI, acvf=max0x, theta_num=100):
    widths = np.zeros((search_num, len(inputs), 3))
    bE = scalr2vec(bE, inputs)
    bI = scalr2vec(bI, inputs)
    for neti in valid_index_s:
        for i in range(len(inputs)):
            b = [bE[i], bI[i] + inputs[i], bI[i] - inputs[i]]
            for j in range(3):
                maxacv = np.max(acvf(network_acvs_moving[neti,i][j,:,-1] + b[j]))
                widths[neti,i,j] = (np.sum((acvf(network_acvs_moving[neti,i][j,:,-1] + b[j])) > (maxacv)/2)) / theta_num * 360
    return widths


def cal_FWHM_3ring_asyb(search_num, inputs, valid_index_s, network_acvs_moving, bE, bl, br, acvf=max0x, theta_num=100):
    widths = np.zeros((search_num, len(inputs), 3))
    bE = scalr2vec(bE, inputs)
    bl = scalr2vec(bl, inputs)
    br = scalr2vec(br, inputs)
    for neti in valid_index_s:
        for i in range(len(inputs)):
            b = [bE[i], bl[i], br[i]]
            for j in range(3):
                maxacv = np.max(acvf(network_acvs_moving[neti,i][j,:,-1] + b[j]))
                widths[neti,i,j] = (np.sum((acvf(network_acvs_moving[neti,i][j,:,-1] + b[j])) > (maxacv)/2)) / theta_num * 360
    return widths


def cal_u_from_f(netacv, w, theta_num, dtheta):
    netacv = netacv.copy()
    y = np.concatenate([netacv[0], netacv[1]])
    u = np.matmul(w,y)*dtheta/(2*np.pi)
    netacv[0] = u[:theta_num]
    netacv[1] = u[theta_num:]
    return netacv


def cal_f_from_u(netacv, w, theta_num, dtheta):
    netacv = netacv.copy()
    det = np.linalg.det(w)
    # Check if the matrix is invertible
    if det != 0:
        w_1 = np.linalg.inv(w)
        y = np.concatenate([netacv[0], netacv[1]])
        f = np.matmul(w_1,y)*dtheta/(2*np.pi)
        netacv[0] = f[:theta_num]
        netacv[1] = f[theta_num:]
        return netacv
    else:
         return (f"The matrix of net is not invertible")


def cal_correlation(inputs, vs):

    '''
    Calculate the correlations between inputs and velocity

    Parameters
    ----------
    inputs : 1darray
        The input (k omega) to the network
    vs : 2darray(input_num, ring_num)
        The angular velocity of the bump

    Returns
    -------
    rs: 2darray(ring_num, 2)
        Dimension 1: correlation for different rings.
        Dimension 2: (1) The correlation between inputs and velocity,
        (2) The correlation between inputs and velocity in the first half of the inputs.
    '''

    ring_num = vs.shape[1]
    
    # Find the index for the first half of inputs
    id_l = (np.abs(inputs - min(inputs)/2)).argmin()
    id_u = (np.abs(inputs - max(inputs)/2)).argmin() + 1

    rs = np.zeros((ring_num, 2))
    for i in range(ring_num):
        rs[i,0] = pearsonr(inputs, vs[:,i])[0]
        rs[i,1] = pearsonr(inputs[id_l:id_u], vs[id_l:id_u,i])[0]

    return rs


def cal_shape_mismatch(acvs, zeroid):

    '''
    Check the shape in moving is the same as the shape in stationary period for the one ring network.

    Parameters
    ----------
    acvs : 1darray(input length){2darray(theta_num, time)}
        The activity of the networks.
    
    '''
    input_len = len(acvs)
    theta_num = acvs[0].shape[0]
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta)

    if_same = np.ones((input_len, theta_num), dtype=bool)
    dev_ratios = np.zeros((input_len, theta_num))
    acv_stationary = acvs[zeroid][:,-1]
    tol = np.max(np.abs(np.diff(acv_stationary))) # the maximum difference between two adjacent points for the stationary bump

    # Extract values
    for i in range(0, input_len):
        membrane = acvs[i][:,-1]

        se = membrane - np.min(membrane)
        shiftv = np.rint(cstat.mean(theta_range, se, axis=0) / (2*np.pi) * theta_num).astype(int)
        membrane = np.roll(membrane, -shiftv)

        dev_ratios[i] = np.abs(membrane - acv_stationary) / tol
        # If there is a better shift that matches the stationary bump better
        shift_alt = [-1,1]
        dev = [1000,1000]
        for j in range(2):
            mem = membrane
            mem = np.roll(mem, shift_alt[j])
            dev_judge = np.abs(mem - acv_stationary) / tol
            while np.mean(dev_judge) < np.mean(dev[j]):
                dev[j] = dev_judge
                mem = np.roll(mem, shift_alt[j])
                dev_judge = np.abs(mem - acv_stationary) / tol

        dev_alt = [dev[0], dev[1], dev_ratios[i]]
        dev_alt_mean = np.mean(dev_alt, axis=1)
        index_min = np.argmin(dev_alt_mean)
        dev_ratios[i] = dev_alt[index_min]
        
        if_same[i] = dev_ratios[i] < 1

    ## Compare
    return if_same, dev_ratios


def cal_shape_mismatch_loop(network_acvs, index_check, zeroid):
    '''
    Parameters
    ----------
    network_acvs : 2darray(search_num, input_length){2darray(theta_num, time)}
    '''
    search_num = network_acvs.shape[0]
    input_num = network_acvs.shape[1]
    theta_num = network_acvs[index_check[0],0].shape[0]
    index_shape_mismatch = []

    dev_ratios = np.zeros((search_num, input_num, theta_num))
    if_match = np.zeros((search_num, input_num, theta_num), dtype=bool)
    for i in index_check:
        if_match[i], dev_ratios[i] = cal_shape_mismatch(network_acvs[i], zeroid)
        if not np.all(if_match[i]):
            index_shape_mismatch.append(i)

    return index_shape_mismatch, dev_ratios, if_match


def cal_lr_shape_match(acvs, zeroid):

    '''
    Check whether the shape of left (right) ring when v = v0 is mirror symmetric to the right (left) ring when v = -v0

    Parameters
    ----------
    acvs : 1darray(input length){3darray(ring, theta_num, time)}
        For 2ring, the order of dimension ring is [left, right]. For 3ring, the order is [central, left, right].
        Contains the activity of the networks.
    
    '''
    input_len = len(acvs)
    len_sum = input_len - 1
    theta_num = acvs[0].shape[1]
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta)
    ringi = acvs[0].shape[0] - 2 # 0 for 2ring, 1 for 3ring

    if_same = np.ones((input_len, theta_num), dtype=bool)
    dev_ratios = np.zeros((input_len, theta_num))

    # Extract values
    for i in range(0, input_len):

        if i == zeroid:
            if_same[i] = True
            continue

        sl = acvs[i][ringi,:,-1]
        sr = np.flip(acvs[len_sum-i][1+ringi,:,-1])

        sel = sl - np.min(sl)
        ser = sr - np.min(sr)
        
        shiftv = np.rint(cstat.mean(theta_range, sel-ser, axis=0) / (2*np.pi) * theta_num).astype(int)
        sl = np.roll(sl, -shiftv)

        adjc_diff = np.concatenate([np.diff(sl), [sl[0] - sl[-1]]])
        tol = np.max(np.abs(adjc_diff))
        dev_ratios[i] = np.abs(sl - sr) / tol
        # If there is a better shift that matches the stationary bump better
        shift_alt = [-1,1]
        dev = [1000,1000]
        for j in range(2):
            mem = sl
            mem = np.roll(mem, shift_alt[j])
            dev_judge = np.abs(mem - sr) / tol
            while np.mean(dev_judge) < np.mean(dev[j]):
                dev[j] = dev_judge
                mem = np.roll(mem, shift_alt[j])
                dev_judge = np.abs(mem - sr) / tol

        dev_alt = [dev[0], dev[1], dev_ratios[i]]
        dev_alt_mean = np.mean(dev_alt, axis=1)
        index_min = np.argmin(dev_alt_mean)
        dev_ratios[i] = dev_alt[index_min]
        
        if_same[i] = dev_ratios[i] < 1

    ## Compare
    return if_same, dev_ratios


def cal_lr_shape_match_loop(network_acvs, index_check, zeroid):
    '''
    Loops for checking whether the shape of left (right) ring when v = v0 is mirror symmetric to the right (left) ring when v = -v0
    
    Parameters
    ----------
    network_acvs : 2darray(search_num, input_length){3darray(ring, theta_num, time)}
    '''
    search_num = network_acvs.shape[0]
    input_num = network_acvs.shape[1]
    theta_num = network_acvs[index_check[0],0].shape[1]
    index_shape_mismatch = []

    dev_ratios = np.zeros((search_num, input_num, theta_num))
    if_match = np.zeros((search_num, input_num, theta_num), dtype=bool)
    for i in index_check:
        if_match[i], dev_ratios[i] = cal_lr_shape_match(network_acvs[i], zeroid)
        if not np.all(if_match[i]):
            index_shape_mismatch.append(i)

    return index_shape_mismatch, dev_ratios, if_match


def cal_lr_shape_same(acvs, zeroid):

    '''
    Check whether the shape of left and right rings are the same

    Parameters
    ----------
    acvs : 1darray(input length){3darray(ring, theta_num, time)}
        For 2ring, the order of dimension ring is [left, right]. For 3ring, the order is [central, left, right].
        Contains the activity of the networks.
    
    '''
    input_len = len(acvs)
    theta_num = acvs[0].shape[1]
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta)
    ringi = acvs[0].shape[0] - 2 # 0 for 2ring, 1 for 3ring

    if_same = np.ones((input_len, theta_num), dtype=bool)
    dev_ratios = np.zeros((input_len, theta_num))

    # Extract values
    for i in range(0, input_len):

        if i == zeroid:
            if_same[i] = True
            continue

        sl = acvs[i][ringi,:,-1]
        sr = acvs[i][1+ringi,:,-1]

        adjc_diff = np.concatenate([np.diff(sl), [sl[0] - sl[-1]]])
        tol = np.max(np.abs(adjc_diff))
        dev_ratios[i] = np.abs(sl - sr) / tol
        # If there is a better shift that matches the stationary bump better
        shift_alt = [-1,1]
        dev = [1000,1000]
        for j in range(2):
            mem = sl
            mem = np.roll(mem, shift_alt[j])
            dev_judge = np.abs(mem - sr) / tol
            while np.mean(dev_judge) < np.mean(dev[j]):
                dev[j] = dev_judge
                mem = np.roll(mem, shift_alt[j])
                dev_judge = np.abs(mem - sr) / tol

        dev_alt = [dev[0], dev[1], dev_ratios[i]]
        try:
            dev_alt_mean = np.mean(dev_alt, axis=1)
        except:
            plt.plot(sl, label='sl')
            plt.plot(sr, label='sr')
            plt.legend()
            plt.show()
            return if_same, dev_ratios
        index_min = np.argmin(dev_alt_mean)
        dev_ratios[i] = dev_alt[index_min]
        
        if_same[i] = dev_ratios[i] < 1

    ## Compare
    return if_same, dev_ratios


def cal_lr_shape_same_loop(network_acvs, index_check, zeroid):
    '''
    Loops for checking whether the shape of left (right) ring when v = v0 is mirror symmetric to the right (left) ring when v = -v0
    
    Parameters
    ----------
    network_acvs : 2darray(search_num, input_length){3darray(ring, theta_num, time)}
    '''
    search_num = network_acvs.shape[0]
    input_num = network_acvs.shape[1]
    theta_num = network_acvs[index_check[0],0].shape[1]
    index_shape_mismatch = []

    dev_ratios = np.zeros((search_num, input_num, theta_num))
    if_match = np.zeros((search_num, input_num, theta_num), dtype=bool)
    for i in index_check:
        if_match[i], dev_ratios[i] = cal_lr_shape_same(network_acvs[i], zeroid)
        if not np.all(if_match[i]):
            index_shape_mismatch.append(i)

    return index_shape_mismatch, dev_ratios, if_match


def cal_central_shape_match(acvs, zeroid):

    '''
    Check whether the shape of central when v = v0 is mirror symmetric to itself when v = -v0

    Parameters
    ----------
    acvs : 1darray(input length){3darray(ring, theta_num, time)}
        For 2ring, the order of dimension ring is [left, right]. For 3ring, the order is [central, left, right].
        Contains the activity of the networks.
    
    '''
    input_len = len(acvs)
    len_sum = input_len - 1
    theta_num = acvs[0].shape[1]
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta)

    if_same = np.ones((input_len, theta_num), dtype=bool)
    dev_ratios = np.zeros((input_len, theta_num))

    # Extract values
    for i in range(0, input_len//2):

        sl = acvs[i][0,:,-1]
        sr = np.flip(acvs[len_sum-i][0,:,-1])

        sel = sl - np.min(sl)
        ser = sr - np.min(sr)
        
        shiftv = np.rint(cstat.mean(theta_range, sel-ser, axis=0) / (2*np.pi) * theta_num).astype(int)
        sl = np.roll(sl, -shiftv)

        adjc_diff = np.concatenate([np.diff(sl), [sl[0] - sl[-1]]])
        tol = np.max(np.abs(adjc_diff))
        dev_ratios[i] = np.abs(sl - sr) / tol
        # If there is a better shift that matches the stationary bump better
        shift_alt = [-1,1]
        dev = [1000,1000]
        for j in range(2):
            mem = sl
            mem = np.roll(mem, shift_alt[j])
            dev_judge = np.abs(mem - sr) / tol
            while np.mean(dev_judge) < np.mean(dev[j]):
                dev[j] = dev_judge
                mem = np.roll(mem, shift_alt[j])
                dev_judge = np.abs(mem - sr) / tol

        dev_alt = [dev[0], dev[1], dev_ratios[i]]
        dev_alt_mean = np.mean(dev_alt, axis=1)
        index_min = np.argmin(dev_alt_mean)
        dev_ratios[i] = dev_alt[index_min]
        
        if_same[i] = dev_ratios[i] < 1

    ## Compare
    return if_same, dev_ratios


def cal_central_shape_match_loop(network_acvs, index_check, zeroid, ring_num=3):
    '''
    Loops for checking whether the shape of left (right) ring when v = v0 is mirror symmetric to the right (left) ring when v = -v0
    
    Parameters
    ----------
    network_acvs : 2darray(search_num, input_length){3darray(ring, theta_num, time)}
    '''
    search_num = network_acvs.shape[0]
    input_num = network_acvs.shape[1]
    theta_num = network_acvs[index_check[0],0].shape[1] if ring_num == 3 else network_acvs[index_check[0],0].shape[0]
    index_shape_mismatch = []

    dev_ratios = np.zeros((search_num, input_num, theta_num))
    if_match = np.zeros((search_num, input_num, theta_num), dtype=bool)
    for i in index_check:
        network_acv_temp = network_acvs[i].copy()
        if ring_num == 1:
            for j in range(input_num):
                network_acv_temp[j] = network_acvs[i,j].reshape(1,theta_num,-1)
        
        if_match[i], dev_ratios[i] = cal_central_shape_match(network_acv_temp, zeroid)
        if not np.all(if_match[i]):
            index_shape_mismatch.append(i)

    return index_shape_mismatch, dev_ratios, if_match


def cal_height_dif(acvs, actfun, zeroid, tol_rptp=None, b=0):

    '''
    Check the shape in moving is the same as the shape in stationary period for the one ring case.

    Parameters
    ----------
    acvs : 1darray(input length){2darray(theta_num, time)}
        The activity of the networks.
    
    '''

    input_len = len(acvs)

    if_same = np.ones((input_len), dtype=bool)
    dev_ratios = np.zeros((input_len))
    f0 = actfun(acvs[zeroid][:,-1] + b)
    height0 = max(f0)

    if tol_rptp is None:
        index_max = np.argmax(f0)
        diffs = np.diff(f0)
        diff_peak_max = max(np.abs(diffs[[index_max-1, index_max]]))
        tol_rptp = (diff_peak_max * 0.5) / height0
        # the half maximum difference between two adjacent points for the stationary bump: 
        # Assume the peak points are two neighbor points to the actual interpolated peak, 
        # and the interpolation slope is the maximum slope from peak to two peak's neighbor points

    # Extract values
    for i in range(input_len):
        height = max(actfun(acvs[i][:,-1]+b))

        dev_ratios[i] = np.abs(height - height0) / height0
        if_same[i] = dev_ratios[i] < tol_rptp + 1e-2

    ## Compare
    # if np.isnan(dev_ratios[0]):
    #     print(height0, tol_rptp, index_max, diffs, diff_peak_max, tol_rptp)
    return if_same, dev_ratios


def cal_height_dif_loop(network_acvs, index_check, actfun, zeroid, tol_rptp=None, b=0):
    '''
    For the one ring case

    Parameters
    ----------
    network_acvs : 2darray(search_num, input_length){2darray(theta_num, time)}
    '''
    search_num = network_acvs.shape[0]
    input_num = network_acvs.shape[1]
    index_height_mismatch = []

    dev_ratios = np.zeros((search_num, input_num))
    if_match = np.zeros((search_num, input_num), dtype=bool)
    for i in index_check:
        if_match[i], dev_ratios[i] = cal_height_dif(network_acvs[i], actfun, zeroid, tol_rptp, b)
        if not np.all(if_match[i]):
            index_height_mismatch.append(i)

    return index_height_mismatch, dev_ratios, if_match


def cal_firate_a_acv_mean_a_peak(network_acvs_moving, inputs, index, b0, actfun, kind='normal', weights=[None]*3):

    '''
    Calculate the mean and the peak membrane potential and firing rate for the network,
    also the difference between left & right rings
    '''

    ring_num = network_acvs_moving[index[0],0].shape[0]
    rladd = ring_num - 2
    search_num = network_acvs_moving.shape[0]

    total_inputs = cal_total_inputs(b0, inputs, ring_num, kind)

    peak_acv = np.zeros((search_num, len(inputs), ring_num))
    mean_acv = np.zeros((search_num, len(inputs), ring_num))
    peak_firate = np.zeros((search_num, len(inputs), ring_num))
    mean_firate = np.zeros((search_num, len(inputs), ring_num))

    for neti in index:
        for i in range(len(inputs)):
            for j in range(ring_num):
                acv = network_acvs_moving[neti,i][j,:,-1]
                input_now = total_inputs[j][i]
                peak_acv[neti,i,j] = max(acv + input_now)
                mean_acv[neti,i,j] = np.average(acv + input_now, weights=weights[j])
                peak_firate[neti,i,j] = max(actfun(acv + input_now))
                mean_firate[neti,i,j] = np.average(actfun(acv + input_now), weights=weights[j])
    
    peak_acv_diff = peak_acv[:,:,1+rladd] - peak_acv[:,:,0+rladd]
    mean_acv_diff = mean_acv[:,:,1+rladd] - mean_acv[:,:,0+rladd]
    peak_firate_diff = peak_firate[:,:,1+rladd] - peak_firate[:,:,0+rladd]
    mean_firate_diff = mean_firate[:,:,1+rladd] - mean_firate[:,:,0+rladd]

    return peak_acv, mean_acv, peak_firate, mean_firate, peak_acv_diff, mean_acv_diff, peak_firate_diff, mean_firate_diff


def cal_input_diff_cor(inputs, diffs, index, linear_range_id=None):
    search_num = diffs[0].shape[0]
    n_diffs = len(diffs)
    cor = np.zeros((search_num, n_diffs))
    ps = np.zeros((search_num, n_diffs))

    if linear_range_id is None:
        for i in index:
            for j in range(n_diffs):
                res = pearsonr(inputs, diffs[j][i])
                cor[i,j] = res.statistic
                ps[i,j] = res.pvalue

    else:
        for i in index:
            range_id = np.arange(linear_range_id[i][0], linear_range_id[i][1]+1).astype(int)
            for j in range(n_diffs):
                res = pearsonr(inputs[range_id], diffs[j][i][range_id])
                cor[i,j] = res.statistic
                ps[i,j] = res.pvalue

    return cor, ps


def cal_total_inputs_2r(b0, inputs):

    left_inputs = [b0-b0*inputs[i] for i in range(len(inputs))]
    right_inputs = [b0+b0*inputs[i] for i in range(len(inputs))]
    total_inputs = [left_inputs, right_inputs]

    return total_inputs


def cal_total_inputs_2r_increaseb(b0, inputs):

    left_inputs = [b0 * (1 - inputs[i] + 0.5 * abs(inputs[i])) for i in range(len(inputs))]  
    right_inputs = [b0 * (1 + inputs[i] + 0.5 * abs(inputs[i])) for i in range(len(inputs))]
    total_inputs = [left_inputs, right_inputs]

    return total_inputs


def cal_total_inputs_3r(b0, inputs):

    bE = b0[0]
    bI = b0[1]

    central_inputs = [bE] * len(inputs)
    left_inputs = [bI - inputs[i] for i in range(len(inputs))]
    right_inputs = [bI + inputs[i] for i in range(len(inputs))]
    total_inputs = [central_inputs, left_inputs, right_inputs]

    return total_inputs


def cal_total_inputs_3r_increaseb(b0, inputs):

    bE = b0[0]
    bI = b0[1]

    central_inputs = [bE * (1 + abs(inputs[i])) for i in range(len(inputs))]
    left_inputs = [bI - inputs[i] for i in range(len(inputs))]
    right_inputs = [bI + inputs[i] for i in range(len(inputs))]
    total_inputs = [central_inputs, left_inputs, right_inputs]

    return total_inputs


def cal_total_inputs_drosophila(b0, inputs):

    bE = b0[0]
    bI = b0[1]

    central_inputs = [0 for i in range(len(inputs))]
    left_inputs = [bI - inputs[i] + bI * (1 + abs(inputs[i])) for i in range(len(inputs))]
    right_inputs = [bI + inputs[i] + bI * (1 + abs(inputs[i])) for i in range(len(inputs))]
    total_inputs = [central_inputs, left_inputs, right_inputs]

    return total_inputs


def cal_total_inputs_3r_decreaseb(b0, inputs):

    bE = b0[0]
    bI = b0[1]

    central_inputs = [bE * (1 - 0.1 * abs(inputs[i])) for i in range(len(inputs))]
    left_inputs = [bI - inputs[i] for i in range(len(inputs))]
    right_inputs = [bI + inputs[i] for i in range(len(inputs))]
    total_inputs = [central_inputs, left_inputs, right_inputs]

    return total_inputs


def cal_total_inputs_zebrafish(b0, inputs):

    bc = b0[0]
    b0 = b0[1]
    input_num = len(inputs)
    
    absv_b = [0.3 * np.sign(np.abs(ratio)) + 0.1 * np.abs(ratio) for ratio in inputs]
    deltab = [0.1 * ratio * b0 for ratio in inputs]

    central_inputs = [bc + absv_b[i] for i in range(input_num)]
    left_inputs = [b0 - deltab[i] + absv_b[i] for i in range(input_num)]
    right_inputs = [b0 + deltab[i] + absv_b[i] for i in range(input_num)]
    total_inputs = [central_inputs, left_inputs, right_inputs]

    return total_inputs


def cal_total_inputs(b0, inputs, ring_num, kind='normal'):
    if (ring_num == 2) & (kind == 'normal'):
        return cal_total_inputs_2r(b0, inputs)
    elif (ring_num == 3) & (kind == 'normal'):
        return cal_total_inputs_3r(b0, inputs)
    elif (ring_num == 2) & (kind == 'increaseb0'):
        return cal_total_inputs_2r_increaseb(b0, inputs)
    elif (ring_num == 3) & (kind == 'increaseb0'):
        return cal_total_inputs_3r_increaseb(b0, inputs)
    elif (ring_num == 3) & (kind == 'decreaseb0'):
        return cal_total_inputs_3r_decreaseb(b0, inputs)
    elif (ring_num == 3) & (kind == 'drosophila'):
        return cal_total_inputs_drosophila(b0, inputs)
    elif (ring_num == 3) & (kind == 'zebrafish'):
        return cal_total_inputs_zebrafish(b0, inputs)
    

def cal_acv_rl_diff(netacvs, index_check):
    search_num = netacvs.shape[0]
    input_num = netacvs.shape[1]
    ring_num = netacvs[index_check[0], 0].shape[0]
    theta_num = netacvs[index_check[0], 0].shape[1]
    rladd = ring_num - 2
    acv_rl_diff = np.zeros((search_num, input_num, theta_num))
    for neti in index_check:
        for inputi in range(input_num):
            acv = netacvs[neti, inputi]
            acv_rl_diff[neti, inputi] = acv[1+rladd,:,-1] - acv[rladd,:,-1]
    
    return acv_rl_diff


def cal_peak_id(acv, theta_range):
    theta_num = len(theta_range)
    se = acv - min(acv)
    peakid = np.rint(cstat.mean(theta_range, se, axis=0) / (2*np.pi) * theta_num + theta_num // 2 - 0.5).astype(int)
    return peakid


def cal_peak_loc(acv, theta_range):
    theta_num = len(theta_range)
    se = acv - min(acv)
    zeroN = np.sum(se > 0)
    if zeroN == 1:
        peakid = np.argmax(se)
    else:
        peakid = cstat.mean(theta_range, se, axis=0) / (2*np.pi) * theta_num + theta_num // 2 - 0.5
    # because, for neuron 0, it stands for [-pi, -pi+dtheta] -> [-24.5-0.5,-24.5+0.5]. If neuron 0 sit at x=0, then it covers [0-0.5, 0+0.5]
    return peakid


def cal_peak_loc_auto(y, theta_range):
    # y: 2d array (cell_num, time_points)
    theta_num = len(theta_range)
    cell_num = y.shape[0]
    time_num = y.shape[1]
    ring_num = cell_num // theta_num
    theta_range_2d = theta_range.reshape(-1, 1)
    theta_range_2d = np.repeat(theta_range_2d, time_num, axis=1)

    if ring_num == 1:
        y = y.reshape(1, cell_num, -1)
    elif ring_num == 2:
        y = np.stack([y[:theta_num], y[theta_num:]], axis=0)
    elif ring_num == 3:
        y = np.stack([y[:theta_num], y[theta_num:2*theta_num], y[2*theta_num:]], axis=0)
    
    peak_loc = np.zeros((ring_num, time_num))
    for r in range(ring_num):
        se = y[r] - np.min(y[r])
        peak_loc[r] = cstat.mean(theta_range_2d, se, axis=0)

    return peak_loc


def cal_linear_range(network_eval_moving, Vels, inputs, valid_index_s, thres=0.99):

    '''
    Intend to cal the range of inputs that the cor between vel and input is > thres
    intend for 2 ring and 3 ring cases
    
    Returns:
    stable_mov_range: 2darray(search_num, 2)
        The range of inputs that the network is stable moving
        2nd dimension: [min, max]
    stable_mov_range_id: 2darray(search_num, 2)
        The index of inputs that the network is stable moving
        2nd dimension: [min_id, max_id]
    linear_mov_range: 2darray(search_num, 2)
        The range of inputs that the network is linear moving
        2nd dimension: [min, max]
    linear_mov_range_id: 2darray(search_num, 2)
        The index of inputs that the network is linear moving
        2nd dimension: [min_id, max_id]
    '''

    input_num = network_eval_moving.shape[1]
    search_num = network_eval_moving.shape[0]

    stable_mov_range = np.zeros((search_num, 2))
    stable_mov_range[:] = np.nan
    stable_mov_range[valid_index_s] = 0

    stable_mov_range_id = np.copy(stable_mov_range)
    linear_mov_range = np.copy(stable_mov_range)
    linear_mov_range_id = np.copy(stable_mov_range)

    for i in valid_index_s:
        for j in range(0, input_num//2):
            check_range = np.arange(j, input_num-j)
            stable_range_get = False
            if np.all(network_eval_moving[i, check_range] == 'stable moving'):
                if not stable_range_get:
                    stable_mov_range_id[i] = [j, input_num-j-1]
                    stable_mov_range[i] = [inputs[j], inputs[input_num-j-1]]
                    stable_range_get = True
                cor = np.abs(pearsonr( inputs[j:input_num-j], np.mean(Vels[i,j:input_num-j], axis=1) )[0])
                if cor > thres:
                    linear_mov_range_id[i] = [j, input_num-j-1]
                    linear_mov_range[i] = [inputs[j], inputs[input_num-j-1]]
                    break
                
    return stable_mov_range, stable_mov_range_id, linear_mov_range, linear_mov_range_id


def cal_linear_range_ascend(network_eval_moving, Vels, inputs, valid_index_s, thres=0.99):

    '''
    Intend to cal the range of inputs that the cor between vel and input is > thres
    intend for 2 ring and 3 ring cases
    '''

    input_num = network_eval_moving.shape[1]
    search_num = network_eval_moving.shape[0]

    linear_mov_range = np.zeros((search_num, 2))
    linear_mov_range[:] = np.nan
    linear_mov_range[valid_index_s] = 0

    linear_mov_range_id = np.copy(linear_mov_range).astype(int)

    for i in valid_index_s:
        test_range = np.flip(np.arange(0, input_num//2))
        curt_linear = True
        for j in test_range:
            check_range = np.arange(j, input_num-j)
            curt_stable = np.all(network_eval_moving[i, check_range] == 'stable moving')
            if curt_stable:
                cor = np.abs(pearsonr( inputs[j:input_num-j], np.mean(Vels[i,j:input_num-j], axis=1) )[0])
                if cor < thres:
                    curt_linear = False
            if (not curt_stable) or (not curt_linear):
                linear_mov_range_id[i] = [j+1, input_num-j-2]
                linear_mov_range[i] = [inputs[j+1], inputs[input_num-j-2]]
                break
            elif j == test_range[-1]:
                linear_mov_range_id[i] = [j, input_num-j-1]
                linear_mov_range[i] = [inputs[j], inputs[input_num-j-1]]
                
    return linear_mov_range, linear_mov_range_id


def cal_cstat_and_print_65(index_cal, cor, ratio, printtitle='', print_ratio=True):

    '''
    0: number of positive correlation, 1: proportion of positive correlation
    1: mean of positive part, 2: sd of positve part
    3. mean of negative part, 4. sd of negative part
    '''
    netnum = len(index_cal)
    division = len(cor[index_cal].flatten()) // netnum

    index_positive = cor[index_cal] > 0
    index_negative = cor[index_cal] <= 0
    cor_stat = np.zeros((7))
    cor_stat[:] = np.nan
    ratio_stat = np.zeros((5))
    ratio_stat[:] = np.nan

    cor_stat[0] = np.sum(cor[index_cal] > 0) // division
    cor_stat[1] = cor_stat[0] / netnum

    if cor_stat[1] > 0:
        cor_stat[2] = cor[index_cal][index_positive].mean()
        cor_stat[3] = cor[index_cal][index_positive].std()
        cor_stat[4] = cor[index_cal][index_positive].max()
        
        ratio_stat[0] = ratio[index_cal][index_positive].mean()
        ratio_stat[1] = ratio[index_cal][index_positive].std()
        ratio_stat[2] = ratio[index_cal][index_positive].max()
    if cor_stat[1] < 1:
        cor_stat[5] = cor[index_cal][index_negative].mean()
        cor_stat[6] = cor[index_cal][index_negative].std()
        
        ratio_stat[3] = ratio[index_cal][index_negative].mean()
        ratio_stat[4] = ratio[index_cal][index_negative].std()

    print(f'{printtitle} Slope: +% = {cor_stat[0]:3.0f} ({cor_stat[1] * 100:3.1f}%), +M = {cor_stat[2]:1.3f}, +SD = {cor_stat[3]:1.3f} +MAX = {cor_stat[4]:1.3f}\
, -M = {cor_stat[5]:.3f}, -SD = {cor_stat[6]:.3f}')
    if print_ratio:
        print(f'{printtitle} Ratio:                   +M = {ratio_stat[0]:1.3f}, +SD = {ratio_stat[1]:1.3f} +MAX = {ratio_stat[2]:1.3f}\
, -M = {ratio_stat[3]:.3f}, -SD = {ratio_stat[4]:.3f}')
    else:
        print('    ')

    return cor_stat, ratio_stat


def cal_rl_and_rl_inputs_cor_loop(acvs, index_c, inputs):
    '''
    Calculate the phase difference between the right and left rings, 
    and the correlation between the phase difference and the inputs.

    acvs 2d array (search_num, input_num) [3d array (ring_num, theta_num, time_num) ]
    '''
    search_num = acvs.shape[0]
    input_num = acvs.shape[1]
    rl_difs = np.zeros((search_num, input_num//2+1))
    rl_difs[:] = np.nan
    cors = np.zeros((search_num))
    cors[:] = np.nan
    
    for i in index_c:
        for j in range(input_num//2+1):
            rl_difs[i,j] = cal_lr_vecmean_dif(acvs[i,j]) # unit: neuron
        if np.any(np.abs(rl_difs[i]) > 0.05):
            cors[i] = pearsonr(rl_difs[i], inputs[:input_num//2+1]).statistic
        else:
            cors[i] = 0

    return rl_difs, cors


def cal_lr_vecmean_dif(acv):
    '''
    acv: 3d array (ring_num, theta_num, time_num) 
        1st dim: l, r / c, l, r

    return: unit: neuron
    '''
    ring_num = acv.shape[0]
    ring_add = ring_num - 2

    theta_num = acv.shape[1]
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta)
    factor = 2 * np.pi / theta_num
    rl_dif = cstat.cdiff(cal_peak_loc(acv[1+ring_add,:,-1], theta_range)*factor, cal_peak_loc(acv[ring_add,:,-1], theta_range)*factor)/factor
    return rl_dif


def cal_phase_dif_and_inputs_cor_loop(acvs, index_c, inputs, vels, actfun=None, bs=None, usef=False, alsomaxloc=False):
    '''
    Calculate the phase difference between the right - left, central - (left + right) / 2 (if 3 rings), 
    and the correlation, slope between the phase difference and the velocity.
    
    acvs 2d array (search_num, input_num) [3d array (ring_num, theta_num, time_num) ]
    '''
    vel = vels.mean(axis=2)
    search_num = acvs.shape[0]
    input_num = acvs.shape[1]
    ring_num = acvs[index_c[0],0].shape[0]
    campar_num = ring_num - 1


    phase_difs = np.zeros((search_num, input_num, campar_num))
    phase_difs[:] = np.nan
    cors = np.zeros((search_num, campar_num))
    cors[:] = np.nan
    slopes = np.zeros((search_num, campar_num))
    slopes[:] = np.nan
    '''
    I made slopes with np.any(np.abs(phase_difs[i,:,k]) < 0.2) as nan, because
    although the small phase difference could due to small vel, and network could
    achieve high vel if given big input, but I didn't check it. So it's better to
    use nan, as if to ignore it
    
    '''
    
    for i in index_c:
        for j in range(input_num):
            phase_difs[i,j] = cal_phase_dif(acvs[i,j], vels[i,j], ring_num, bs, inputs, j, actfun, usef) # unit: neuron
        for k in range(campar_num):
            if np.any(np.abs(phase_difs[i,:,k]) > 0.2) & (not np.any(np.isnan(phase_difs[i,:,k]))):
                # cors[i,k] = np.abs(pearsonr(phase_difs[i,input_num//2:,k], inputs[input_num//2:]).statistic)
                # slopes[i,k] = linregress(inputs[input_num//2:], phase_difs[i,input_num//2:,k]).slope
                cors[i,k] = np.abs(pearsonr(phase_difs[i,input_num//2:,k], vel[i,input_num//2:]).statistic)
                slopes[i,k] = linregress(np.abs(vel[i,input_num//2:]), phase_difs[i,input_num//2:,k]).slope

    if alsomaxloc:
        maxloc_difs = np.zeros((search_num, input_num, campar_num))
        maxloc_difs[:] = np.nan
        cors_maxloc = np.zeros((search_num, campar_num))
        cors_maxloc[:] = np.nan
        slopes_maxloc = np.zeros((search_num, campar_num))
        slopes_maxloc[:] = np.nan

        for i in index_c:
            for j in range(input_num):
                maxloc_difs[i,j] = cal_phase_dif(acvs[i,j], vels[i,j], ring_num, bs, inputs, j, actfun, usef, alsomaxloc) # unit: neuron
            for k in range(campar_num):
                if np.any(np.abs(maxloc_difs[i,:,k]) > 0.2):
                    cors_maxloc[i,k] = np.abs(pearsonr(maxloc_difs[i,input_num//2:,k], inputs[input_num//2:]).statistic)
                    slopes_maxloc[i,k] = linregress(inputs[input_num//2:], maxloc_difs[i,input_num//2:,k]).slope
                else:
                    slopes_maxloc[i,k] = 0

        return np.squeeze(phase_difs), np.squeeze(cors), np.squeeze(slopes), np.squeeze(maxloc_difs), np.squeeze(cors_maxloc), np.squeeze(slopes_maxloc)

    return np.squeeze(phase_difs), np.squeeze(cors), np.squeeze(slopes)


def cal_phase_dif(acv, vel, ring_num, bs0, inputs, j, f, usef=False, maxloc=False):
    '''
    acv: 3d array (ring_num, theta_num, time_num) 
        1st dim: l, r / c, l, r

    return: unit: neuron
    '''
    acv = acv.copy()
    ring_add = ring_num - 2
    move_dir = np.sign(vel.mean())
    phase_difs = np.zeros((ring_num-1))

    theta_num = acv.shape[1]
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta)
    
    factor = 2 * np.pi / theta_num
    if usef:
        for i in range(ring_num):
            bs = cal_total_inputs(bs0, inputs, ring_num)
            acv[i,:,-1] = f(acv[i,:,-1]+bs[i][j])

    if not maxloc:
        # Right - Left
        peak1 = cal_peak_loc(acv[ring_add+1,:,-1], theta_range) * factor
        peak2 = cal_peak_loc(acv[ring_add,:,-1], theta_range) * factor
        phase_difs[0] = cstat.cdiff(peak1, peak2) / factor * move_dir

        # Center - (l + r) / 2
        if ring_num == 3:
            peak1 = cal_peak_loc(acv[0,:,-1], theta_range) * factor
            if np.any(acv[1,:,-1] > 0) & np.any(acv[2,:,-1] > 0):
                peak2 = ( cal_peak_loc(acv[1,:,-1], theta_range) + cal_peak_loc(acv[2,:,-1], theta_range) ) * factor / 2
            elif np.any(acv[1,:,-1] > 0):
                peak2 = cal_peak_loc(acv[1,:,-1], theta_range) * factor
            elif np.any(acv[2,:,-1] > 0):
                peak2 = cal_peak_loc(acv[2,:,-1], theta_range) * factor
            phase_difs[1] = cstat.cdiff(peak1, peak2) / factor * move_dir
            # if np.any(np.isnan(phase_difs[1])):
            #     print(vel)
            #     print(peak1, peak2, phase_difs[1])
            #     print(j)
            #     plt.plot(acv[0,:,-1], label='c')
            #     plt.plot(acv[1,:,-1], label='l')
            #     plt.plot(acv[2,:,-1], label='r')
            #     plt.legend()
            #     plt.show()
    else:
        # Right - Left
        peak1 = np.argmax(acv[ring_add+1,:,-1]) * factor
        peak2 = np.argmax(acv[ring_add,:,-1]) * factor
        phase_difs[0] = cstat.cdiff(peak1, peak2) / factor * move_dir

        # Center - (l + r) / 2
        if ring_num == 3:
            peak1 = np.argmax(acv[0,:,-1]) * factor
            peak2 = ( np.argmax(acv[1,:,-1]) + np.argmax(acv[2,:,-1]) ) * factor / 2
            phase_difs[1] = cstat.cdiff(peak1, peak2) / factor * move_dir

    return phase_difs # r - l, c - (l + r) / 2


def vel_phaselag_slope_cor_loop(Vels, slopes, inputs, index_vel_slope):
    '''
    Calculate the correlations between the slopes that using inputs to predict vels, 
    and the slopes that using inputs to predict the phase lag between rings.

    Vels: 3d array (search_num, input_num, ring_num)
    slopes: 3d array (search_num, phase_comparison_num)
    '''
    compar_num = slopes.shape[1]
    index_cs = np.zeros((compar_num), dtype=object)
    for i in range(compar_num):
        index_cs[i] = np.where((slopes[:,i] != 0) & (~np.isnan(slopes[:,i])))[0]

    search_num = Vels.shape[0]
    vel = np.mean(Vels, axis=2)
    vel_slopes = np.zeros((search_num))
    vel_slopes[:] = np.nan
    cors = np.zeros((compar_num))

    for i in index_vel_slope:
        vel_slopes[i] = np.abs(linregress(inputs, vel[i]).slope)

    for i in range(compar_num):
        if len(index_cs[i]) > 1:
            cors[i] = np.abs(pearsonr(slopes[index_cs[i],i], vel_slopes[index_cs[i]]).statistic)
    
    return vel_slopes, cors


def cal_vel_slope(Vels, inputs, index_vel_slope):
    '''
    Calculate the correlations between the slopes that using inputs to predict vels, 
    and the slopes that using inputs to predict the phase lag between rings.

    Vels: 3d array (search_num, input_num, ring_num)
    slopes: 3d array (search_num, phase_comparison_num)
    '''
    search_num = Vels.shape[0]
    vel = np.mean(Vels, axis=2)
    vel_slopes = np.zeros((search_num))
    vel_slopes[:] = np.nan

    for i in index_vel_slope:
        vel_slopes[i] = linregress(inputs, vel[i]).slope
    
    return vel_slopes


def cal_vel_slop_stat(Vels, valid_index_linear_move, inputs):
    vel_slopes = cal_vel_slope(Vels, inputs, valid_index_linear_move)
    index_neg_slope = np.where(vel_slopes < 0)[0]
    num_neg_slope = len(index_neg_slope)
    index_pos_slope = np.where(vel_slopes > 0)[0]
    num_pos_slope = len(index_pos_slope)

    meann, stdn, maxvn, minvn = cal_statistics(vel_slopes[index_neg_slope])
    if num_pos_slope > 0:
        meanp, stdp, maxvp, minvp = cal_statistics(vel_slopes[index_pos_slope])
    else:
        meanp, stdp, maxvp, minvp = [np.nan]*4

    return num_neg_slope, num_pos_slope, meann, stdn, maxvn, minvn, meanp, stdp, maxvp, minvp


def cal_statistics(var):
    mean = np.nanmean(var)
    std = np.nanstd(var)
    maxv = np.nanmax(var)
    minv = np.nanmin(var)
    return mean, std, maxv, minv


def cal_skewness_cor_slope_loop(acvs, index_c, vels, actfun=None, bs=None, mode='pewsey', Doabs=True):
    '''
    Calculate the skewness and its correlation and slope when using vel to predict.
    ring num can be 2 or 3
    
    acvs 2d array (search_num, input_num) [3d array (ring_num, theta_num, time_num) ]
    '''
    vel = vels.mean(axis=2)
    search_num = acvs.shape[0]
    input_num = acvs.shape[1]
    ring_num = acvs[index_c[0],0].shape[0]
    record_num = ring_num - 1
    
    theta_num = acvs[index_c[0],0].shape[1]
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta)

    skewness = np.zeros((search_num, input_num, record_num))
    skewness[:] = np.nan
    cors = np.zeros((search_num, record_num))
    cors[:] = np.nan
    slopes = np.zeros((search_num, record_num))
    slopes[:] = np.nan

    rids = [-1,0] # 2ring, 1 record_num: right; 3 ring: right, central
    
    for i in index_c:
        for ri in range(record_num):
            r = rids[ri]
            for j in range(input_num):
                if actfun is not None:
                    fir = actfun(acvs[i,j][r,:,-1] + bs[r])
                    skewness[i,j,ri] = cstat.skewness(theta_range, w=fir, mode=mode)
                else:
                    skewness[i,j,ri] = cstat.skewness(theta_range, w=acvs[i,j][r,:,-1] - np.min(acvs[i,j][r,:,-1]), mode=mode)

            if np.any(np.abs(skewness[i,:,ri]) > 0.01) & (not np.any(np.isnan(skewness[i,:,ri]))):
                cors[i,ri] = np.abs(pearsonr(skewness[i,:,ri], vel[i]).statistic)
                slopes[i,ri] = linregress(vel[i], skewness[i,:,ri]).slope

    if Doabs:
        slopes = np.abs(slopes)
    return np.squeeze(skewness), np.squeeze(cors), np.squeeze(slopes)


def find_parspace_center_id(network_pars, linear_mov_id):
    net_pars = network_pars[linear_mov_id]
    par_num = network_pars.shape[1]
    selected_par = np.zeros(par_num)
    for i in range(par_num):
        par_count = np.unique(net_pars[:,i], return_counts=True)
        mean_weight = np.average(par_count[0], weights=par_count[1])
        nearest_id = np.argmin(np.abs(par_count[0] - mean_weight))
        selected_par[i] = par_count[0][nearest_id]
    
    net_id = np.where((network_pars == selected_par).all(axis=1))[0][0]
    return net_id, selected_par


def cal_sim_t_angVel(traj_sample, traj_type, slope, factor=1):
    # t in ms -> int values
    t = (traj_sample.t.values * 1000).astype(int)*factor
    angVel = traj_sample.angVel.values

    if traj_type == 'fly':
        t_spans = np.stack((t[:-1], t[1:]), axis=1)
        ratiovs = angVel[1:] / slope
        t_evals = t_spans.copy()
    elif traj_type == 'fish':
        t_spans = []
        ratiovs = []
        t_evals = []
        bout_ids = np.where(angVel != 0)[0]
        if bout_ids[0] > 1:
            t_span = (t[0], t[bout_ids[0]-1])
            t_spans.append(t_span)
            ratiovs.append(0)
            t_evals.append(np.linspace(t_span[0], t_span[1], bout_ids[0]))
        for i, bout_id in enumerate(bout_ids):
            t_span = (t[bout_id-1], t[bout_id])
            t_spans.append(t_span)
            ratiovs.append(angVel[bout_id] / slope)
            t_evals.append(t_span)
            if bout_id + 1 < len(angVel):
                if angVel[bout_id + 1] == 0:
                    next_id = bout_ids[i+1] - 1 if i+1 < len(bout_ids) else len(t) - 1
                    t_end = t[next_id]
                    t_span = (t[bout_id], t_end)
                    t_spans.append(t_span)
                    ratiovs.append(0)
                    t_evals.append(np.linspace(t_span[0], t_span[1], next_id - bout_id + 1))
                
        t_spans = np.array(t_spans)
        ratiovs = np.array(ratiovs)
        
    return t_spans, ratiovs, t_evals


def cal_print_increase_u_f(index_cal, network_acvs_moving, inputs, zeroid, actfun, b0):
    '''
    network_acvs_moving: 2d array (search_num, input_num) [3d array (ring_num, theta_num, time_num) ]or [3d array (theta_num, time_num) ]
    '''
    
    corr_stat_list = []
    ratio_stat_list = []
    search_num = network_acvs_moving.shape[0]
    acv = np.zeros((search_num, len(inputs), 4)) # 3rd dim: peaku, meanu, peakf, meanf
    slope = np.zeros((search_num, 4))
    ratio = np.zeros((search_num, 4))

    for i in index_cal:
        for j in range(len(inputs)):
            # u = network_acvs_moving[i,j][:,-1] + b0 if ring_num == 1 else np.mean(network_acvs_moving[i,j][:,:,-1], axis=0) + b0 * (1 + 0.5 * abs(inputs[j])) # ring average
            u = network_acvs_moving[i,j][:,-1] + b0
            f = actfun(u)
            acv[i,j,0] = max(u)
            acv[i,j,1] = np.mean(u)
            acv[i,j,2] = max(f)
            acv[i,j,3] = np.mean(f)

        for j in range(4):
            res = linregress(inputs[zeroid:], acv[i,zeroid:,j])
            slope[i,j] = res.slope
            ratio[i,j] = res.slope / res.intercept

    print(f'Net num: {len(index_cal)}')
    printtitle = ['peak u', 'mean u', 'peak f', 'mean f']
    for i in range(4):
        print_ratio = True if i >= 2 else False
        corr_stat, ratio_stat = cal_cstat_and_print_65(index_cal, slope[:,i], ratio[:,i], printtitle=printtitle[i], print_ratio=print_ratio)
        corr_stat_list.append(corr_stat)
        ratio_stat_list.append(ratio_stat)
    print('All min except mean u')
    corr_stat, ratio_stat = cal_cstat_and_print_65(index_cal, slope[:,[0,2,3]].min(axis=1), ratio[:,[0,2,3]].min(axis=1), printtitle=f'      ', print_ratio=True)
    corr_stat_list.append(corr_stat)
    ratio_stat_list.append(ratio_stat)
    print('All min')
    corr_stat, ratio_stat = cal_cstat_and_print_65(index_cal, slope.min(axis=1), ratio.min(axis=1), printtitle=f'      ', print_ratio=False)
    corr_stat_list.append(corr_stat)
    ratio_stat_list.append(ratio_stat)
    
    return corr_stat_list, ratio_stat_list, slope


def cal_print_increase_u_f_2(index_cal, network_acvs_moving, inputs, zeroid, actfun, b0):
    '''
    inputs:
    b0: 2ring b0 (scalar), 3ring [bE, bI]
    
    network_acvs_moving: 2d array (search_num, input_num) [3d array (ring_num, theta_num, time_num) ]or [3d array (theta_num, time_num) ]
    '''
    corr_stat_list = []
    ratio_stat_list = []
    bump_amplitudes = cal_firate_a_acv_mean_a_peak(network_acvs_moving, inputs, index_cal, b0, actfun, kind='increaseb0')
    
    search_num = network_acvs_moving.shape[0]
    acv = np.zeros((search_num, len(inputs), 4)) # 3rd dim: peaku, meanu, peakf, meanf
    slope = np.zeros((search_num, 4))
    ratio = np.zeros((search_num, 4))

    for i in index_cal:
        for j in range(4):
            acv[i,:,j] = np.mean(bump_amplitudes[j][i,:,:], axis=-1)
            res = linregress(inputs[zeroid:], acv[i,zeroid:,j])
            slope[i,j] = res.slope
            ratio[i,j] = res.slope / res.intercept

    print(f'Net num: {len(index_cal)}')
    printtitle = ['peak u', 'mean u', 'peak f', 'mean f']
    for i in range(4):
        print_ratio = True if i >= 2 else False
        corr_stat, ratio_stat = cal_cstat_and_print_65(index_cal, slope[:,i], ratio[:,i], printtitle=printtitle[i], print_ratio=print_ratio)
        corr_stat_list.append(corr_stat)
        ratio_stat_list.append(ratio_stat)
    print('All min except mean u')
    corr_stat, ratio_stat  = cal_cstat_and_print_65(index_cal, slope[:,[0,2,3]].min(axis=1), ratio[:,[0,2,3]].min(axis=1), printtitle=f'      ', print_ratio=True)
    corr_stat_list.append(corr_stat)
    ratio_stat_list.append(ratio_stat)
    
    
    print('All min')
    corr_stat, ratio_stat  = cal_cstat_and_print_65(index_cal, slope.min(axis=1), ratio.min(axis=1), printtitle=f'      ', print_ratio=False)
    corr_stat_list.append(corr_stat)
    ratio_stat_list.append(ratio_stat)
    
    return corr_stat_list, ratio_stat_list, slope


def cal_print_increase_u_f_3(index_cal, network_acvs_moving, inputs, zeroid, actfun, b0):
    '''
    inputs:
    b0: 3ring [bE, bI]
    
    network_acvs_moving: 2d array (search_num, input_num) [3d array (ring_num, theta_num, time_num) ]or [3d array (theta_num, time_num) ]
    '''
    corr_stat_list = []
    ratio_stat_list = []
    bump_amplitudes = cal_firate_a_acv_mean_a_peak(network_acvs_moving, inputs, index_cal, b0, actfun, kind='increaseb0')
    
    search_num = network_acvs_moving.shape[0]
    acv = np.zeros((search_num, len(inputs), 2, 4)) # 3rd dim: central ring, rotational ring; 4rd dim: peaku, meanu, peakf, meanf
    slope = np.zeros((search_num, 2, 4))
    ratio = np.zeros((search_num, 2, 4))

    for i in index_cal:
        for j in range(4):
            acv[i,:,0, j] = bump_amplitudes[j][i,:,0]
            acv[i,:,1, j] = np.mean(bump_amplitudes[j][i,:,1:], axis=1)
            for ringi in range(2):
                res = linregress(inputs[zeroid:], acv[i,zeroid:,ringi,j])
                slope[i,ringi,j] = res.slope
                ratio[i,ringi,j] = res.slope / res.intercept

    print(f'Net num: {len(index_cal)}')
    printtitle = [['Central peak u', 'rotational peak u'], ['Central mean u', 'rotational mean u'], ['Central peak f', 'rotational peak f'], ['Central mean f', 'rotational mean f']]
    for i in range(4):
        print_ratio = True if i >= 2 else False
        for ringi in range(2):
            corr_stat, ratio_stat = cal_cstat_and_print_65(index_cal, slope[:,ringi,i], ratio[:,ringi,i], printtitle=printtitle[i][ringi], print_ratio=print_ratio)
            corr_stat_list.append(corr_stat)
            ratio_stat_list.append(ratio_stat)
    print('All min except mean u')
    corr_stat, ratio_stat = cal_cstat_and_print_65(index_cal, slope[:,:,[0,2,3]].min(axis=(1,2)), ratio[:,:,[0,2,3]].min(axis=(1,2)), printtitle=f'      ', print_ratio=True)
    corr_stat_list.append(corr_stat)
    ratio_stat_list.append(ratio_stat)
    
    
    print('All min')
    corr_stat, ratio_stat = cal_cstat_and_print_65(index_cal, slope.min(axis=(1,2)), ratio.min(axis=(1,2)), printtitle=f'      ', print_ratio=False)
    corr_stat_list.append(corr_stat)
    ratio_stat_list.append(ratio_stat)
    
    return corr_stat_list, ratio_stat_list, slope
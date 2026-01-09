import numpy as np

def cal_input_2_meanu_slope(inputs, diffs, index, linear_range_id, kind='normal'):
    diffs = diffs[1] # only use the mean membran potential
    search_num = diffs.shape[0]
    slopes = np.zeros((search_num))

    for i in index:
        range_id = np.arange(linear_range_id[i][0], linear_range_id[i][1]+1).astype(int)
        
        x = inputs[range_id] if kind == 'normal' else inputs[range_id] * 0.1
        y = diffs[i][range_id]
        slopes[i] = np.sum(x*y)/ np.sum(x*x)
        # print(x,y,slopes[i])

    return slopes[index]

def cal_input_2_meanu_slope_bound(search_num, index, network_pars, phi, theta_num, theta_range, weight_fun, ring_num=2):
    
    slope_bounds = np.zeros((search_num))
    w_diff = np.zeros((search_num))
    
    for net_id in index:
    
        if ring_num == 2:
            aw = weight_fun(*network_pars[net_id], phi, theta_num, theta_range)
            ws_mean = aw[:theta_num,0].mean()
            wd_mean = aw[theta_num:,0].mean()
        elif ring_num == 3:
            try:
                aw = weight_fun(*network_pars[net_id], phi, theta_range)
            except:
                aw = weight_fun(*network_pars[net_id], phi, theta_num)
            ws_mean = aw[theta_num:theta_num*2,theta_num].mean()
            wd_mean = aw[theta_num:theta_num*2:,2*theta_num].mean()
            
        slope_bounds[net_id] = 2/(1-2*np.pi*(ws_mean-wd_mean))
        w_diff[net_id] = ws_mean-wd_mean
    
    return slope_bounds[index], w_diff[index]
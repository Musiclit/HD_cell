'''
Functions to create ring attractor network for simulation
'''
import numpy as np
from typing import Callable, List


# create network input with desired shape
def produce_inputb(bs, ratio, theta_num, ring_num):
    if ring_num == 1:
        b = np.repeat(bs, theta_num)
    elif ring_num == 2:
        b = produce_inputs_2ring(bs, ratio, theta_num)
    elif ring_num == 3:
        b = steady_inputb_withb0_3r(bs[0], bs[1], ratio, theta_num)
    return b


## Section: three rings
def steady_inputb_withb0_3r(bE, bI, ratio, theta_num):
    deltab = ratio
    bl = bI - deltab
    br = bI + deltab
    b = np.concatenate([np.repeat(bE, theta_num), np.repeat(bl, theta_num), np.repeat(br, theta_num)])
    return b


def steady_inputb_withb0_drosophila(bE, bI, ratio, theta_num):
    deltab = ratio
    bl = bI + max0x(-deltab)
    br = bI + max0x(deltab)
    b = np.concatenate([np.repeat(bE, theta_num), np.repeat(bl, theta_num), np.repeat(br, theta_num)])
    return b


def steady_inputb_asy_addb_3r(bE:float, bI:float, ratio:float, theta_num:int) -> np.ndarray:
    deltab = ratio
    bl = bI + max0x(-deltab)
    br = bI + max0x(deltab)
    b = np.concatenate([np.repeat(bE, theta_num), np.repeat(bl, theta_num), np.repeat(br, theta_num)])
    return b


def steady_inputb_asy_subb_3r(bE:float, bI:float, ratio:float, theta_num:int) -> np.ndarray:
    deltab = ratio
    bl = bI - max0x(deltab)
    br = bI - max0x(-deltab)
    b = np.concatenate([np.repeat(bE, theta_num), np.repeat(bl, theta_num), np.repeat(br, theta_num)])
    return b


## Section: two rings + readout
def steady_inputb_2rr(bc, b0, ratio, theta_num):
    deltab = ratio * b0
    bl = b0 - deltab
    br = b0 + deltab
    b = np.concatenate([np.repeat(bc, theta_num), np.repeat(bl, theta_num), np.repeat(br, theta_num)])
    return b


def steady_inputb_2rr_b_increase(bc, b0, ratio, theta_num):
    absv_b = 0.3 * np.sign(np.abs(ratio)) + 0.1 * np.abs(ratio)
    deltab = 0.1 * ratio * b0
    bl = b0 - deltab + absv_b
    br = b0 + deltab + absv_b
    bc = bc + absv_b
    b = np.concatenate([np.repeat(bc, theta_num), np.repeat(bl, theta_num), np.repeat(br, theta_num)])
    return b


## Section: two rings
def produce_inputs_2ring(b0, ratio, theta_num):
    deltab = ratio * b0
    bl = b0 - deltab
    br = b0 + deltab
    b = np.concatenate([np.repeat(bl, theta_num), np.repeat(br, theta_num)])
    return b


def steady_inputb_withb0(b0, ratio, theta_num):
    '''
    Create the input array of a two-ring network
    '''
    deltab = ratio * b0
    bl = b0 - deltab # input to the left ring
    br = b0 + deltab # input to the right ring
    b = np.concatenate([np.repeat(bl, theta_num), np.repeat(br, theta_num)])
    # Repeat input by the number of neurons to create the required input array
    return b


def steady_inputb_increaseb0(b0, ratio, theta_num, ratio2=1):
    deltab = ratio * b0
    bl = b0 * (1 - deltab + ratio2 * np.abs(deltab))
    br = b0 * (1 + deltab + ratio2 * np.abs(deltab))
    b = np.concatenate([np.repeat(bl, theta_num), np.repeat(br, theta_num)])
    return b


def steady_inputb_decreaseb0(b0, ratio, theta_num, ratio2=1):
    deltab = ratio * b0
    bl = b0 * (1 - deltab - ratio2 * np.abs(deltab))
    br = b0 * (1 + deltab - ratio2 * np.abs(deltab))
    b = np.concatenate([np.repeat(bl, theta_num), np.repeat(br, theta_num)])
    return b


def steady_inputb_with_asy_b0(b0, ratio, theta_num):
    deltab = ratio * b0
    bl = b0 
    br = b0 + deltab
    b = np.concatenate([np.repeat(bl, theta_num), np.repeat(br, theta_num)])
    return b


def steady_inputb_with_asy_b02(b0, ratio, theta_num):
    deltab = ratio * b0
    bl = b0 - deltab
    br = b0 
    b = np.concatenate([np.repeat(bl, theta_num), np.repeat(br, theta_num)])
    return b


def steady_inputb_nob0(b0, ratio, theta_num):
    deltab = ratio * b0
    bl = - deltab
    br = deltab
    b = np.concatenate([np.repeat(bl, theta_num), np.repeat(br, theta_num)])
    return b


# Section: create network initial state
def net_ini_v2(theta_range: np.ndarray, ring_num=3, offset=0):
    '''with adjustable ring num'''
    # Initial state: a bump in middle -> to speed up converge and fix peak location
    bump = np.cos(theta_range + offset) * 1e-8 # initial small pertubation of flat state
    s = np.where(bump>0, bump, 0) + \
        np.ones_like(bump) * 1e-8 # + 1e-8 to avoid true zero division
    s = np.concatenate([s]*ring_num)
    return s


def net_ini_v3(theta_range: List[np.ndarray], ring_num=3, offset=0):
    '''
    For three rings with unequal theta density
    '''
    theta_range = np.concatenate(theta_range)
    theta_num = len(theta_range)
    s = np.ones(theta_num)
    # Initial state: a bump in middle -> to speed up converge and fix peak location
    bump = np.cos(theta_range + offset) * 1e-8 # initial small pertubation of flat state
    s = np.where(bump>0, bump, 0) + s * 1e-8 # + 1e-8 to avoid true zero division
    return s


def net_ini(theta_num, theta_range):
    '''deprecated, for two rings'''
    # Store netwoek activation
    sl = np.ones(theta_num) # theta * t
    sr = np.ones(theta_num) # theta * t
    # Initial state: a bump in middle -> to speed up converge and fix peak location
    bump = np.cos(theta_range) * 1e-8 # initial small pertubation of flat state
    sl = np.where(bump>0, bump, 0) + sl * 1e-12 # + 1e-8 to avoid true zero division
    sr = np.where(bump>0, bump, 0) + sr * 1e-12 # + 1e-8 to avoid true zero division
    s = np.concatenate([sl, sr])
    return s


def net_ini_flat_one(theta_num, theta_range):
    # Store netwoek activation
    sl = np.ones(theta_num) # theta * t
    sr = np.ones(theta_num) # theta * t
    s = np.concatenate([sl, sr])
    return s


# Section: activation funtion
def max0x(x):
    y = np.where(x>0,x,0)
    return y


def piecewise_linear(x, x_max, x_min=0):
    y = np.where(x>x_max, x_max, x)
    y = np.where(y<x_min, x_min, y)
    return y


def hyperb(x,slope=1,x0=0):
    y = 0.5*(1+np.tanh(slope*(x-x0)))
    return y


def sigmoid(x, fmax=1, beta=0.8, b=10, c=-0.5):
    a = fmax/(np.log(1 + np.exp(b * (1 - c))) ** beta)
    y = a * np.log(1 + np.exp(b * (x - c))) ** beta
    return y


def sigmoid2(x, fmax=1, beta=1.3, b=1, c=-0.5):
    a = fmax/(np.log(1 + np.exp(b * (1 - c))) ** beta)
    y = a * np.log(1 + np.exp(b * (x - c))) ** beta
    return y


def tanh1(x):
    return np.tanh(x) + 1


# Section: Weight functioin
def cos(theta, J0, J1, phi, theta_range):
    y = J0+J1*np.cos(theta+phi-theta_range)
    return y


def box(theta, J0, J1, phi, theta_range):
    y = J0+J1*np.cos(theta+phi-theta_range)
    y[y>0] = J0+J1
    y[y<=0] = J0-J1
    return y


def vonmises_weight_drosophila(J0, J1, kappa, phi, theta_range):
    
    '''
    Weight function for the model of drosophila
    J0 shall be smaller than zero for drosophila's network
    '''

    theta_num = len(theta_range)
    w_sr = np.ones([theta_num, theta_num])
    w_sl = np.ones([theta_num, theta_num])
    w_ps = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        w_ps[thetai] = vonmises2(theta, J0, J1, 0, kappa, theta_range)
        w_sl[thetai] = vonmises2(theta, 0, J1, -phi, kappa, theta_range) + 0.5 * vonmises2(theta, 0, J1, 0, kappa, theta_range)
        w_sr[thetai] = vonmises2(theta, 0, J1, phi, kappa, theta_range) + 0.5 * vonmises2(theta, 0, J1, 0, kappa, theta_range)

    w = np.concatenate([np.concatenate([wz, w_sl, w_sr], axis=1),
                        np.concatenate([w_ps, wz, wz], axis=1),
                        np.concatenate([w_ps, wz, wz], axis=1)])
    return w


## Section: three rings
def cos_weight_3r(K0, K1, H0, H1, L0, alpha, theta_range):
    phi = -np.pi + alpha
    theta_num = len(theta_range)
    wL = np.ones([theta_num, theta_num]) * L0
    wH = np.ones([theta_num, theta_num])
    wEl = np.ones([theta_num, theta_num])
    wEr = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        wH[thetai] = cos(theta, H0, H1, 0, theta_range)
        wEl[thetai] = cos(theta, K0, K1, phi, theta_range)
        wEr[thetai] = cos(theta, K0, K1, -phi, theta_range)
    w = np.concatenate([np.concatenate([wz, -wEl, -wEr], axis=1),
                        np.concatenate([wH, wz, -wL], axis=1),
                        np.concatenate([wH, -wL, wz], axis=1)])
    return w


def cos_weight_3r_Icos(K0, K1, H0, H1, L0, alpha, theta_range):
    phi = -np.pi + alpha
    theta_num = len(theta_range)
    wII = np.ones([theta_num, theta_num])
    wH = np.ones([theta_num, theta_num])
    wEl = np.ones([theta_num, theta_num])
    wEr = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        wH[thetai] = cos(theta, H0, H1, 0, theta_range)
        wEl[thetai] = cos(theta, K0, K1, phi, theta_range)
        wEr[thetai] = cos(theta, K0, K1, -phi, theta_range)
        wII[thetai] = cos(theta, L0, L0, np.pi, theta_range)
    w = np.concatenate([np.concatenate([wz, -wEl, -wEr], axis=1),
                        np.concatenate([wH, wz, -wII], axis=1),
                        np.concatenate([wH, -wII, wz], axis=1)])
    return w


def cos_weight_3r_Icos_s(K0, H0, L0, alpha, theta_range):
    phi = -np.pi + alpha
    theta_num = len(theta_range)
    wII = np.ones([theta_num, theta_num])
    wH = np.ones([theta_num, theta_num])
    wEl = np.ones([theta_num, theta_num])
    wEr = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        wH[thetai] = cos(theta, H0, H0, 0, theta_range)
        wEl[thetai] = cos(theta, K0, K0, phi, theta_range)
        wEr[thetai] = cos(theta, K0, K0, -phi, theta_range)
        wII[thetai] = cos(theta, L0, L0, np.pi, theta_range)
    w = np.concatenate([np.concatenate([wz, -wEl, -wEr], axis=1),
                        np.concatenate([wH, wz, -wII], axis=1),
                        np.concatenate([wH, -wII, wz], axis=1)])
    return w


def cos_weight_3r_allI(KCL, KLC, KLL, alpha, theta_range):
    phi = -np.pi + alpha
    theta_num = len(theta_range)
    wLC = np.ones([theta_num, theta_num])
    wCl = np.ones([theta_num, theta_num])
    wCr = np.ones([theta_num, theta_num])
    wLL = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        wLC[thetai] = cos(theta, KLC, KLC, np.pi, theta_range)
        wCl[thetai] = cos(theta, KCL, KCL, phi, theta_range)
        wCr[thetai] = cos(theta, KCL, KCL, -phi, theta_range)
        wLL[thetai] = cos(theta, KLL, KLL, np.pi, theta_range)
    w = np.concatenate([np.concatenate([wz, -wCl, -wCr], axis=1),
                        np.concatenate([-wLC, wz, -wLL], axis=1),
                        np.concatenate([-wLC, -wLL, wz], axis=1)])
    return w


def cos_weight_3r_allI_more(KCL0, KCL1, KLC0, KLC1, KLL0, KLL1, alpha, theta_range):
    phi = -np.pi + alpha
    theta_num = len(theta_range)
    wLC = np.ones([theta_num, theta_num])
    wCl = np.ones([theta_num, theta_num])
    wCr = np.ones([theta_num, theta_num])
    wLL = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        wLC[thetai] = cos(theta, KLC0+KLC1, KLC1, np.pi, theta_range)
        wCl[thetai] = cos(theta, KCL0+KCL1, KCL1, phi, theta_range)
        wCr[thetai] = cos(theta, KCL0+KCL1, KCL1, -phi, theta_range)
        wLL[thetai] = cos(theta, KLL0+KLL1, KLL1, np.pi, theta_range)
    w = np.concatenate([np.concatenate([wz, -wCl, -wCr], axis=1),
                        np.concatenate([-wLC, wz, -wLL], axis=1),
                        np.concatenate([-wLC, -wLL, wz], axis=1)])
    return w


def cos_weight_3r_Icos_s_IIs(K0, H0, L0, alpha, theta_range):
    phi = -np.pi + alpha
    theta_num = len(theta_range)
    wII = np.ones([theta_num, theta_num])
    wH = np.ones([theta_num, theta_num])
    wEl = np.ones([theta_num, theta_num])
    wEr = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        wH[thetai] = cos(theta, H0, H0, 0, theta_range)
        wEl[thetai] = cos(theta, K0, K0, phi, theta_range)
        wEr[thetai] = cos(theta, K0, K0, -phi, theta_range)
        wII[thetai] = cos(theta, L0, L0, np.pi, theta_range)
    w = np.concatenate([np.concatenate([wz, -wEl, -wEr], axis=1),
                        np.concatenate([wH, -wII, -wII], axis=1),
                        np.concatenate([wH, -wII, -wII], axis=1)])
    return w


def cos_weight_3r_Icos_half_lateral(K0, K1, H0, H1, L0, alpha, theta_range):
    phi = -np.pi + alpha
    theta_num = len(theta_range)
    wII = np.ones([theta_num, theta_num])
    wH = np.ones([theta_num, theta_num])
    wEl = np.ones([theta_num, theta_num])
    wEr = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        wH[thetai] = cos(theta, H0, H1, 0, theta_range)
        wEl[thetai] = cos(theta, K0, K1, phi, theta_range)
        wEr[thetai] = cos(theta, K0, K1, -phi, theta_range)
        wII[thetai] = cos(theta, L0, L0, np.pi, theta_range)
    wEl[:,:theta_num//2] = 0
    wEr[:,theta_num//2:] = 0
    
    wlr = wII.copy()
    wrl = wII.copy()
    wlE = wH.copy()
    wrE = wH.copy()
    
    wlE[:theta_num//2,:] = 0
    wrE[theta_num//2:,:] = 0
    
    wlr[:theta_num//2,:] = 0
    wlr[:,theta_num//2:] = 0
    
    wrl[theta_num//2:,:] = 0
    wrl[:,:theta_num//2] = 0
    w = np.concatenate([np.concatenate([wz, -wEl, -wEr], axis=1),
                        np.concatenate([wlE, wz, -wlr], axis=1),
                        np.concatenate([wrE, -wII, wz], axis=1)])
    return w


def cos_weight_3r_Icos_decrease(K0, K1, H0, H1, L0, alpha, theta_range, ratio, ratiomax=0.9):
    phi = -np.pi + alpha
    theta_num = len(theta_range)
    wII = np.ones([theta_num, theta_num])
    wH = np.ones([theta_num, theta_num])
    wEl = np.ones([theta_num, theta_num])
    wEr = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        wH[thetai] = cos(theta, H0, H1, 0, theta_range)
        wEl[thetai] = cos(theta, K0, K1, phi, theta_range)
        wEr[thetai] = cos(theta, K0, K1, -phi, theta_range)
        wII[thetai] = cos(theta, L0, L0, np.pi, theta_range)
    w = np.concatenate([np.concatenate([wz, -wEl, -wEr], axis=1),
                        np.concatenate([wH, wz, -wII], axis=1),
                        np.concatenate([wH, -wII, wz], axis=1)])
    return w * (ratiomax - abs(ratio)) / ratiomax


def vonMises_weight_3r(sigmaE, sigmaI, sigmaII, alpha, theta_range):
    '''
    The parameter setup is similar to Song's paper
    '''
    phi = -np.pi + alpha
    theta_num = len(theta_range)
    wII = np.ones([theta_num, theta_num])
    wH = np.ones([theta_num, theta_num])
    wEl = np.ones([theta_num, theta_num])
    wEr = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        wH[thetai] = vonmises_Song(theta, 0, sigmaE, theta_range)
        wEl[thetai] = vonmises_Song(theta, phi, sigmaI, theta_range)
        wEr[thetai] = vonmises_Song(theta, -phi, sigmaI, theta_range)
        wII[thetai] = vonmises_Song(theta, np.pi, sigmaII, theta_range)
    w = np.concatenate([np.concatenate([wz, -wEl, -wEr], axis=1),
                        np.concatenate([wH, -wII, -wII], axis=1),
                        np.concatenate([wH, -wII, -wII], axis=1)])
    return w


def vonMises_weight_3r_v2(cI, cE, lI, kappa, alpha, theta_range):
    '''
    the lateral rings don't connect to themselves
    '''
    phi = -np.pi + alpha
    theta_num = len(theta_range)
    wII = np.ones([theta_num, theta_num])
    wH = np.ones([theta_num, theta_num])
    wEl = np.ones([theta_num, theta_num])
    wEr = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        wH[thetai] = vonmises_3r(theta, cE, 0, kappa, theta_range)
        wEl[thetai] = vonmises_3r(theta, cI, phi, kappa, theta_range)
        wEr[thetai] = vonmises_3r(theta, cI, -phi, kappa, theta_range)
        wII[thetai] = vonmises_3r(theta, lI, np.pi, kappa, theta_range)
    w = np.concatenate([np.concatenate([wz, -wEl, -wEr], axis=1),
                        np.concatenate([wH, wz, -wII], axis=1),
                        np.concatenate([wH, -wII, wz], axis=1)])
    return w


def vonMises_weight_3r_v2_II(cI, cE, lI, kappa, alpha, theta_range):
    '''
    the lateral rings connect to themselves
    '''
    phi = -np.pi + alpha
    theta_num = len(theta_range)
    wII = np.ones([theta_num, theta_num])
    wH = np.ones([theta_num, theta_num])
    wEl = np.ones([theta_num, theta_num])
    wEr = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        wH[thetai] = vonmises_3r(theta, cE, 0, kappa, theta_range)
        wEl[thetai] = vonmises_3r(theta, cI, phi, kappa, theta_range)
        wEr[thetai] = vonmises_3r(theta, cI, -phi, kappa, theta_range)
        wII[thetai] = vonmises_3r(theta, lI, np.pi, kappa, theta_range)
    w = np.concatenate([np.concatenate([wz, -wEl, -wEr], axis=1),
                        np.concatenate([wH, -wII, -wII], axis=1),
                        np.concatenate([wH, -wII, -wII], axis=1)])
    return w


def vonmises_3r(theta, J1, phi, kappa, theta_range):
    y = J1*np.exp( kappa*( np.cos((theta+phi)+np.flip(theta_range)) - 1 ) )
    return y

def vonmises_Song(theta, phi, sigma, theta_range):
    y = np.exp( np.cos((theta+phi)+np.flip(theta_range)) / (sigma * np.pi / 180)**2 )
    y = y * len(theta_range) / np.sum(y) # normalize, average value of neuron's weight is 1
    return y


## Section: 2ring + 1readout
def cos_weight_2r_r(J0, J1, phi, theta_num=50, theta_range=None):
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) if theta_range is None else theta_range
    w_sl = np.ones([theta_num, theta_num]) # result theta, input theta
    w_dl = np.ones([theta_num, theta_num])
    w_sr = np.ones([theta_num, theta_num])
    w_dr = np.ones([theta_num, theta_num])
    w_sym = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        w_sl[thetai] = cos(theta, J0, J1, -phi, theta_range)
        w_dl[thetai] = cos(theta, J0, J1, phi, theta_range)
        w_sr[thetai] = cos(theta, J0, J1, phi, theta_range)
        w_dr[thetai] = cos(theta, J0, J1, -phi, theta_range)
        w_sym[thetai] = cos(theta, J0, J1, 0, theta_range)
    w = np.concatenate([np.concatenate([wz, w_sym, w_sym], axis=1),
                        np.concatenate([wz, w_sl, w_dl], axis=1),
                        np.concatenate([wz, w_dr, w_sr], axis=1)])
    return w


def cos_weight_2rr_3p(J0, J1, k0, phi, theta_num, theta_range):
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) if theta_range is None else theta_range
    w_sl = np.ones([theta_num, theta_num]) # result theta, input theta
    w_dl = np.ones([theta_num, theta_num])
    w_sr = np.ones([theta_num, theta_num])
    w_dr = np.ones([theta_num, theta_num])
    w_sym = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        w_sl[thetai] = cos(theta, J0, J1, -phi, theta_range)
        w_dl[thetai] = cos(theta, k0, J1, phi, theta_range)
        w_sr[thetai] = cos(theta, J0, J1, phi, theta_range)
        w_dr[thetai] = cos(theta, k0, J1, -phi, theta_range)
        w_sym[thetai] = cos(theta, J0, J1, 0, theta_range)
    w = np.concatenate([np.concatenate([wz, w_sym, w_sym], axis=1),
                        np.concatenate([wz, w_sl, w_dl], axis=1),
                        np.concatenate([wz, w_dr, w_sr], axis=1)])
    return w


def cos_weight_2rr_3p_cut_excit_no_self(J0, J1, k0, phi, theta_num, theta_range=None):
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) if theta_range is None else theta_range
    w_sl = np.ones([theta_num, theta_num]) # result theta, input theta
    w_dl = np.ones([theta_num, theta_num])
    w_sr = np.ones([theta_num, theta_num])
    w_dr = np.ones([theta_num, theta_num])
    w_listq = [w_sl, w_dl, w_sr, w_dr]
    w_sym = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])

    for thetai, theta in enumerate(theta_range):
        w_sl[thetai] = cos(theta, J0, J1, -phi, theta_range)
        w_dl[thetai] = cos(theta, k0, J1, phi, theta_range)
        w_sr[thetai] = cos(theta, J0, J1, phi, theta_range)
        w_dr[thetai] = cos(theta, k0, J1, -phi, theta_range)
        J_all = np.sum(w_sl[thetai]) - w_sl[thetai, thetai]
        w_sym[thetai, (thetai + theta_num//2) % theta_num] = -J_all / 2

    for w in w_listq:
        w[w>0] = 0
    np.fill_diagonal(w_sl, 0)
    np.fill_diagonal(w_sr, 0)

    w = np.concatenate([np.concatenate([wz, w_sym, w_sym], axis=1),
                        np.concatenate([wz, w_sl, w_dl], axis=1),
                        np.concatenate([wz, w_dr, w_sr], axis=1)])
    return w


def vonmises_weight_s_2rr(J0, J1, kappa, phi, theta_num, theta_range):
    w_sl = np.ones([theta_num, theta_num]) # result theta, input theta
    w_dl = np.ones([theta_num, theta_num])
    w_sr = np.ones([theta_num, theta_num])
    w_dr = np.ones([theta_num, theta_num])
    w_sym = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        w_sl[thetai] = vonmises(theta, J0, J1, -phi, kappa, theta_range)
        w_dl[thetai] = vonmises(theta, J0, J1, phi, kappa, theta_range)
        w_sr[thetai] = vonmises(theta, J0, J1, phi, kappa, theta_range)
        w_dr[thetai] = vonmises(theta, J0, J1, -phi, kappa, theta_range)
        w_sym[thetai] = vonmises(theta, J0, J1, 0, kappa, theta_range)

    w = np.concatenate([np.concatenate([wz, w_sym, w_sym], axis=1),
                        np.concatenate([wz, w_sl, w_dl], axis=1),
                        np.concatenate([wz, w_dr, w_sr], axis=1)])
    return w


def vonmises_weight_2rr_3p(J0, J1, k0, kappa, phi, theta_num, theta_range):
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) if theta_range is None else theta_range
    w_sl = np.ones([theta_num, theta_num]) # result theta, input theta
    w_dl = np.ones([theta_num, theta_num])
    w_sr = np.ones([theta_num, theta_num])
    w_dr = np.ones([theta_num, theta_num])
    w_sym = np.ones([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        w_sl[thetai] = vonmises(theta, J0, J1, -phi, kappa, theta_range)
        w_dl[thetai] = vonmises(theta, k0, J1, phi, kappa, theta_range)
        w_sr[thetai] = vonmises(theta, J0, J1, phi, kappa, theta_range)
        w_dr[thetai] = vonmises(theta, k0, J1, -phi, kappa, theta_range)
        w_sym[thetai] = vonmises(theta, J0, J1, 0, kappa, theta_range)
    w = np.concatenate([np.concatenate([wz, w_sym, w_sym], axis=1),
                        np.concatenate([wz, w_sl, w_dl], axis=1),
                        np.concatenate([wz, w_dr, w_sr], axis=1)])
    return w


def vonmises_weight_2i1r_3p_cut_excit(J0, J1, k0, kappa, phi, theta_num, theta_range=None):
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) if theta_range is None else theta_range

    w_sl = np.zeros([theta_num, theta_num]) # result theta, input theta
    w_dl = np.zeros([theta_num, theta_num])
    w_sr = np.zeros([theta_num, theta_num])
    w_dr = np.zeros([theta_num, theta_num])
    w_listq = [w_sl, w_dl, w_sr, w_dr]
    w_sym = np.zeros([theta_num, theta_num])
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        w_sl[thetai] = vonmises(theta, J0, J1, -phi, kappa, theta_range)
        w_dl[thetai] = vonmises(theta, k0, J1, phi, kappa, theta_range)
        w_sr[thetai] = vonmises(theta, J0, J1, phi, kappa, theta_range)
        w_dr[thetai] = vonmises(theta, k0, J1, -phi, kappa, theta_range)
        w_sym[thetai, (thetai + theta_num//2) % theta_num] = -1
    for w in w_listq:
        w[w>0] = 0
    w = np.concatenate([np.concatenate([wz, w_sym, w_sym], axis=1),
                        np.concatenate([wz, w_sl, w_dl], axis=1),
                        np.concatenate([wz, w_dr, w_sr], axis=1)])
    return w


def vonmises_weight_2i1r_3p_cut_excit_no_self(J0, J1, k0, kappa, phi, theta_num, theta_range=None):
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) if theta_range is None else theta_range

    w_sl = np.zeros([theta_num, theta_num]) # result theta, input theta
    w_dl = np.zeros([theta_num, theta_num])
    w_sr = np.zeros([theta_num, theta_num])
    w_dr = np.zeros([theta_num, theta_num])
    w_sym = np.zeros([theta_num, theta_num])
    w_listq = [w_sl, w_dl, w_sr, w_dr, w_sym]
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        # theta_range around thetai should be eliminated
        w_sl[thetai] = vonmises(theta, J0, J1, -phi, kappa, theta_range)
        w_dl[thetai] = vonmises(theta, k0, J1, phi, kappa, theta_range)
        w_sr[thetai] = vonmises(theta, J0, J1, phi, kappa, theta_range)
        w_dr[thetai] = vonmises(theta, k0, J1, -phi, kappa, theta_range)
        w_sym[thetai] = vonmises(theta, -J1/2, J1, np.pi, 5, theta_range)
        # J_all = np.sum(w_sl[thetai]) - w_sl[thetai, thetai]
        # w_sym[thetai, (thetai + theta_num//2) % theta_num] = J_all / 2 # it is negative
    for w in w_listq:
        w[w<0] = 0
    np.fill_diagonal(w_sl, 0)
    np.fill_diagonal(w_sr, 0)
    w = np.concatenate([np.concatenate([wz, -w_sym, -w_sym], axis=1),
                        np.concatenate([wz, -w_sl, -w_dl], axis=1),
                        np.concatenate([wz, -w_dr, -w_sr], axis=1)])
    return w


def vonmises_weight_2i1r_2(J0, J1, k0, kappa, phi, theta_num, theta_range=None):
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) if theta_range is None else theta_range

    w_sl = np.zeros([theta_num, theta_num]) # result theta, input theta
    w_dl = np.zeros([theta_num, theta_num])
    w_sr = np.zeros([theta_num, theta_num])
    w_dr = np.zeros([theta_num, theta_num])
    w_sym = np.zeros([theta_num, theta_num])
    w_listq = [w_sl, w_dl, w_sr, w_dr, w_sym]
    wz = np.zeros([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        # theta_range around thetai should be eliminated
        w_sl[thetai] = vonmises(theta, J0, J1, -phi, kappa, theta_range)
        w_dl[thetai] = vonmises(theta, k0, J1, phi, kappa, theta_range)
        w_sr[thetai] = vonmises(theta, J0, J1, phi, kappa, theta_range)
        w_dr[thetai] = vonmises(theta, k0, J1, -phi, kappa, theta_range)
        w_sym[thetai] = vonmises(theta, k0, J1, np.pi, kappa, theta_range)
    for w in w_listq:
        w[w<0] = 0
    np.fill_diagonal(w_sl, 0)
    np.fill_diagonal(w_sr, 0)
    w = np.concatenate([np.concatenate([wz, -w_sym, -w_sym], axis=1),
                        np.concatenate([wz, -w_sl, -w_dl], axis=1),
                        np.concatenate([wz, -w_dr, -w_sr], axis=1)])
    return w


def vonmises_weight_2i1r_unequal_theta(J0, J1, k0, kappa, phi, theta_num, theta_range, dtheta):
    
    
    
    scale = dtheta/(2*np.pi/theta_num)

    w_cl = np.zeros([theta_num, theta_num]) # result theta, input theta
    w_cr = np.zeros([theta_num, theta_num])
    w_ll = np.zeros([theta_num, theta_num])
    w_rl = np.zeros([theta_num, theta_num])
    w_rr = np.zeros([theta_num, theta_num])
    w_lr = np.zeros([theta_num, theta_num])
    w_listq = [w_cl, w_cr, w_ll, w_rl, w_rr, w_lr]
    wz = np.zeros([theta_num, theta_num])
    for thetai in range(theta_num):
        
        w_cl[thetai] = vonmises2(theta_range[0][thetai], k0, J1, np.pi, kappa, theta_range[1]) * scale[theta_num:2*theta_num]
        w_cr[thetai] = vonmises2(theta_range[0][thetai], k0, J1, np.pi, kappa, theta_range[2]) * scale[2*theta_num:]
        w_ll[thetai] = vonmises2(theta_range[1][thetai], J0, J1, -phi, kappa, theta_range[1]) * scale[theta_num:2*theta_num]
        # w_rl[thetai] = vonmises2(theta_range[1][thetai], k0, J1, phi, kappa, theta_range[2]) * scale[2*theta_num:]
        w_lr[thetai] = vonmises2(theta_range[1][thetai], k0, J1, phi, kappa, theta_range[2]) * scale[2*theta_num:]
        w_rr[thetai] = vonmises2(theta_range[2][thetai], J0, J1, phi, kappa, theta_range[2]) * scale[2*theta_num:]
        # w_lr[thetai] = vonmises2(theta_range[2][thetai], k0, J1, -phi, kappa, theta_range[1]) * scale[theta_num:2*theta_num]
        w_rl[thetai] = vonmises2(theta_range[2][thetai], k0, J1, -phi, kappa, theta_range[1]) * scale[theta_num:2*theta_num]
        
    for w in w_listq:
        w[w<0] = 0
    np.fill_diagonal(w_ll, 0)
    np.fill_diagonal(w_rr, 0)
    
    # w = np.concatenate([np.concatenate([wz, -w_cl, -w_cr], axis=1),
    #                     np.concatenate([wz, -w_ll, -w_rl], axis=1),
    #                     np.concatenate([wz, -w_lr, -w_rr], axis=1)])
    
    w = np.concatenate([np.concatenate([wz, -w_cl, -w_cr], axis=1),
                        np.concatenate([wz, -w_ll, -w_lr], axis=1),
                        np.concatenate([wz, -w_rl, -w_rr], axis=1)])
    return w


def vonmises_weight_2i1r_unequal_theta_acv_scale(J0, J1, k0, kappa, phi, theta_num, theta_range, dtheta):
    
    scale = dtheta/(2*np.pi/theta_num)

    w_cl = np.zeros([theta_num, theta_num]) # result theta, input theta
    w_cr = np.zeros([theta_num, theta_num])
    w_ll = np.zeros([theta_num, theta_num])
    w_rl = np.zeros([theta_num, theta_num])
    w_rr = np.zeros([theta_num, theta_num])
    w_lr = np.zeros([theta_num, theta_num])
    w_listq = [w_cl, w_cr, w_ll, w_rl, w_rr, w_lr]
    wz = np.zeros([theta_num, theta_num])
    for thetai in range(theta_num):
        
        w_cl[thetai] = vonmises2(theta_range[0][thetai], k0, J1, np.pi, kappa, theta_range[1])
        w_cr[thetai] = vonmises2(theta_range[0][thetai], k0, J1, np.pi, kappa, theta_range[2])
        w_ll[thetai] = vonmises2(theta_range[1][thetai], J0, J1, -phi, kappa, theta_range[1]) * scale[thetai + theta_num]
        w_lr[thetai] = vonmises2(theta_range[1][thetai], k0, J1, phi, kappa, theta_range[2]) * scale[thetai + theta_num]
        w_rr[thetai] = vonmises2(theta_range[2][thetai], J0, J1, phi, kappa, theta_range[2]) * scale[thetai + 2*theta_num]
        w_rl[thetai] = vonmises2(theta_range[2][thetai], k0, J1, -phi, kappa, theta_range[1]) * scale[thetai + 2*theta_num]
        
    for w in w_listq:
        w[w<0] = 0
    np.fill_diagonal(w_ll, 0)
    np.fill_diagonal(w_rr, 0)
    
    w = np.concatenate([np.concatenate([wz, -w_cl, -w_cr], axis=1),
                        np.concatenate([wz, -w_ll, -w_lr], axis=1),
                        np.concatenate([wz, -w_rl, -w_rr], axis=1)])
    return w


def vonmises_weight_2i1r_unequal_theta_v2(J0, J1, k0, kappa, phi, theta_num, theta_range, dtheta):
    '''
    Interesting, in this network, although the projection from left and right rings are skewed by their unequal density, but it still works
    '''
    
    scale = dtheta/(2*np.pi/theta_num)

    w_sl = np.zeros([theta_num, theta_num]) # result theta, input theta
    w_dl = np.zeros([theta_num, theta_num])
    w_sr = np.zeros([theta_num, theta_num])
    w_dr = np.zeros([theta_num, theta_num])
    w_sym = np.zeros([theta_num, theta_num])
    w_listq = [w_sl, w_dl, w_sr, w_dr, w_sym]
    wz = np.zeros([theta_num, theta_num])
    for thetai in range(theta_num):
        
        w_sym[thetai] = vonmises2(theta_range[0][thetai], k0, J1, np.pi, kappa, theta_range[0]) 
        w_sl[thetai] = vonmises2(theta_range[1][thetai], J0, J1, -phi, kappa, theta_range[1]) * scale[theta_num:2*theta_num]
        w_dl[thetai] = vonmises2(theta_range[1][thetai], k0, J1, phi, kappa, theta_range[2]) * scale[2*theta_num:]
        w_sr[thetai] = vonmises2(theta_range[2][thetai], J0, J1, phi, kappa, theta_range[2]) * scale[2*theta_num:]
        w_dr[thetai] = vonmises2(theta_range[2][thetai], k0, J1, -phi, kappa, theta_range[1]) * scale[theta_num:2*theta_num]
        
    for w in w_listq:
        w[w<0] = 0
    np.fill_diagonal(w_sl, 0)
    np.fill_diagonal(w_sr, 0)
    
    w = np.concatenate([np.concatenate([wz, -w_sym, -w_sym], axis=1),
                        np.concatenate([wz, -w_sl, -w_dl], axis=1),
                        np.concatenate([wz, -w_dr, -w_sr], axis=1)])
    return w


## Section: two rings
def cos_weight(J0, J1, phi, theta_num=50, theta_range=None):
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) if theta_range is None else theta_range
    np.ones([theta_num, theta_num])
    w_sl = np.ones([theta_num, theta_num]) # result theta, input theta
    w_dl = np.ones([theta_num, theta_num])
    w_sr = np.ones([theta_num, theta_num])
    w_dr = np.ones([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        w_sl[thetai] = cos(theta, J0, J1, -phi, theta_range)
        w_dl[thetai] = cos(theta, J0, J1, phi, theta_range)
        w_sr[thetai] = cos(theta, J0, J1, phi, theta_range)
        w_dr[thetai] = cos(theta, J0, J1, -phi, theta_range)
    ws = diag_mat([w_sl, w_sr])
    wd = subdiag_mat([w_dl, w_dr])
    return ws+wd


def cos_weight2(J0, J1, k0, k1, phi, psi, theta_num=50, theta_range=None):
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) if theta_range is None else theta_range
    w_sl = np.ones([theta_num, theta_num]) # result theta, input theta
    w_dl = np.ones([theta_num, theta_num])
    w_sr = np.ones([theta_num, theta_num])
    w_dr = np.ones([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        w_sl[thetai] = cos(theta, J0, J1, -phi, theta_range)
        w_dl[thetai] = cos(theta, k0, k1, psi, theta_range)
        w_sr[thetai] = cos(theta, J0, J1, phi, theta_range)
        w_dr[thetai] = cos(theta, k0, k1, -psi, theta_range)
    ws = diag_mat([w_sl, w_sr])
    wd = subdiag_mat([w_dl, w_dr])
    return ws+wd


def cos_weight3(J0, J1, k0, phi, theta_num=50, theta_range=None):
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) if theta_range is None else theta_range
    w_sl = np.ones([theta_num, theta_num]) # result theta, input theta
    w_dl = np.ones([theta_num, theta_num])
    w_sr = np.ones([theta_num, theta_num])
    w_dr = np.ones([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        w_sl[thetai] = cos(theta, J0, J1, -phi, theta_range)
        w_dl[thetai] = cos(theta, k0, J1, phi, theta_range)
        w_sr[thetai] = cos(theta, J0, J1, phi, theta_range)
        w_dr[thetai] = cos(theta, k0, J1, -phi, theta_range)
    ws = diag_mat([w_sl, w_sr])
    wd = subdiag_mat([w_dl, w_dr])
    return ws+wd


def box_weight(J0, J1, k0, k1, phi, psi, theta_num=100, theta_range=None):
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) if theta_range is None else theta_range
    w_sl = np.ones([theta_num, theta_num]) # result theta, input theta
    w_dl = np.ones([theta_num, theta_num])
    w_sr = np.ones([theta_num, theta_num])
    w_dr = np.ones([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        w_sl[thetai] = box(theta, J0, J1, -phi, theta_range)
        w_dl[thetai] = box(theta, k0, k1, psi, theta_range)
        w_sr[thetai] = box(theta, J0, J1, phi, theta_range)
        w_dr[thetai] = box(theta, k0, k1, -psi, theta_range)
    ws = diag_mat([w_sl, w_sr])
    wd = subdiag_mat([w_dl, w_dr])
    return ws+wd


def vonmises(theta, J0, J1, phi, kappa, theta_range):
    y = J0+J1*np.exp( kappa*( np.cos((theta+phi)+np.flip(theta_range)) - 1 ) )
    return y


def vonmises2(theta, J0, J1, phi, kappa, theta_range):
    '''
    the original version with np.flip seems incorrect
    '''
    y = J0+J1*np.exp( kappa*( np.cos(theta + phi - theta_range) - 1 ) )
    return y


def vonmises_weight(J0, J1, K0, K1, kappa, phi, psi, theta_num, theta_range):
    w_sl = np.ones([theta_num, theta_num]) # result theta, input theta
    w_dl = np.ones([theta_num, theta_num])
    w_sr = np.ones([theta_num, theta_num])
    w_dr = np.ones([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        w_sl[thetai] = vonmises(theta, J0, J1, -phi, kappa, theta_range)
        w_dl[thetai] = vonmises(theta, K0, K1, psi, kappa, theta_range)
        w_sr[thetai] = vonmises(theta, J0, J1, phi, kappa, theta_range)
        w_dr[thetai] = vonmises(theta, K0, K1, -psi, kappa, theta_range)
    ws = diag_mat([w_sl, w_sr])
    wd = subdiag_mat([w_dl, w_dr])
    return ws+wd


def vonmises_weight3(J0, J1, K0, kappa, phi, theta_num, theta_range=None):
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) if theta_range is None else theta_range
    w_sl = np.ones([theta_num, theta_num]) # result theta, input theta
    w_dl = np.ones([theta_num, theta_num])
    w_sr = np.ones([theta_num, theta_num])
    w_dr = np.ones([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        w_sl[thetai] = vonmises(theta, J0, J1, -phi, kappa, theta_range)
        w_dl[thetai] = vonmises(theta, K0, J1, phi, kappa, theta_range)
        w_sr[thetai] = vonmises(theta, J0, J1, phi, kappa, theta_range)
        w_dr[thetai] = vonmises(theta, K0, J1, -phi, kappa, theta_range)
    ws = diag_mat([w_sl, w_sr])
    wd = subdiag_mat([w_dl, w_dr])
    return ws+wd


def vonmises_weight_s(J0, J1, kappa, phi, theta_num, theta_range=None):
    dtheta = (2*np.pi)/theta_num
    theta_range = np.arange(-np.pi+dtheta/2, np.pi, dtheta) if theta_range is None else theta_range
    w_sl = np.ones([theta_num, theta_num]) # result theta, input theta
    w_dl = np.ones([theta_num, theta_num])
    w_sr = np.ones([theta_num, theta_num])
    w_dr = np.ones([theta_num, theta_num])
    for thetai, theta in enumerate(theta_range):
        w_sl[thetai] = vonmises(theta, J0, J1, -phi, kappa, theta_range)
        w_dl[thetai] = vonmises(theta, J0, J1, phi, kappa, theta_range)
        w_sr[thetai] = vonmises(theta, J0, J1, phi, kappa, theta_range)
        w_dr[thetai] = vonmises(theta, J0, J1, -phi, kappa, theta_range)
    ws = diag_mat([w_sl, w_sr])
    wd = subdiag_mat([w_dl, w_dr])
    return ws+wd


## Section: one ring
def asym_vw(JI, JE, theta_range, deltab=0):
    '''Follow the formula in Zhang, 1996 J Neurosci'''
    N = len(theta_range)
    sine_matrix = np.zeros((N,N)) # Initialize
    for i in range(N):
        sine_matrix[i] = theta_range[i] - theta_range
    return JI + JE * np.cos(sine_matrix) - deltab * JE * np.sin(sine_matrix)


def asym_vw_wde(JI, JE, theta_range, deltab, ratio=0.5):
    N = len(theta_range)
    sine_matrix = np.zeros((N,N))
    for i in range(N):
        sine_matrix[i] = theta_range[i] - theta_range
    JI = JI * (1 - ratio * deltab)
    JE = JE * (1 - ratio * deltab) + 4 * ratio * deltab
    return JI + JE * np.cos(sine_matrix) - deltab * JE * np.sin(sine_matrix)


def asym_vw_wde_general(JI, JE, theta_range, deltab, ratio=0.5):
    N = len(theta_range)
    sine_matrix = np.zeros((N,N))
    for i in range(N):
        sine_matrix[i] = theta_range[i] - theta_range
    JI = JI * (1 - ratio * np.abs(deltab))
    JE = JE * (1 - ratio * np.abs(deltab))
    return JI + JE * np.cos(sine_matrix) - deltab * JE * np.sin(sine_matrix)


def asym_vw_wincrease(JI, JE, theta_range, deltab, ratio=0.5):
    N = len(theta_range)
    sine_matrix = np.zeros((N,N))
    for i in range(N):
        sine_matrix[i] = theta_range[i] - theta_range
    JI = JI * (1 + ratio * np.abs(deltab))
    JE = JE * (1 + ratio * np.abs(deltab))
    return JI + JE * np.cos(sine_matrix) - deltab * JE * np.sin(sine_matrix)


def asym_vw_vonMises(JI, JE, kappa, theta_range, deltab=0):
    N = len(theta_range)
    sine_matrix = np.zeros((N,N))
    for i in range(N):
        sine_matrix[i] = theta_range[i] - theta_range

    w = JI + JE * np.exp( kappa*( np.cos(sine_matrix) - 1 ) ) - \
        deltab * JE * np.exp( kappa * ( np.cos(sine_matrix) - 1 )) * kappa * np.sin(sine_matrix)
    return w


# section: network integration equations
def net_diff_equa_f_in(t, y, w, tau, b, dtheta, actfun, funargs=None, c=0):
    f = actfun(y + b, *funargs) if funargs != None else actfun(y + b)
    wf = np.matmul(w,f)*dtheta/(2*np.pi) # i.e., *1/50
    return 1/tau*(-y + wf + c)


def net_diff_equa_f_in_noise(t, y, w, tau, b, dtheta, actfun, noise, funargs=None, c=0):
    f = actfun(y + b, *funargs) * (1 + noise) if funargs != None else actfun(y + b) * (1 + noise)
    wf = np.matmul(w,f)*dtheta/(2*np.pi)
    return 1/tau*(-y + wf + c)


def net_diff_equa_f_in_dleaky(t, y, w, tau, b, dtheta, actfun, deltab, funargs=None, c=0):
    f = actfun(y + b, *funargs) if funargs != None else actfun(y + b)
    wf = np.matmul(w,f)*dtheta/(2*np.pi)
    return 1/tau*(-y * (1-0.5*deltab) + wf + c)


def net_diff_equa_f_out(t, y, w, tau, b, dtheta, actfun, funargs=None, c=0):
    wu = np.matmul(w,y)*dtheta/(2*np.pi)
    f = actfun(wu + b, *funargs) if funargs != None else actfun(wu + b)
    return 1/tau*(-y + f + c)


def get_net_result(solve_result, w, b, theta_num, dtheta, actfun, funargs=None):
    t = solve_result.t
    y = solve_result.y
    sl = y[:theta_num]
    sr = y[theta_num:]
    return sl, sr, t


def explode_event(t, y, *args):
    '''
    Event used in solve_ivp to stop the integration when the network activity explodes
    1e9 is a hand-tuned threshold
    '''
    return np.max(y) - 1e9

# Terminate the integration when this event occurs
explode_event.terminal = True
# Only trigger when the value goes from negative to positive
explode_event.direction = 1


def net_var_append(oldvar, newvar):
    '''
    When the last axis's first element of each newvar is the same
    as each oldvar's end element, 
    Use this function to discard the repeating elements when concatenating.
    
    Parameters:
    oldvar: list of np.array (1d or 2d, same shape as newvar)
        The original variable to be appended
    newvar: list of np.array (1d or 2d, same shape as oldvar)
        The new variable to be appended to oldvar
        
    Returns:
    oldvar: list of np.array (1d or 2d, same shape as oldvar)
        The concatenated variable, concatenated along the last axis
    '''
    # Newvar only has one element,
    # which should be the same as oldvar's last one.
    # No need to append, return oldvar directly.
    if len(newvar[-1]) == 1:
        return oldvar
    
    # For each variable in the oldvar list
    for i in range(len(oldvar)):
        axis = len(oldvar[i].shape) - 1 # get its last axis
        
        # Delete the first element along the last axis in newvar
        # Only works for 1d or 2d array
        if axis == 0:
            newvar[i] = newvar[i][1:]
        elif axis == 1:
            newvar[i] = newvar[i][:,1:]
        else:
            raise ValueError('input should not exceed 2 dimension')
        
        # Concatenate along the last axis
        oldvar[i] = np.concatenate([oldvar[i], newvar[i]], axis=axis)
        
    return oldvar


# Diagonal matrix
def subdiag_mat(rem=[]):
    z = np.zeros_like(rem[0])
    result = np.block(
        [
            [z, rem[0]],
            [rem[1], z],
        ]
    )
    return result


def diag_mat(rem=[], result=np.empty((0, 0))):
    if not rem:
        return result
    m = rem.pop(0)
    result = np.block(
        [
            [result, np.zeros((result.shape[0], m.shape[1]))],
            [np.zeros((m.shape[0], result.shape[1])), m],
        ]
    )
    return diag_mat(rem, result)


# Network evaluation
def sep_slist(y, ring_num):
    '''
    convert the activity array from (theta_num*ring_num) 
    to a list of each ring's activity
    Order: symmetric, left, right
    '''
    theta_num = len(y) // ring_num
    if ring_num == 1:
        slist = [y]
    elif ring_num == 2:
        slist = [y[:theta_num], y[theta_num:2*theta_num]]
    elif ring_num == 3:
        slist = [y[:theta_num], y[theta_num:2*theta_num], y[2*theta_num:]]
    
    return slist
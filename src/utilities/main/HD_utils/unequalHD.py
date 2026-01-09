'''Create networks with unequal HD distribution for zebrafish HD system'''
import numpy as np

def asy_theta_range(theta_range_sym, shift):
    dtheta = ((np.cos(theta_range_sym + shift) + 1) * 0.3 + 0.1)
    dtheta = dtheta / np.sum(dtheta) * 2 * np.pi
    theta_range = np.cumsum(dtheta) - dtheta/2
    return dtheta, theta_range - np.pi
'''
Functions to create specific types of matrics
'''
import numpy as np


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

def sparse_matrix(w): # w must have n*n dimension, n//2=0
    n = w.shape[0]
    for i in range(n):
        starti = i%2
        for j in range(n//2):
            w[i,starti+j*2] = 0
    return w
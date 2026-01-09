import numpy as np
from numba import njit, prange


@njit(parallel=True)
def quantify_corr_with_heading(phase, fictive_heading, wnd_pts=500):
    correlations = np.zeros(len(phase) - 2 * wnd_pts)
    for i in prange(len(correlations)):
        t_slice = slice(i, i + wnd_pts * 2)
        correlations[i] = np.corrcoef(phase[t_slice], fictive_heading[t_slice])[0, 1]

    return correlations

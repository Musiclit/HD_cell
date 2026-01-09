import numpy as np

from lotr.plotting.stack_coloring import _fill_roi_stack


def make_proj(roi_stack, traces, idxs, i):
    n_cells = traces.shape[1]
    filling = np.zeros((n_cells, 1))
    filling[idxs, 0] = traces[i, idxs]
    filled = _fill_roi_stack(roi_stack, filling, background=np.array([[0]]))[:, :, :, 0]

    return filled.mean(0)

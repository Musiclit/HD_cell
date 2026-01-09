import numpy as np

from lotr.utils import roll_columns_jit


def resample_and_shift(exp, n_bins_resampling=100):
    """Function that returns the angle-shifted bump of activity over time
    from one experiment. Tutorial in the '3. Activation profile.ipynb' notebook.
    """
    sort_idxs = np.argsort(exp.rpc_angles)

    # we will resample over the (-pi, pi) interval
    resampling_base = np.linspace(-np.pi, np.pi, n_bins_resampling)

    angle_resampled_traces = np.zeros((exp.n_pts, n_bins_resampling))
    for i in range(exp.n_pts):
        angle_resampled_traces[i, :] = np.interp(
            resampling_base,
            exp.rpc_angles[sort_idxs],
            exp.traces[i, exp.hdn_indexes[sort_idxs]],
        )

    # Find the right amount of shift over time to have the bump centered:
    # by first stretching phase to (-0.5, 0.5) interval and then
    # to (-n_rois//2, n_rois//2) interval. In this way, we will center phase 0 of
    # the network on position of angle 0
    phase_shifts_res = (exp.network_phase / (2 * np.pi)) * (n_bins_resampling - 1)

    # Then, apply shifts to traces:
    reshaped_traces = roll_columns_jit(
        angle_resampled_traces, -np.round(phase_shifts_res)
    )

    return angle_resampled_traces, reshaped_traces

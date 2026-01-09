import numpy as np
import pandas as pd
from scipy.signal import detrend, medfilt


def detrend_norm(trace, wnd=3000):
    """Normalize trace by dividing it for a super low-pass filtered
    version of it.
    """
    st = pd.Series(trace).rolling(wnd, center=True).mean().values

    # Fill nan values:
    st[: wnd // 2] = st[wnd // 2]
    st[-wnd // 2 :] = st[-wnd // 2]
    return trace / st


def preprocess_traces(traces, fn, smooth_wnd_s=5, detrend_wnd_s=800):
    """Preprocess traces by using the detrend normalization funciton
    above, smoothing them with a median filter, and zscoring them.

    Parameters
    ----------

    traces : timepoints x nrois np.array
        traces
    fn : int
        sampling frequency
    smooth_wnd_s : float
        duration of window for the smoothing (sec)
    detrend_wnd_s : float
        duration of window for the detrending (sec)


    Returns
    -------
    timepoints x nrois np.array
        Filtered traces


    """
    traces_sm = traces.copy()

    # solve problem for even-valued windows:
    wnd_pts = (
        int(fn * smooth_wnd_s)
        if (int(fn * smooth_wnd_s) % 2) == 1
        else int(fn * smooth_wnd_s) + 1
    )

    for i in range(traces_sm.shape[1]):
        traces_sm[:, i] = medfilt(traces[:, i], wnd_pts)
        if detrend_wnd_s is not None:
            traces_sm[:, i] = detrend_norm(traces_sm[:, i], wnd=int(fn * detrend_wnd_s))
    if detrend_wnd_s is None:
        traces_sm = detrend(traces_sm, axis=1)

    return (traces_sm - np.nanmean(traces_sm, 0)) / np.nanstd(traces_sm, 0)

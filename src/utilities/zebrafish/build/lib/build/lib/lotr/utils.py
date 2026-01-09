import numpy as np
import pandas as pd
from numba import njit
from scipy.interpolate import interp1d


def zscore(array):
    """Nan zscoring function."""
    return (array - np.nanmean(array)) / np.nanstd(array)


def interpolate(source_x, source_y, target_x):
    return interp1d(source_x, source_y, fill_value="extrapolate")(target_x)


def pearson_regressors(traces, regressors):
    """Gives the pearson correlation coefficient. Adapted from Vilim's functions.

    Parameters
    ----------
    traces : np.array
        (t, n_rois) array with the traces.
    regressors : np.array
        (t, n_regressors) array with the regressors.

    Returns
    -------
    np.array
        (n_regressors, n_rois) matrix of regressors.

    """
    # two versions, depending whether there is one or multiple regressors
    X = traces
    Y = regressors
    if len(Y.shape) == 1:
        numerator = np.dot(X.T, Y) - X.shape[0] * np.nanmean(X, 0) * np.nanmean(Y)
        denominator = (X.shape[0] - 1) * np.nanstd(X, 0) * np.nanstd(Y)
        result = numerator / denominator
    else:
        numerator = np.dot(X.T, Y) - X.shape[0] * np.outer(
            np.nanmean(X, 0), np.nanmean(Y, 0)
        )
        denominator = (X.shape[0] - 1) * np.outer(np.nanstd(X, 0), np.nanstd(Y, 0))
        result = (numerator / denominator).T

    return result


def linear_regression(x, y):
    """Get slope and intercept of linear regression between two vectors.

    Parameters
    ----------
    x : np.array
    y : np.array

    Returns
    -------
    (off, coef)
        Offset and coefficient of regression
    """
    x_mat = np.vstack((np.ones(len(x)), x)).T
    off, coef = np.linalg.inv(x_mat.T.dot(x_mat)).dot(x_mat.T).dot(y)
    return off, coef


def reduce_to_pi(angle):
    """Puts an angle or array of angles inside the (-pi, pi) range"""
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def get_rot_matrix(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def get_vect_angle(vect):
    return np.angle(vect[0] + 1j * vect[1])


@njit
def roll_columns_jit(matrix, shifts):
    """Shift every column of the matrix by a specified amount.

    Parameters
    ----------
    matrix : np.array
        (to_roll, n) matrix to roll over the first dimension.
    shifts : np.array
        (to_roll,) array of integer indexes specifying the shift.

    Returns
    -------
    np.array
        matrix shifted over first dimension.

    """
    rolled = np.empty_like(matrix)
    for i in range(matrix.shape[0]):
        rolled[i, :] = np.roll(matrix[i, :], int(shifts[i]))
    return rolled


def crop(traces, events, pre_int=20, post_int=30, dwn=1):
    """Apply cropping functions defined below depending on the dimensionality
    of the input (one cell or multiple cells). If input is pandas Series
    or DataFrame, it strips out the values first.

    Parameters
    ----------
    traces : np.array or pd.Series
        1D (n_timepoints,) or 2D (n_timepoints, n_cells) array, pd.Series or
        pd.DataFrame with time over index.
    events : np.array
        Events around which to crop.
    pre_int : int or float
        Interval to crop before the event, in points.
    post_int : int or float
        Interval to crop after the event, in points.
    dwn : int
        Downsampling factor, if required.

    Returns
    -------
    np.array
        (n_events, n_pts) np.array if traces is 1D or (x,y,z check) if traces is 2D.

    """
    pre_int, post_int = int(pre_int), int(post_int)
    kwargs = dict(pre_int=pre_int, post_int=post_int, dwn=dwn)
    if isinstance(traces, pd.DataFrame) or isinstance(traces, pd.Series):
        traces = traces.values
    if isinstance(events, pd.DataFrame) or isinstance(events, pd.Series):
        events = events.values.flatten().astype(np.int)
    if len(traces.shape) == 1:
        return _crop_trace(traces, events, **kwargs)
    elif len(traces.shape) == 2:
        return _crop_block(traces, events, **kwargs)
    else:
        raise TypeError("traces matrix must be at most 2D!")


@njit
def _crop_trace(trace, events, pre_int=20, post_int=30, dwn=1):
    """Crop the trace around specified events in a window given by parameters."""

    # Avoid problems with spikes at the borders:
    valid_events = (events > pre_int) & (events < len(trace) - post_int)

    mat = np.zeros((int((pre_int + post_int) / dwn), events.shape[0]))

    for i, s in enumerate(events):
        if valid_events[i]:
            cropped = trace[s - pre_int : s + post_int : dwn].copy()
            mat[: len(cropped), i] = cropped

    return mat


@njit
def _crop_block(traces_block, events, pre_int=20, post_int=30, dwn=1):
    """Crop a block of traces."""

    n_timepts = traces_block.shape[0]
    n_cells = traces_block.shape[1]
    # Avoid problems with spikes at the borders:
    valid_events = events[(events > pre_int) & (events < n_timepts - post_int)]

    mat = np.full((int((pre_int + post_int) / dwn), events.shape[0], n_cells), np.nan)

    for i, s in enumerate(events):
        if valid_events[i]:
            cropped = traces_block[s - pre_int : s + post_int : dwn, :].copy()
            mat[: len(cropped), i, :] = cropped

    return mat


@njit
def resample_matrix(x, fx, matrix):
    """Resample all row of a matrix using the np.interp function.

    Parameters
    ----------
    x : np.array
        (n_newpts) coords array over which to resample.
    fx : np.array
        (n_pts) coords array of source data
    matrix : np.array
        (n_pts, n) matrix of data to resample along first dimension.

    Returns
    -------
    np.array
        (n_newpts, n) resampled matrix.

    """
    resampled = np.full((len(x), matrix.shape[1]), np.nan)

    for i in range(matrix.shape[1]):
        resampled[:, i] = np.interp(x, fx, matrix[:, i])

    return resampled


def convolve_with_tau(array, tau_fs, n_kernel_pts=1000):
    kernel = np.exp(-np.arange(n_kernel_pts) / tau_fs)
    kernel = kernel / np.sum(kernel)

    return np.convolve(array, kernel)[: len(array)]


def map_to_range(arr, newrange, from_range=None):
    if from_range is None:
        from_range = np.nanmin(arr), np.nanmax(arr)
    arr = arr - from_range[0]
    arr = arr / (from_range[1] - from_range[0])
    return arr * [newrange[1] - newrange[0]] + newrange[0]


def nan_phase_jumps(phase_arr):
    nanned_phase = phase_arr.copy()

    # set to nan jumps in the derivative:
    nanned_phase[1:][np.abs(np.diff(nanned_phase)) > np.pi] = np.nan

    return nanned_phase


@njit
def circular_corr(th, ph):
    """Circular correlation coefficient between arrays.
    Definition after Fisher&Lee, Biometrika 1983.

    Parameters
    ----------
    th : np.array
        1D array.
    ph : np.array
        Second array, mush have same length of th.

    Returns
    -------
    float
        Fisher-Lee circular correlation coefficient.

    """
    n = len(th)
    num = sum(
        [
            sum([np.sin(th[i] - th[j]) * np.sin(ph[i] - ph[j]) for i in range(j)])
            for j in range(n)
        ]
    )
    den1 = sum(
        [sum([np.sin(th[i] - th[j]) ** 2 for i in range(j)]) for j in range(n)]
    ) ** (1 / 2)
    den2 = sum(
        [sum([np.sin(ph[i] - ph[j]) ** 2 for i in range(j)]) for j in range(n)]
    ) ** (1 / 2)
    return num / (den1 * den2)

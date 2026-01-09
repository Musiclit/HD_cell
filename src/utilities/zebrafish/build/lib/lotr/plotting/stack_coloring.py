"""Coloring code adapted from Vilim's notebook_utilities.stack_coloring
and motions.color.
"""

import numpy as np
from numba import njit

from lotr.plotting.color_utils import (
    _get_categorical_colors,
    get_continuous_colors,
)


@njit
def _fill_roi_stack(
    rois,
    roi_colors,
    background=np.array(
        [
            0,
        ]
        * 4
    ),
):
    coloured = np.zeros(rois.shape + (roi_colors.shape[1],), dtype=roi_colors.dtype)
    for i in range(rois.shape[0]):
        for j in range(rois.shape[1]):
            for k in range(rois.shape[2]):
                if rois[i, j, k] > -1:
                    coloured[i, j, k] = roi_colors[rois[i, j, k], :]
                else:
                    coloured[i, j, k] = background[: roi_colors.shape[1]]
    return coloured


def _color_anatomy_rgb(rois, roi_colors, anatomy, alpha=0.9, invert_anatomy=True):
    """Colours a new stack of zeros with ROI colors

    :param rois: the ROI stack
    :param roi_colors: colors for each ROI
    :param anatomy: anatomy stack
    :param alpha: alpha of the ROI coloring
    :return:
    """
    colored_rois = _fill_roi_stack(
        rois,
        roi_colors,
        background=np.array(
            [
                0,
            ]
            * 4
        ),
    )
    overimposed = np.concatenate(
        [
            anatomy[:, :, :, np.newaxis],
        ]
        * 3,
        3,
    )

    if invert_anatomy:
        overimposed = 255 - overimposed

    overimposed[rois > -1] = overimposed[rois > -1] * (1 - alpha)
    overimposed = overimposed + colored_rois * alpha

    return overimposed.astype(np.uint8)


def _normalize_to_255(stack, hist_percentiles=(5, 99)):
    hist_boundaries = [np.percentile(stack, p) for p in hist_percentiles]
    stack = stack - hist_boundaries[0]
    stack = (stack / hist_boundaries[1]) * 255
    stack[stack < 0] = 0
    stack[stack > 255] = 255

    return stack


def color_stack(
    rois,
    variable,
    color_scheme=None,
    anatomy=None,
    categorical=None,
    background="transparent",
    vlims=None,
    lum=60,
    sat=60,
    hshift=0,
    alpha=0.9,
    invert_anatomy=True,
    hist_percentiles=None,
):
    """
    Parameters
    ----------
    rois : 3D np.array
        stack of ROIs (fimpy convention: -1 in empty voxels)

    variable : 1D np.array
        An array of length==n_rois based on which colors stack will be colored.
        If it contains integers, we'll assume the variable is categorical
        unless specified otherwise with the `categorical` parameter.
        ROIs can be excluded from the coloring by setting their value to -1
        (for categorical variables) or to np.nan (for non categorical variables).

    categorical : bool (optional)
        If true, variable will be treated as categorical (normally inferred
        from `variable`).

    color_scheme : str or dict (optional)
        Depending on the variable:
            - categorical: dictionary with the mapping [i] = np.array([r, g, b]).
                By default, a set of constant luminance and saturation colors will be
                generated.
            - non categorical: string specifying a matplotlib color palette.
                By default, viridis will be used.

    anatomy : 3D numpy array (optional)
        If specified, ROIs will be overimposed on it, with tht specified the `alpha`

    background : str or np.array (optional)
        Filling for the empty voxels. Default options are "w", "k" and "transparent".

    vlims : tuple or list (optional)
        Limits for the colormap (used only for non-categorical variable).

    lum : int (optional)
        Luminance for the generation of the categories colors, from 0 to 100
        (used only for categorical variable).

    sat : int (optional)
        Saturation for the generation of the categories colors, from 0 to 100
        (used only for categorical variable).

    hshift : int (optional)
        Hue shift for the generation of the categories colors, from 0 to 360
        (used only for categorical variable).

    alpha : float (optional)
        Alpha value for the overlapping of anatomy and ROIs, from 0 to 1. Default 0.9.

    invert_anatomy : bool (optional)
        If True, anatomy will be inverted (black signal on white background).
        Default True.

    hist_percentiles : tuple (optional)
        Range used for the normalization of the anatomy histogram, if anatomy is not
        already scaled. Default (5, 99)

    Returns
    -------

    """

    BACKGROUNDS = dict(
        k=np.array([0, 0, 0, 255]),
        transparent=np.array(
            [
                0,
            ]
            * 4
        ),
        w=np.array(
            [
                255,
            ]
            * 4
        ),
    )

    # We infer if the variable is categorical or not:
    if categorical is None:
        categorical = np.issubdtype(np.array(variable).dtype, np.integer)

    if categorical:
        # Esclude from the stack rois with nan variable, using the filling function
        # TODO refactor together this and the non categorical condition
        if (variable < 0).any():
            print("excluding")
            nan_filling = np.arange(rois.max() + 1)[:, np.newaxis]
            nan_filling[np.argwhere(variable < 0)[:, 0], :] = -1
            rois = _fill_roi_stack(rois, nan_filling, background=np.array([[-1]]))[
                :, :, :, 0
            ]

        # get roi colors:
        roi_colors = _get_categorical_colors(
            variable, color_scheme=color_scheme, lum=lum, sat=sat, hshift=hshift
        )

    else:
        # Esclude from the stack rois with nan variable, using the filling function:
        if np.isnan(variable).any():
            nan_filling = np.arange(rois.max() + 1)[:, np.newaxis]
            nan_filling[np.argwhere(np.isnan(variable))[:, 0], :] = -1
            rois = _fill_roi_stack(rois, nan_filling, background=np.array([[-1]]))[
                :, :, :, 0
            ]
        # get roi colors:
        roi_colors = get_continuous_colors(
            variable, color_scheme=color_scheme, vlims=vlims
        )

    if isinstance(background, str):
        background = BACKGROUNDS[background]

    if anatomy is None:
        return _fill_roi_stack(rois, roi_colors, background=background)
    else:
        # If required, normalize the anatomy stack:
        if hist_percentiles is not None or anatomy.max() > 255 or anatomy.min() < 0:
            if hist_percentiles is None:
                hist_percentiles = (5, 99)

            anatomy = _normalize_to_255(anatomy, hist_percentiles=hist_percentiles)

        return _color_anatomy_rgb(
            rois,
            roi_colors[:, :3],
            anatomy.astype(np.uint8),
            alpha=alpha,
            invert_anatomy=invert_anatomy,
        )


def color_zproject(stack, mode="overlay"):
    if mode == "overlay":
        projected = np.zeros(stack.shape[1:], dtype=np.uint8)
        for plane in range(stack.shape[0]):
            selection = stack[plane, ...] > 0
            projected[selection] = stack[plane, ...][selection]

    elif mode == "transparency":
        projected = np.zeros(stack.shape[1:])
        for plane in range(stack.shape[0]):
            selection = stack[plane, ...] > 0
            projected[selection] += stack[plane, ...][selection]
        scale = projected[..., -1].max() / 255
        projected /= scale  # projected[..., -1] /= scale
        projected = projected.astype(np.uint8)
    else:
        raise ValueError("'mode' should be either overlay or transparency!")

    return projected

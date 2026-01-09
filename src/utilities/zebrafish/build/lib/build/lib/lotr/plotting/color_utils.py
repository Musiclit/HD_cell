"""Coloring code adapted from Vilim's notebook_utilities.stack_coloring
and motions.color.
"""

import colorspacious
import matplotlib
import numpy as np
from matplotlib.colors import to_rgb


def shift_lum(c, s=0.2):
    c = to_rgb(c)
    if s <= 1:
        s = s * 100
    jch_c = colorspacious.cspace_convert(c, "sRGB1", "JCh")
    jch_c[0] += s
    return np.clip(colorspacious.cspace_convert(jch_c, "JCh", "sRGB1"), 0, 1)


def dark_col(col, val=0.2):
    if type(col) == str:
        col = [float(col) for _ in range(3)]
    return [max(0, c - val) for c in col]


def _jch_to_rgb255(x):
    output = np.clip(colorspacious.cspace_convert(x, "JCh", "sRGB1"), 0, 1)
    return (output * 255).astype(np.uint8)


def get_n_colors(n_cols, lum=60, sat=60, hshift=0):
    return _jch_to_rgb255(
        np.stack(
            [
                np.full(n_cols, lum),
                np.full(n_cols, sat),
                (-np.arange(0, 360, 360 / n_cols) + hshift),
            ],
            1,
        )
    )


def _get_categorical_colors(variable, color_scheme=None, lum=60, sat=60, hshift=0):
    if color_scheme is None:
        unique_vals = np.unique(variable[variable >= 0])
        colors = get_n_colors(len(unique_vals), lum=lum, sat=sat, hshift=hshift)
        color_scheme = {v: colors[i] for i, v in enumerate(unique_vals)}

    color_scheme[-1] = np.array([0, 0, 0])

    roi_colors = np.array([color_scheme[v] for v in variable])

    return np.concatenate([roi_colors, np.full((len(variable), 1), 255)], 1)


def get_continuous_colors(variable, color_scheme=None, vlims=None):
    if color_scheme is None:
        color_scheme = "viridis"

    if vlims is None:
        vlims = np.nanmin(variable), np.nanmax(variable)

    # cmap function:
    cmap_fun = (
        color_scheme if callable(color_scheme) else matplotlib.cm.get_cmap(color_scheme)
    )

    # normalization function:
    norm = matplotlib.colors.Normalize(vmin=vlims[0], vmax=vlims[1])

    return (np.array([cmap_fun(norm(v)) for v in variable]) * 255).astype(np.uint8)

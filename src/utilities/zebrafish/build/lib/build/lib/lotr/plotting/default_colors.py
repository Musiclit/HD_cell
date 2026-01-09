import numpy as np
from matplotlib.colors import ListedColormap

from lotr.plotting.color_utils import get_continuous_colors, get_n_colors

COLS = dict(
    sides=dict(lf=(0.0, 0.623, 0.886), rt=(0.835, 0.364, 0.282), fw=(0.5,) * 3),
    dff_img="Greens",
    dff_plot="gray",
    time="viridis",
    beh=(0.4,) * 3,
    ring=(0.847, 0.102, 0.376),
    shuf=".7",
    phase=ListedColormap(get_n_colors(1000, lum=45, sat=70, hshift=90) / 255),
    phase_light=ListedColormap(get_n_colors(1000, lum=60, sat=45, hshift=90) / 255),
    dff_opp="PiYG",
    ph_plot=[0.08, 0.58, 0.16],
    th_plot=[0.55, 0.49, 0.0],
    stim_conditions=dict(
        darkness=".5",
        natural_motion=[0.71, 0.54, 0.15],
        directional_motion=[0.62, 0.29, 0.63],
        closed_loop={
            0.5: [0.57, 0.85, 0.47],
            1: [0.33, 0.6, 0.25],
            2: [0.13, 0.43, 0.07],
            -1: [0.84, 0.35, 0.3],
        },
    ),
    fish_cols=[
        "#bf3f76",
        "#577b34",
        "#9d6620",
        "#c54238",
        "#925b84",
        "#546dae",
        "#976a61",
        "#397b74",
        "#5981a3",
    ]
    * 5,
    qualitative=[
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
    ],
)


def get_default_phase_col(phase):
    (col,) = get_continuous_colors([phase], COLS["phase"], vlims=(-np.pi, np.pi)) / 255
    return col

from pathlib import Path

import flammkuchen as fl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from lotr import plotting as pltltr
from lotr.file_utils import get_figures_location


def rot_coords(contour, alpha):
    return (
        np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        @ contour.T
    ).T


data_path = Path(
    "/Volumes/Shared/experiments/virtual_gradients_experiments/freely_swim_Ruben/data_for_movie.h5"
)
data = fl.load(data_path)

sm = 500  # smoothing factor
s = 22000  # start in pts
d = 20000  # duration in pts

coords = np.array(
    [data["f0_x"].rolling(sm).mean().values, data["f0_y"].rolling(sm).mean().values]
).T

diff = coords[1:, :] - coords[:-1, :]
angle = np.arctan2(diff[:, 0], diff[:, 1])

coords = coords[s : s + d, :]
angle = angle[s : s + d]

size = 10
arrow = np.array([(-0.5, -0.5), (0.5, -0.5), (0.0, 1.25), (-0.5, -0.5)]) * 20
space_coef = 10
xlim = coords[:, 0].min() - size * space_coef, coords[:, 0].max() + size * space_coef
ylim = coords[:, 1].min() - size * space_coef, coords[:, 1].max() + size * space_coef

c = pltltr.shift_lum(pltltr.COLS["ph_plot"], -0.25)


def network_phase_animation(dest=None, frames=None):
    if dest is None:
        dest = (
            get_figures_location()
            / f"network_phase_bump_traces_t{frames[0]}-{frames[-1]}+{frames[1]}.mp4"
        )

    frame = 0
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(6, 2),
        gridspec_kw=dict(
            wspace=0, left=0, right=1, top=1, bottom=0, width_ratios=[2, 1]
        ),
    )

    for a in ax:
        a.axis("off")
    arrow_pts = rot_coords(arrow, angle[frame]) + coords[frame, :]
    (fish_plot,) = ax[0].fill(arrow_pts[:, 0], arrow_pts[:, 1], lw=0, fc=c)
    # ax[0].plot(coords[:, 0], coords[:, 1], lw=1)

    cell_angles = np.arange(-np.pi, np.pi, 2 * np.pi / 8)

    l_w = 1.7
    distances = (
        np.maximum(
            (
                np.cos(angle[frame]) * np.cos(cell_angles)
                + np.sin(angle[frame]) * np.sin(cell_angles)
            ),
            0,
        )
        ** 4
    )
    cells_plot = ax[1].scatter(
        np.sin(cell_angles),
        np.cos(cell_angles),
        c=distances,
        s=300,
        cmap="gray_r",
        lw=0.5,
        ec="k",
        vmin=0,
        vmax=1.2,
    )
    for a in cell_angles:
        plt.arrow(
            np.sin(a),
            np.cos(a),
            np.sin(a) / 2.5,
            np.cos(a) / 2.5,
            width=0.05,
            zorder=-100,
            lw=0,
            fc=c,
        )

    actors = (cells_plot, fish_plot)

    def init():
        # For some mysterious reasons, some of those specifications need to happen
        # in the init. Therefore, we put here a whole bunch of them:
        ax[1].set(xlim=(-l_w, l_w), ylim=(-l_w, l_w))
        ax[0].set(xlim=xlim, ylim=ylim)

        pltltr.add_cbar(
            cells_plot,
            ax[1],
            (0.0, 0.1, 0.04, 0.2),
            ticks=[],
            ticklabels=[],
            titlesize=8,
            label="Firing rate",
            orientation="vertical",
        )

        return actors

    def update(frame):
        distances = (
            np.maximum(
                (
                    np.cos(angle[frame]) * np.cos(cell_angles)
                    + np.sin(angle[frame]) * np.sin(cell_angles)
                ),
                0,
            )
            ** 4
        )
        cells_plot.set_array(distances)

        arrow_pts = rot_coords(arrow, angle[frame]) + coords[frame, :]
        new_coords = np.array([arrow_pts[:, 0], arrow_pts[:, 1]]).T

        fish_plot.set_xy(new_coords)

        return actors

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=50,
        init_func=init,
        blit=True,
    )
    ani.save(dest / "running_fish.mp4", dpi=300)

    return fig


if __name__ == "__main__":
    network_phase_animation(frames=list(range(1000, d, 8)))

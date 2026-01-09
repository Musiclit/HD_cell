import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from lotr import A_FISH, LotrExperiment
from lotr.file_utils import get_figures_location
from lotr.pca import pca_and_phase
from lotr.plotting import COLS, add_cbar, add_scalebar, get_default_phase_col
from lotr.rpca_calculation import get_zero_mean_weights
from lotr.utils import get_vect_angle


def network_phase_animation(dest=None, frames=None):
    if frames is None:
        frames = list(range(0, 4000, 5))
    if dest is None:
        dest = (
            get_figures_location()
            / f"network_phase_t{frames[0]}-{frames[-1]}+{frames[1]}.mp4"
        )
    # Data preparation
    # ----------------
    # Compute quantities to plot:
    exp = LotrExperiment(A_FISH)
    pca_scores_t, roi_pc_angles, _, circle_params = pca_and_phase(
        exp.traces[:, exp.hdn_indexes].T
    )
    pca_scores_t[:, :2] = pca_scores_t[:, :2] - circle_params[:2]

    norm_activity = get_zero_mean_weights(exp.traces[:, exp.hdn_indexes].T).T
    avg_vects = np.einsum("ij,ik->jk", norm_activity.T, pca_scores_t[:, :2])

    network_phase = get_vect_angle(avg_vects.T)

    # Plot parameters
    # ---------------
    scale_arr = 5000
    scale_mn = 60
    f_lim = 0.01
    w_lw, netw_lw = 0.3, 1.5
    w_c = (0.7,) * 3

    # Figure
    # ------
    # Prepare figure globals:
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    # Scatter plot:
    activity_sc = ax.scatter(pca_scores_t[:, 0], pca_scores_t[:, 1], lw=0)

    # Timestamp:
    tx = ax.text(-100, 100, "t = 0 s", fontsize=8)

    # Individual weights:
    weight_actors = []
    for i in range(len(exp.hdn_indexes)):
        # comma is essential!
        (weights_plot,) = ax.plot(
            [], [], c=w_c, lw=w_lw, label="ROI vect." if i == 0 else "_nolegend_"
        )
        weight_actors.append(weights_plot)

    add_scalebar(ax, xlabel="rPC1", ylabel="rPC2", xlen=30, ylen=30)

    # Network average:
    (network_phase_plot,) = ax.plot(
        [],
        [],
        c=w_c,
        lw=netw_lw,
        label="netw. phase",
        solid_capstyle="round",
    )

    # Legend:
    leg = ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(0.95, 1.1, 0.2, 0.04))
    _, leg_line = leg.get_lines()

    # put together all actors:
    actors = (activity_sc, tx, network_phase_plot, *weight_actors, leg_line)

    def init():
        # For some mysterious reasons, some of those specifications need to happen
        # in the init. Therefore, we put here a whole bunch of them:
        ax.axis("equal")
        activity_sc.set_clim(-f_lim, f_lim)
        activity_sc.set_cmap(COLS["dff_opp"])

        add_cbar(
            activity_sc,
            ax,
            (0.85, 0.02, 0.2, 0.04),
            ticks=(-f_lim + 0.001, f_lim - 0.001),
            ticklabels=("-", "+"),
            titlesize=8,
            title="ΔF weight",
            orientation="horizontal",
        )

        return actors

    def update(frame):
        tx.set_text(f"{frame / exp.fs:3.0f} s")

        activity_sc.set_array(norm_activity[frame])
        network_phase_plot.set_data(
            [0, np.cos(network_phase[frame]) * scale_mn],
            [0, np.sin(network_phase[frame]) * scale_mn],
        )

        new_col = get_default_phase_col(network_phase[frame])
        for line in network_phase_plot, leg_line:
            line.set_color(new_col)

        for r, plot in enumerate(weight_actors):
            plot.set_data(
                [0, np.cos(roi_pc_angles[r]) * norm_activity[frame, r] * scale_arr],
                [0, np.sin(roi_pc_angles[r]) * norm_activity[frame, r] * scale_arr],
            )

        return actors

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=50,
        init_func=init,
        blit=True,
    )
    ani.save(dest, dpi=300)

    return fig


if __name__ == "__main__":
    dest = get_figures_location() / "gifs"
    dest.mkdir(exist_ok=True)
    network_phase_animation(frames=list(range(0, 4000, 5)))

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from lotr import A_FISH, LotrExperiment
from lotr.behavior import get_fictive_heading
from lotr.file_utils import get_figures_location
from lotr.plotting import COLS, add_cbar, add_scalebar, get_default_phase_col
from lotr.rpca_calculation import get_zero_mean_weights
from lotr.utils import convolve_with_tau, get_rot_matrix, linear_regression


def network_phase_animation(dest):
    # Data preparation
    # ----------------
    # Compute quantities to plot:
    exp = LotrExperiment(A_FISH)
    rpca_scores = exp.rpc_scores
    network_phase = exp.network_phase
    fict_heading = get_fictive_heading(exp.n_pts, exp.bouts_df)
    convolved_head = convolve_with_tau(fict_heading, 5)

    t_slice = slice(5180, 6540)
    o, c = linear_regression(network_phase[t_slice], convolved_head[t_slice])
    convolved_head = (convolved_head - o) / c

    norm_activity = get_zero_mean_weights(exp.traces[:, exp.hdn_indexes].T).T
    # avg_vects = np.einsum("ij,ik->jk", norm_activity.T, rpca_scores[:, :2])

    # network_phase = get_vect_angle(avg_vects.T)

    # Plot parameters
    # ---------------
    # scale_arr = 5000
    scale_mn = 60
    f_lim = 0.01
    # w_lw, netw_lw = 0.3, 1.5
    netw_lw = 1.5
    w_c = (0.7,) * 3

    # Figure
    # ------
    # Prepare figure globals:
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    # Scatter plot:
    activity_sc = ax.scatter(-rpca_scores[:, 1], rpca_scores[:, 0], lw=0)

    # Timestamp:
    tx = ax.text(-100, 100, "t = 0 s", fontsize=8)

    # Individual weights:
    # weight_actors = []
    # for i in range(len(exp.hdn_indexes)):
    #    # comma is essential!
    #    (weights_plot,) = ax.plot(
    #        [], [], c=w_c, lw=w_lw, label="ROI vect." if i == 0 else "_nolegend_"
    #    )
    #    weight_actors.append(weights_plot)

    # Network average:
    (network_phase_plot,) = ax.plot(
        [],
        [],
        c=w_c,
        lw=netw_lw,
        label="netw. phase",
        solid_capstyle="round",
    )

    (fish_heading_plot,) = ax.plot(
        [],
        [],
        c=".6",
        lw=netw_lw,
        label="fish front",
        solid_capstyle="round",
    )

    add_scalebar(ax, xlabel="PC1", ylabel="PC2", xlen=30, ylen=30)
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

    # Legend:
    leg = ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(0.95, 1.1, 0.2, 0.04))
    leg_line, _ = leg.get_lines()

    # put together all actors:
    actors = (activity_sc, tx, network_phase_plot, leg_line)  # *weight_actors)

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
        rot_mat = get_rot_matrix(-convolved_head[frame] + np.pi)
        # activity_sc.set_offsets
        network_phase_coords = (
            np.array(
                [[0, -np.sin(network_phase[frame])], [0, -np.cos(network_phase[frame])]]
            )
            * scale_mn
        )

        fish_head_coords = np.array([[0, 1], [0, 0]]) * scale_mn

        network_phase_coords = rot_mat @ network_phase_coords
        fish_head_coords = rot_mat @ fish_head_coords

        network_phase_plot.set_data(
            network_phase_coords[0, :], network_phase_coords[1, :]
        )
        fish_heading_plot.set_data(fish_head_coords[0, :], fish_head_coords[1, :])

        rot_rpc_coords = (rot_mat @ rpca_scores.T).T

        activity_sc.set_offsets(
            np.array([-rot_rpc_coords[:, 1], rot_rpc_coords[:, 0]]).T
        )

        activity_sc.set_array(norm_activity[frame])

        new_col = get_default_phase_col(network_phase[frame])
        for line in network_phase_plot, leg_line:
            line.set_color(new_col)

        # for r, plot in enumerate(weight_actors):
        #    plot.set_data(
        #        [0, np.cos(roi_pc_angles[r]) * norm_activity[frame, r] * scale_arr],
        #        [0, np.sin(roi_pc_angles[r]) * norm_activity[frame, r] * scale_arr],
        #    )

        tx.set_text(f"{(frame - t_slice.start) / exp.fs:3.0f} s")
        return actors

    ani = FuncAnimation(
        fig,
        update,
        frames=list(range(t_slice.start, t_slice.stop, 1)),
        interval=50,
        init_func=init,
        blit=True,
    )
    ani.save(dest, dpi=300)

    return fig


if __name__ == "__main__":
    network_phase_animation(get_figures_location() / "rotating_network_gif.mp4")

import flammkuchen as fl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from lotr import A_FISH, LotrExperiment
from lotr.plotting import COLS
from lotr.plotting.gifs_gen.gif_utils import make_proj

sns.set(style="ticks", palette="deep")
cols = sns.color_palette()

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Libertinus Sans"]


FILENAMES = ["raw_all_movie.mp4", "raw_selected_movie_new.mp4"]
FN = 5
master_path = A_FISH.parent
exp = LotrExperiment(A_FISH)

arr = np.zeros(exp.n_rois, dtype=np.int) - 1
arr[exp.hdn_indexes] = 1

rois = exp.rois_stack
ring_rois = exp.color_rois_by(arr, color_scheme={1: COLS["ring"]}, categorical=True)
proj = ring_rois.mean(0)

# 13 excluded
for path in tqdm(list(master_path.glob("*_f*"))):
    anatomy_stack = fl.load(path / "data_from_suite2p_unfiltered.h5", "/anatomy_stack")
    rois_stack = fl.load(path / "data_from_suite2p_unfiltered.h5", "/rois_stack")
    coords = fl.load(path / "data_from_suite2p_unfiltered.h5", "/coords")
    bouts_df = fl.load(path / "bouts_df.h5")
    traces = fl.load(path / "filtered_traces.h5", "/detr")
    n_pts, n_cells = traces.shape

    for i, selected in enumerate([np.arange(n_cells), fl.load(path / "selected.h5")]):
        (n_sel,) = selected.shape

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        col = fig.add_axes((0.9, 0.4, 0.015, 0.15))

        im = ax.imshow(
            make_proj(rois_stack, traces, selected, 0).T,
            origin="lower",
            cmap="gray",
            vmin=-0.5,
            vmax=0.3,
        )
        ax.contour(proj[:, :, 0] > 0, levels=1, colors=[COLS["ring"]], linewidths=1)

        tx = ax.text(70, 10, f"{0 / 5:3.0f} s", ha="right", c="w")
        actors = (im, tx)

        cbar = plt.colorbar(im, cax=col, label="norm. dF")
        cbar.set_ticks([])

        def init():
            im.set_clim(-0.2, 0.5)
            im.set_zorder(-1000)
            cbar.outline.set_visible(False)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("left - right")
            ax.set_ylabel("posterior - anterior")
            [
                ax.axes.spines[s].set_visible(False)
                for s in ["left", "right", "top", "bottom"]
            ]
            # sns.despine(trim=True)
            return actors

        def update(frame):
            tx.set_text(f"{frame / 5:3.0f} s")
            im.set_data(make_proj(rois_stack, traces, selected, frame).T)
            return actors

        ani = FuncAnimation(
            fig,
            update,
            frames=np.arange(0, n_pts, 10),
            interval=50,
            init_func=init,
            blit=True,
        )
        # plt.show()
        ani.save(path / FILENAMES[i])

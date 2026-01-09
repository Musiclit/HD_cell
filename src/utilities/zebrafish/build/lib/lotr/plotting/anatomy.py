import numpy as np
from bg_atlasapi import BrainGlobeAtlas
from matplotlib import pyplot as plt

from lotr.plotting.general import projection_contours
from lotr.plotting.standard_addons import add_anatomy_scalebar


class AtlasPlotter:
    def __init__(
        self,
        atlas=None,
        structures=None,
        alpha_dict=None,
        projections=None,
        mask_slices=None,
        fontsize=14,
        bounds_dict=None,
        fill_kw=None,
        smooth_wnd=15,
    ):
        if atlas is None:
            atlas = BrainGlobeAtlas("ipn_zfish_0.5um")
        self.atlas = atlas
        self.space = self.atlas.space

        self.mask_slices = {
            k: [
                slice(None, None),
            ]
            * 3
            for k in self.space.sections
        }
        for i, k in enumerate(self.space.sections):
            if mask_slices is not None and k in mask_slices.keys():
                self.mask_slices[k][i] = mask_slices[k]
            self.mask_slices[k] = tuple(self.mask_slices[k])

        if structures is None:
            structures = [k for k in self.atlas.structures.keys()]
        self.visible_structures = structures

        if alpha_dict is None:
            alpha_dict = {k: 0.2 for k in self.visible_structures}
        self.alpha_dict = alpha_dict

        # Compute contours only once:
        self.contours_dict = dict()
        for s in self.visible_structures:
            projection_dict = [
                self.project_mask(s, p, smooth_wnd=smooth_wnd)
                for p in self.space.sections
            ]
            self.contours_dict[s] = projection_dict

        self.projections = self.space.sections if projections is None else projections

        self.fill_kw = dict(
            alpha=1, facecolor="none", fill=False, linewidth=0.5, edgecolor=".03"
        )
        if fill_kw is not None:
            self.fill_kw.update(fill_kw)

        # TODO refactor
        self.font_size = fontsize

        self.custom_switch = None

        self.bounds_dict = {k: None for k in self.space.sections}
        if bounds_dict is not None:
            self.bounds_dict.update(bounds_dict)

    def project_mask(self, structure, projection, smooth_wnd=7):
        # Support both semantic and index indication of the axis
        if type(projection) is str:
            proj_i = self.space.sections.index(projection)
            proj_lab = projection
        elif projection is int:
            proj_i = projection
            proj_lab = self.space.sections[projection]
        else:
            raise ValueError("Projection should be str or int")

        mask = self.atlas.get_structure_mask(structure)[self.mask_slices[proj_lab]]

        contours = projection_contours(mask.max(proj_i), smooth_wnd=smooth_wnd, thr=0.5)

        return [c * self.space.resolution[0] for c in contours]

    def get_switch(self, projection):
        if self.custom_switch is None:
            return 0 if projection == "sagittal" else 1
        else:
            return self.custom_switch[projection]

    def get_bounds(self, projection, edge=115):
        if self.bounds_dict[projection] is None:
            projection_idx = self.space.sections.index(projection)
            swtch = self.get_switch(projection)

            bounds = list()

            pair = self.space.index_pairs[projection_idx]
            for p in pair:
                d = (self.atlas.shape_um[p] - edge) // 2
                bounds.append((d, d + edge))

            return bounds[1 - swtch], bounds[swtch]
        else:
            return self.bounds_dict[projection]

    def get_labels(self, projection):
        idx = self.space.sections.index(projection)
        return [["dors.", "right"], ["rostr.", "right"], ["dors.", "caud."]][idx]

    def plot_on_axis(self, ax, projection, title=True, labels=True, edge=115):
        """Prepare background silhouette on a specific axis."""
        projection_idx = self.space.sections.index(projection)
        bs = self.get_bounds(projection, edge=edge)
        swtch = self.get_switch(projection)

        for structure in self.visible_structures:
            for cs in self.contours_dict[structure][projection_idx]:
                ax.fill(cs[:, swtch], cs[:, 1 - swtch], **self.fill_kw)

        if title:
            ax.set_title(f"{projection} view")

        if labels:
            s = 10
            s_px = int(s / self.space.resolution[0])

            o = 10
            lw = 0.5
            col = (0.3,) * 3
            position = (bs[1][0] + o * 1.5, bs[0][1] - o)
            add_anatomy_scalebar(
                ax=ax,
                length=s_px,
                pos=position,
                lw=lw,
                c=col,
                plane=projection,
                cartesian=True,
                switchlabels=1 - swtch,
            )
        else:
            ax.axis("equal")
            ax.axis("off")

            """
            ax.plot([bs[1][0] + o, bs[1][0] + s_px + o],
                    [bs[0][1] - o, bs[0][1] - o], linewidth=lw, c=col)
            ax.plot([bs[1][0] + o, bs[1][0] + o],
                    [bs[0][1] - o, bs[0][1] - s_px - o], linewidth=lw, c=col)
            ax.text(bs[1][0] + o, bs[0][1] - s_px - o * 2, labels_txt[0],
                    ha="center", va="bottom", fontsize=self.font_size, color=col)
            ax.text(bs[1][0] + s_px + o * 2, bs[0][1] - o, labels_txt[1],
                    ha="left", va="center", fontsize=self.font_size, color=col)
            """

        ax.set_xlim((bs[1][0], bs[1][1]))
        ax.set_ylim((bs[0][1], bs[0][0]))
        # ax.set_aspect("equal")
        # despine(ax, "all")

    def plot_neuron_projection(
        self,
        ax,
        neuron,
        projection,
        select="all",
        soma_s=30,
        mirrored=False,
        slice_sel=None,
        **kwargs,
    ):
        """Plot a neuronal projection an axes."""

        default_kw = dict(color=".5", alpha=1)
        default_kw.update(kwargs)

        idx = self.space.plane_normals[projection].index(1)
        swtch = self.get_switch(projection)

        cs = neuron.generate_plotlines_from_skeleton(select=select)
        if slice_sel is not None:
            sel = (cs[:, idx] > slice_sel[0]) & (cs[:, idx] < slice_sel[1])
            cs[~sel, :] = np.nan
        cs = np.delete(cs, idx, axis=1)[:, :]

        (l,) = ax.plot(cs[:, swtch], cs[:, 1 - swtch], **kwargs)

        if neuron.soma_idx is not None and soma_s > 0:
            cs = neuron.coords_ipn[neuron.soma_idx : neuron.soma_idx + 1, :]
            cs = np.delete(cs, idx, axis=1)[:, :]
            ax.scatter(
                cs[:, swtch],
                cs[:, 1 - swtch],
                s=soma_s,
                color=l.get_color(),
                alpha=kwargs.get("alpha", 1),
                linewidth=0,
            )

    def ax_scatterplot(self, ax, projection, coords, **kwargs):
        idx = self.space.plane_normals[projection].index(1)
        swtch = self.get_switch(projection)
        coords = np.delete(coords, idx, axis=1)[:, :]

        ax.scatter(coords[:, swtch], coords[:, 1 - swtch], **kwargs)

    def axs_scatterplot(self, axs, coords, **kwargs):
        for i, projection in enumerate(self.projections):
            self.ax_scatterplot(axs[i], projection, coords, **kwargs)

    def plot_neurons(self, axs, neurons, select="all", **kwargs):
        """Plot neurons projections over a triplet of axs"""
        if type(neurons) is not list:
            neurons = [neurons]

        for neuron in neurons:
            for i, projection in enumerate(self.projections):
                self.plot_neuron_projection(
                    axs[i], neuron, projection, select=select, **kwargs
                )

    def generate_projection_plots(
        self,
        axs=None,
        figsize=(9, 3),
        labels=False,
        title=False,
        tight_layout=False,
        edge=120,
    ):
        """Plot the outline of the IPN structures."""

        if axs is None:
            f, axs = plt.subplots(1, 3, figsize=figsize)
        else:
            f = plt.gcf()
        for i, projection in enumerate(self.projections):
            self.plot_on_axis(axs[i], projection, edge=edge, labels=labels, title=title)

        if tight_layout:
            plt.tight_layout()

        return f, axs

import numpy as np
import pandas as pd

from lotr.em.tracing_tree import (
    _edges_selection,
    _find_all_connected,
    _find_axon_link,
    _plotlines_from_skeleton,
    find_bifurcations,
)

from .skeleton_mesh import make_cylinder_tree, make_full_neuron
from .transformations import em2ipnref, em2mpinref


def _find_comment_idx(pts_df, key):
    clean_comments = pts_df["comment"].fillna("")

    try:
        idx = clean_comments[clean_comments.str.contains(key, case=False)].index[0]
        return idx
    except IndexError:
        return


MIDLINES = dict(ipn=109, mpin=284)


class EmNeuron:
    """Class to manipulate tracing data from EM."""

    def __init__(self, xml_element):
        # This heuristics might have to change if we load from other sources.

        # This might have to change if naming system in master file is changed:
        try:
            self.id = xml_element.attrib["comment"][:4]
            self.comments = xml_element.attrib["comment"]
            self.include = "[??]" not in self.comments
        except KeyError:
            self.id = ""

        # Ugly but necessary with XML format:
        pts_df = pd.DataFrame([e.attrib for e in list(list(xml_element)[0])])
        edges_df = pd.DataFrame([e.attrib for e in list(list(xml_element)[1])])

        # Fix data type from string:
        pts_df.loc[:, ["id", "x", "y", "z"]] = pts_df.loc[
            :, ["id", "x", "y", "z"]
        ].astype(np.int)

        self.points_df = pts_df

        # Find soma and axons ids, if available:
        self.ax_start_idx = _find_comment_idx(pts_df, "axon")

        # Read soma:
        self.soma_idx = _find_comment_idx(pts_df, "soma")

        # Replace node ids with indexes in the edges dataframe:
        edges_df = edges_df.astype(np.int)
        id_table = pd.Series(self.points_df.index, index=self.points_df.id)
        for k in edges_df.columns:
            edges_df[k] = edges_df[k].map(id_table)
        self.edges = edges_df.values

        # TODO remove this hardcoding as cringe as fuck for fixing neuron:
        if self.id == "p085":
            self.edges = np.concatenate([self.edges, [[1157, 561]]])

        self._axon_idxs = []
        self._dendr_idxs = []
        self._coords_mpin = None
        self._coords_ipn = None
        self.coords_em = self.points_df[["x", "y", "z"]].values.astype(np.float)

        self.edges_dict = dict(
            all=self.edges, dendrites=self.dendrites_edges, axon=self.axon_edges
        )

        self.mirror = False

    @property
    def side(self):
        if self.coords_ipn[self.soma_idx, :] > MIDLINES["ipn"]:
            return "l"
        else:
            return "r"

    @property
    def is_axon(self):
        return self.soma_idx is None

    @property
    def coords_mpin(self):
        if self._coords_mpin is None:
            self._coords_mpin = em2mpinref(self.coords_em)
        return self._coords_mpin

    @property
    def coords_ipn(self):
        if self._coords_ipn is None:
            self._coords_ipn = em2ipnref(self.coords_em)

        out_coords = self._coords_ipn.copy()
        m = MIDLINES["ipn"]
        if (
            (self.soma_idx is not None)
            and (out_coords[self.soma_idx, 2] > 115)
            and (self.mirror == "right")
        ):
            out_coords[:, 2] = m - (out_coords[:, 2] - m)
        elif (
            self.soma_idx is not None
            and (out_coords[self.soma_idx, 2] < 115)
            and (self.mirror == "left")
        ):
            out_coords[:, 2] = m + (m - out_coords[:, 2])

        return out_coords

    @property
    def is_left(self):
        # This is duplicated with the soma side finding and "mirror" attribute.
        # Horrible but I don't want to break notebooks now.
        if self.is_axon:
            centroid = np.median(self.coords_ipn[self.axon_idxs, :], 0)
        else:
            centroid = np.median(self.coords_ipn[self.dendr_idxs, :], 0)

        return centroid[2] > MIDLINES["ipn"]

    def _get_coords_unilat(self, coords, midline):
        # Find dendrites centroid, and flip cell if on left side of midline
        if self.is_axon:
            centroid = np.median(coords[self.axon_idxs, :], 0)
        else:
            centroid = np.median(coords[self.dendr_idxs, :], 0)
        if centroid[2] > midline:
            return coords
        else:
            new_coords = coords.copy()
            new_coords[:, 2] = midline - (new_coords[:, 2] - midline)
            return new_coords

    @property
    def coords_unilat_mpin(self):
        return self._get_coords_unilat(self.coords_mpin, MIDLINES["mpin"])

    @property
    def coords_unilat_ipn(self):
        return self._get_coords_unilat(self.coords_ipn, MIDLINES["ipn"])

    @property
    def has_axon(self):
        return self.ax_start_idx is not None

    @property
    def axon_idxs(self):
        # If this is an axon, all ids are axon nodes
        if self.is_axon:
            return self.points_df.index.values
        else:
            if len(self._axon_idxs) == 0 and self.has_axon:
                self.find_ax_idxs()
            return self._axon_idxs

    @property
    def dendr_idxs(self):
        if len(self._dendr_idxs) == 0:
            sel = np.ones(len(self.points_df), dtype=bool)
            sel[self.axon_idxs] = False
            self._dendr_idxs = np.argwhere(sel)[:, 0]
        return self._dendr_idxs

    @property
    def dendrites_edges(self):
        return _edges_selection(self.edges, self.dendr_idxs)

    @property
    def axon_edges(self):
        if len(self.axon_idxs) > 0:
            return _edges_selection(self.edges, self.axon_idxs)
        else:
            return []

    def get_coords(self, space):
        return getattr(self, f"coords_{space}")

    def find_ax_idxs(self):
        # Make moving possible in both directions:
        extend_edg = np.concatenate([self.edges, np.flip(self.edges, 1)]).copy()

        if self.soma_idx is not None:
            # Find node connecting soma and axon:
            link = _find_axon_link(extend_edg, self.soma_idx, self.ax_start_idx)
            # Disconnect all edges from last point visited before finding axon start:
            extend_edg[(extend_edg[:, 0] == link) | (extend_edg[:, 1] == link)] = 0

        # Then, start from axon and find all connected points:
        idxs = np.array(_find_all_connected(extend_edg, self.ax_start_idx))

        self._axon_idxs = np.unique(idxs)

    def generate_mesh(
        self, dendrite_radius=4, axon_radius=4, soma_radius=100, space="em"
    ):
        N_SECTIONS = 7
        SOMA_SUBDIVS = 4

        coords = self.get_coords(space).copy()
        coords[:, 2] = -coords[:, 2]
        if not self.is_axon:
            return make_full_neuron(
                coords,
                self.soma_idx,
                self.dendrites_edges,
                self.axon_edges,
                dendrite_radius=dendrite_radius,
                axon_radius=axon_radius,
                soma_radius=soma_radius,
                n_sections=N_SECTIONS,
                soma_subdivisions=SOMA_SUBDIVS,
            )
        else:
            return make_cylinder_tree(
                coords,
                self.axon_edges,
                n_sections=N_SECTIONS,
                radius=axon_radius,
            )

    def generate_plotlines_from_skeleton(self, select="all", space="ipn", mirror=False):
        return _plotlines_from_skeleton(self.get_coords(space), self.edges_dict[select])

    def find_centroid_bifurcation(self, select="all", space="ipn"):
        bifurcations = find_bifurcations(self.edges_dict[select])
        return np.median(self.get_coords(space)[bifurcations, :], 0)

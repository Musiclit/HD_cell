import numpy as np
from numba import jit


def find_bifurcations(edges):
    duplicated = np.concatenate([edges, edges[:, ::-1]])
    unique, counts = np.unique(duplicated, return_counts=True)
    return unique[counts > 4]


@jit(nopython=True)
def _find_axon_link(extend_edg, soma_id, ax_id):
    """Function to find last link between soma and axon node."""
    tovisit = [soma_id]
    visited = []

    keepsearch = True
    while len(tovisit) > 0 and keepsearch:
        visiting = tovisit.pop()
        if visiting not in visited:
            new_nodes = extend_edg[extend_edg[:, 0] == visiting][:, 1]
            for n in new_nodes:
                if ax_id == n:
                    keepsearch = False
            tovisit.extend(new_nodes)
        visited.append(visiting)

    return visiting


@jit(nopython=True)
def _find_all_connected(extend_edg, ax_id):
    """Function to find all points connected to seed.
    To be used after disconnecting soma from axon.
    """
    tovisit = [ax_id]
    visited = []
    while len(tovisit) > 0:
        visiting = tovisit.pop()

        if visiting not in visited:
            new_nodes = extend_edg[extend_edg[:, 0] == visiting][:, 1]
            tovisit.extend(new_nodes)
        visited.append(visiting)

    return visited


@jit(nopython=True)
def _edges_selection(edges, idxs):
    """Find all edges that contain an index from a index array."""
    good_edges = np.full(edges.shape, -1)

    for i, edge in enumerate(edges):
        if (edge[0] == idxs).any() or (edge[1] == idxs).any():
            good_edges[i, :] = edge

    return good_edges[good_edges[:, 0] >= 0, :]


@jit(nopython=True)
def _plotlines_from_skeleton(coords, edges):
    lines_arr = np.full((len(edges) * 3, 3), np.nan)

    for i, (start, end) in enumerate(edges):
        for j, entry in enumerate(
            [coords[start, :], coords[end, :], np.full(3, np.nan)]
        ):
            lines_arr[i * 3 + j, :] = entry

    return lines_arr

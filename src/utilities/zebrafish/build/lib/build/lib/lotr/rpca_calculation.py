from itertools import product

import numpy as np

from lotr.utils import get_rot_matrix, get_vect_angle, reduce_to_pi


# TODO rename to a more insightful descriptive name
def get_zero_mean_weights(coords):
    """Normalize coords to be used as weights for fit.

    Parameters
    ----------
    coords : np.array
        (n_rois, 3) matrix with the roi coordinates.

    Returns
    -------
    np.array
        (n_rois, 3) matrix with the normalized coordinates to be used as weights.


    """

    # For PCA registration, convert coords to have mean 0 anb be used as weights.
    # Use percentiles along every axis to normalize them for removing outliers.
    w_coords = coords - np.percentile(coords, 2, axis=0)
    w_coords[w_coords < 0] = 0

    for i in range(coords.shape[1]):
        thr = np.percentile(coords[:, i], 98)
        coords[coords[:, i] > thr, i] = thr

    w_coords = w_coords / np.sum(w_coords, 0)
    w_coords = w_coords - np.mean(w_coords, 0)

    return w_coords


def reorient_pcs(cpc_scores, w_coords):
    """Reorient centered PC scores (over time) so that the position of
    cells in the PC space matches their anatomical location.

    Parameters
    ----------
    cpc_scores : np.array
        (n_rois, 2) matrix with projection of cells over centered
        principal components.
    w_coords : np.array
        (n_rois, 3) matrix with the coordinates to be used as weights
        (already normalized).

    Returns
    -------
    np.array
        (n_rois, 2) matrix with projection of cells over rotated
        principal components.
    """

    # We compute the PCs vector averages across the population using coordinates along
    # each anatomical axis as weights.
    # The result is a 3 x 2 matrix containing the average vector for each of the
    # 3 anatomical axes used as weights
    avg_vects = np.einsum("ij,ik->jk", cpc_scores, w_coords)

    # We then compute average angle for all axes:
    avg_angles = get_vect_angle(avg_vects)
    # and we take the mean between left-right and front-caud axes angles:
    # and we take the mean between left-right and anterior-posterior axes angles:
    mean_angle = np.angle(np.sum(np.cos(avg_angles)) + 1j * np.sum(np.sin(avg_angles)))

    # Since PC signs can be arbitrary, we also need to find whether left and right
    # were flipped. This we decide based on the difference in sign from the angle
    # between the lateral and sagittal axes fit in PC space:
    s = np.sign(reduce_to_pi(avg_angles[1] - avg_angles[0]))
    invert_mat = np.array([[1, 0], [0, s]])

    # At this point, we simply need to rotate the coordinates so that the
    # mean angle between vector pointing forward and vector pointing rightward is placed
    # at (1/4)*pi (angle NE):
    FINAL_TH_SHIFT = -(1 / 4) * np.pi

    rpc_scores = (
        get_rot_matrix(FINAL_TH_SHIFT)
        @ get_rot_matrix(-mean_angle * s)
        @ invert_mat
        @ cpc_scores.T
    ).T

    return rpc_scores

    return rpc_scores


def match_rpc_and_neuron_phases(rpc_phases, neuron_phases):
    """Function to match phase fit from neuron's best activation
    over network trajectory to neuron phase in rPC.

    Parameters
    ----------
    rpc_phases
    neuron_phases

    Returns
    -------

    """
    shifts = np.arange(-np.pi * 2, np.pi * 2, 0.05)
    coefs = [1, -1]
    params_list = list(product(coefs, shifts))
    residuals = np.zeros(len(params_list))
    for i, (coef, shift) in enumerate(params_list):
        new_phases = reduce_to_pi(neuron_phases * coef + shift)
        residuals[i] = np.sum(np.abs(new_phases - rpc_phases))

    return params_list[np.argmin(residuals)]

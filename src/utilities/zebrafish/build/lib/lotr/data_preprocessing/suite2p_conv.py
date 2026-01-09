import json
from pathlib import Path

import flammkuchen as fl
import numpy as np
from numba import njit, prange


def export_suite2p(data_path, output_filename=None, include_all_rois=False):
    """Export data from suite2p. A bit drafty and untested for the option include_all_rois=True.
    It saves a file containing the following:
     - traces: raw F traces from suite2p
     - coords: coordinates of traces from suite2p
     - areas: areas of cells from suite2p
     - anatomy_stack: stack with anatomy after suite2p alignment
     - rois_stack: stack with the ROIs positions
     - suite2p idxs: map of the suite2p indexes for all the filtered cells (to compare things in the viewer)
    Parameters
    ----------
    data_path : str or Path obj
        path to the folder containing suite2p data (usually, the "combined" folder)
    dest_path : str or Path obj (default: None)
        destination folder. If None, parent of the parent of path (normally, the fish data folder)
    include_all_rois : bool (default: False)
        If true, all the ROIs are extracted, including the ones classified as not cells by Suite2p
    """

    data_path = Path(data_path)
    OUTPUT_FILENAME = "data_from_suite2p.h5"

    # Load suite2p data:
    traces = np.load(data_path / "F.npy")[()]
    iscell = np.load(data_path / "iscell.npy")
    ops = np.load(data_path / "ops.npy", allow_pickle=True)[()]
    stat = np.load(data_path / "stat.npy", allow_pickle=True)[()]
    rois_stack = np.zeros((ops["nplanes"],) + ops["refImg"].shape, dtype=np.int)

    # Create the ROI stack, excluding ROIs that are all zero and, if required,
    # ROIs excluded by suite2p.
    # We'll use the suite2p_idxs variable to map old suite2p
    # ndices to new exported ones.
    suite2p_idxs = []
    k = 1
    for j, a in enumerate(stat):
        ypix = a["ypix"].flatten()
        xpix = a["xpix"].flatten()
        # If required, do this only if the ROI is classified as cell. In any case,
        # exclude if the ROI has all 0s in F trace:
        if (iscell[j, 0] or include_all_rois) and not (traces[j, :] == 0).all():
            try:
                i_plane = a["iplane"]
            except KeyError:
                i_plane = 0
            rois_stack[
                i_plane, ypix % rois_stack.shape[1], xpix % rois_stack.shape[2]
            ] = k
            k += 1
            suite2p_idxs.append(j)

    rois_stack = rois_stack - 1

    # Create the anatomy stack:
    anatomy_stack = np.zeros((ops["nplanes"],) + ops["refImg"].shape)
    flat_img = ops["meanImg"]
    n = [m // r for m, r in zip(flat_img.shape, ops["refImg"].shape)]

    for i_plane in range(ops["nplanes"]):
        img_idxs = (i_plane // n[1], i_plane % n[1])
        anatomy_stack[i_plane, :, :] = flat_img[
            tuple(
                [
                    slice(idx * s, (idx + 1) * s)
                    for idx, s in zip(img_idxs, anatomy_stack.shape[1:])
                ]
            )
        ]

    # Filter cells if requested:
    if not include_all_rois:
        traces = traces[iscell[:, 0].astype(np.int) > 0, :]

    # Exclude traces of all zeros:
    traces = traces[~(traces == 0).all(1), :]

    # Extract roi coords and areas
    coords, areas = get_roi_coords_areas(rois_stack)

    if output_filename is None:
        output_filename = data_path.parent.parent / OUTPUT_FILENAME
    else:
        output_filename = Path(output_filename)

    fl.save(
        output_filename,
        dict(
            traces=traces,
            coords=coords,
            areas=areas,
            anatomy_stack=anatomy_stack,
            suite2p_idxs=suite2p_idxs,
            rois_stack=rois_stack,
        ),
        compression=None,
    )


def export_suite2p_data_2p(master_path, output_filename=None, include_all_rois=False):
    """Export data from suite2p for 2p data (plane-wise analyzed).
    It saves a file containing the following:
     - traces: raw F traces from suite2p
     - coords: coordinates of traces from suite2p
     - areas: areas of cells from suite2p
     - anatomy_stack: stack with anatomy after suite2p alignment
     - rois_stack: stack with the ROIs positions
     - suite2p idxs: map of the suite2p indexes for all the filtered cells (to compare things in the viewer)
    Parameters
    ----------
    data_path : str or Path obj
        path to the folder containing suite2p data (usually, the "combined" folder)
    dest_path : str or Path obj (default: None)
        destination folder. If None, parent of the parent of path (normally, the fish data folder)
    include_all_rois : bool (default: False)
        If true, all the ROIs are extracted, including the ones classified as not cells by Suite2p
    """

    master_path = Path(master_path)
    OUTPUT_FILENAME = "data_from_suite2p.h5"

    k = 1

    planes = list(master_path.glob("[0-9][0-9][0-9][0-9]"))

    traces_list = []
    rois_planes_list = []
    anatomies_planes_list = []

    if len(planes) == 0:
        raise FileNotFoundError("No planes directories in the specified path!")

    for plane in planes:
        data_path = plane / "plane0"  # Path(data_path)
        # Load suite2p data:
        traces = np.load(data_path / "F.npy")[()]
        iscell = np.load(data_path / "iscell.npy")
        ops = np.load(data_path / "ops.npy", allow_pickle=True)[()]
        stat = np.load(data_path / "stat.npy", allow_pickle=True)[()]
        rois_stack = np.zeros((ops["nplanes"],) + ops["refImg"].shape, dtype=np.int)

        suite2p_idxs = []
        for j, a in enumerate(stat):
            ypix = a["ypix"].flatten()
            xpix = a["xpix"].flatten()
            # If required, do this only if the ROI is classified as cell. In any case, exclude if the ROI has all
            # 0s in F trace:
            if (iscell[j, 0] or include_all_rois) and not (traces[j, :] == 0).all():
                rois_stack[
                    0, ypix % rois_stack.shape[1], xpix % rois_stack.shape[2]
                ] = k
                k += 1
                suite2p_idxs.append(j)

        # exclude all zeros and filter cells if requested:
        if not include_all_rois:
            traces = traces[iscell[:, 0].astype(np.int) > 0, :]
        traces = traces[~(traces == 0).all(1), :]

        traces_list.append(traces)
        rois_planes_list.append(rois_stack)
        anatomies_planes_list.append(ops["meanImg"][np.newaxis, :, :])

    rois_stack = np.concatenate(rois_planes_list, 0)
    # Extract roi coords and areas
    coords, areas = get_roi_coords_areas(rois_stack)

    if output_filename is None:
        output_filename = master_path.parent / OUTPUT_FILENAME
    else:
        output_filename = Path(output_filename)

    fl.save(
        output_filename,
        dict(
            traces=traces_list,
            coords=coords,
            areas=areas,
            anatomy_stack=np.concatenate(anatomies_planes_list, 0),
            suite2p_idxs=suite2p_idxs,
            rois_stack=rois_stack,
        ),
        compression=None,
    )


@njit(parallel=True)
def get_roi_coords_areas(rois: np.ndarray):
    """A function to efficiently extract ROI coordinates and areas from a ROI stack
    :param rois: int np.array where each ROI is labeled by an integer, and -1 otherwise
    :return: coords, areas
    """
    n_rois = np.max(rois + 1)
    coords = np.zeros((n_rois, len(rois.shape)))
    areas = np.zeros(n_rois, np.int32)
    for i in prange(rois.shape[0]):
        for j in range(rois.shape[1]):
            for k in range(rois.shape[2]):
                roi_id = rois[i, j, k]
                if roi_id > -1:
                    areas[roi_id] += 1
                    coords[roi_id, 0] += i
                    coords[roi_id, 1] += j
                    coords[roi_id, 2] += k

    for i in range(n_rois):
        coords[i, :] /= areas[i]

    return coords, areas


def export_suite2p_registered(fish_path, s2p_path=None, output_path=None):
    """Export suite2p registered data from bin files to split-dataset
    Parameters
    ----------
    fish_path : str or Path obj
        path to the fish data folder (containing original folder and suite2p data)
    s2p_path : str or Path obj (default: None)
        path to the folder containing suite2p data. If None, the same path as fish_path
    output_path : str or Path obj (default: None)
        destination folder. If None, fish_path
    """
    OUTPUT_DIRNAME = "s2p_registered"

    fish_path = Path(fish_path)
    if s2p_path is None:
        s2p_path = fish_path
    else:
        s2p_path = Path(s2p_path)

    bin_list = list(s2p_path.glob("**/data.bin"))
    if len(bin_list) == 0:
        raise FileNotFoundError("No data.bin file was found in the specified path!")
    bin_list = sorted(bin_list)

    if output_path is None:
        output_path = fish_path / OUTPUT_DIRNAME
    else:
        output_path = Path(output_path) / OUTPUT_DIRNAME
    output_path.mkdir()

    with open(fish_path / "original/stack_metadata.json") as f:
        meta = f.read()
    with open(output_path / "stack_metadata.json", "w") as f:
        f.write(meta)
    shape = json.loads(meta)["shape_block"]

    for i, b in enumerate(bin_list):
        registered = np.fromfile(b, dtype=np.int16)
        registered = registered.reshape(shape)
        fl.save(
            output_path / "{:04d}.h5".format(i),
            {"stack_4D": registered},
            compression="blosc",
        )

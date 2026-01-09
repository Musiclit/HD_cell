from pathlib import Path

import flammkuchen as fl
import numpy as np
from bouter import EmbeddedExperiment

from lotr.behavior import get_fictive_heading
from lotr.data_preprocessing.anatomy import reshape_stack, transform_points
from lotr.data_preprocessing.stimulus import get_all_trials_df
from lotr.default_vals import (
    LIGHTSHEET_CAMERA_RES_XY,
    PCA_TIME_PAD_S,
    TO_IPNREF_MTX,
)
from lotr.pca import pca_and_phase
from lotr.plotting import color_stack
from lotr.rpca_calculation import get_zero_mean_weights, reorient_pcs


class LotrExperiment(EmbeddedExperiment):
    """Main class for data loading. Look here to follow how any experimental
    quantity loaded in a notebook is taken from the raw files. To check
    how semi-processed files are generated,
    look into lotr/scripts/00_folder_preprocessing.py.

    NOTES: Anatomical space
        Anatomical stacks are returned to follow the following convention:
            (inferior-superior, posterior-anterior, left-right)
        This means that the voxel 0 is located at the ventral-caudal-left corner.

        **Figures are all coded to be top view (left=left of the fish, top=rostral).**

        Therefore:
        plt.imshow for the stacks should be used specifying the origin="lower" argument
        for left to be left of the fish in top view!

        For the coordinates to follow the same convention (origin is set to be
        at the ventral-caudal-left corner when scattering the second two
        coords, left is left and right is right), we have to change their order.
        Coordinates will be:
            (inferior-superior, left-right, posterior-anterior)

        The following code will produce correctly oriented fish:

        # Pixel coordinates:
        plt.imshow(exp.rois_stack.max(0), aspect="auto", origin="lower")
        plt.scatter(exp.coords[:, 1], exp.coords[:, 2])

        # Microns coordinates:
        plt.imshow(exp.rois_stack.max(0), origin="lower", extent=exp.plane_ext_um)
        plt.scatter(exp.coords_um[:, 1], exp.coords_um[:, 2])
    """

    def __init__(self, *args, selected=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.microscope_config = self["imaging"]["microscope_config"]

        self._dt_imaging = None
        self._resolution = None
        self._n_planes = None
        self._data_shape = None
        self._bouts_df = None
        self._traces = None
        self._hdn_indexes = selected
        self._motor_regressors = None
        self._coords = None
        self._raw_traces = None
        self._anatomy_stack = None
        self._rois_stack = None
        self._nonhdn_indexes = None
        self._rndcnt_indexes = None
        self._rpc_scores = None
        self._rpc_scores_shuf = None
        self._rpc_angles = None
        self._rpc_angles_shuf = None
        self._network_phase = None
        self._network_phase_shuf = None
        self._stim_trials_df = None
        self._morphed_coords = None
        self._noZ_traces = None

    @property
    def fn(self):
        # Just an outdated alias for fn
        return self.fs

    @property
    def fs(self):
        try:
            return int(
                self.microscope_config["lightsheet"]["scanning"]["z"]["frequency"]
            )
        except KeyError:
            return int(self.microscope_config["scanning"]["framerate"])

    @property
    def n_planes(self):
        try:
            return self.microscope_config["lightsheet"]["scanning"]["triggering"][
                "n_planes"
            ]
        except KeyError:
            print("Fix definition for 2p data!")

    @property
    def voxel_size_um(self):
        try:
            z_conf = self.microscope_config["lightsheet"]["scanning"]["z"]
            z_um = (z_conf["piezo_max"] - z_conf["piezo_min"]) / self.n_planes

            return (z_um,) + LIGHTSHEET_CAMERA_RES_XY
            return

        except KeyError:
            z_config = self.microscope_config["recording"]["dz"]
            voltage_max = self.microscope_config["scanning"]["voltage"]

            x_res = compute_resolution_for_2p(voltage_max, 355)
            y_res = compute_resolution_for_2p(voltage_max, 355)
            return (z_config, x_res, y_res)

    @property
    def lr_extent_um(self):
        return 0, self.rois_stack.shape[2] * self.voxel_size_um[2]

    @property
    def pa_extent_um(self):
        return 0, self.rois_stack.shape[1] * self.voxel_size_um[1]

    @property
    def plane_ext_um(self):
        return self.lr_extent_um + self.pa_extent_um

    @property
    def dt_imaging(self):
        return 1 / self.fs

    @property
    def dir_name(self):
        return self.root.name

    @property
    def bouts_df(self):
        if self._bouts_df is None:
            # TODO move this to preprocessing
            df = fl.load(self.root / "bouts_df.h5")
            self._bouts_df = df[df["t_start"] > 0]
        return self._bouts_df

    @property
    def motor_regressors(self):
        if self._motor_regressors is None:
            self._motor_regressors = fl.load(self.root / "motor_regressors.h5")
        return self._motor_regressors

    @property
    def traces(self):
        if self._traces is None:
            self._traces = fl.load(self.root / "filtered_traces.h5", "/detr")
        return self._traces

    @property
    def raw_traces(self):
        if self._raw_traces is None:
            self._raw_traces = fl.load(
                self.root / "data_from_suite2p_unfiltered.h5", "/traces"
            ).T
        return self._raw_traces
    
    @property
    def noZ_traces(self):
        if self._noZ_traces is None:
            self._noZ_traces = np.load(self.root / "traces_no_zscore.npy", allow_pickle=True)
        return self._noZ_traces

    @property
    def rois_stack(self):
        if self._rois_stack is None:
            self._rois_stack = reshape_stack(
                fl.load(self.root / "data_from_suite2p_unfiltered.h5", "/rois_stack")
            )
        return self._rois_stack

    @property
    def anatomy_stack(self):
        if self._anatomy_stack is None:
            self._anatomy_stack = reshape_stack(
                fl.load(self.root / "data_from_suite2p_unfiltered.h5", "/anatomy_stack")
            )
        return self._anatomy_stack

    @property
    def coords(self):
        if self._coords is None:
            self._coords = fl.load(
                self.root / "data_from_suite2p_unfiltered.h5", "/coords"
            )
        return self._coords

    @property
    def coords_um(self):
        return self.coords * self.voxel_size_um

    @property
    def trasform_mat_to_lsref(self):
        return np.load(self.root / "centering_mtx.npy")

    @property
    def morphed_coords(self):
        try:
            transform_mat = self.trasform_mat_to_lsref
            return transform_points(self.coords, transform_mat)
        except FileNotFoundError:
            raise Warning("No centering_mtx.npy file found for morphing, returning raw")
            return self.coords

    @property
    def morphed_coords_um(self):
        return self.morphed_coords * self.voxel_size_um

    @property
    def ipnref_coords_um(self):
        return transform_points(self.morphed_coords_um, TO_IPNREF_MTX)

    @property
    def hdn_indexes(self):
        if self._hdn_indexes is None:
            self._hdn_indexes = np.array(fl.load(self.root / "selected.h5"))
        return self._hdn_indexes

    @property
    def nonhdn_indexes(self):
        if self._nonhdn_indexes is None:
            non_hdns = np.ones(self.n_rois, dtype=bool)
            non_hdns[self.hdn_indexes] = False
            self._nonhdn_indexes = non_hdns
        return self._nonhdn_indexes

    @property
    def rndcnt_indexes(self):
        all_possible = np.argwhere(self.nonhdn_indexes).flatten()
        np.random.shuffle(all_possible)
        return all_possible[: len(self.hdn_indexes)]

    @property
    def n_pts(self):
        return self.traces.shape[0]

    @property
    def n_rois(self):
        return self.traces.shape[1]

    @property
    def n_hdns(self):
        return len(self.hdn_indexes)

    @property
    def has_hdn(self):
        return (self.root / "selected.h5").exists()

    @property
    def time_arr(self):
        return np.arange(1, self.n_pts + 1) / self.fs

    @property
    def exp_type(self):
        # TODO use experiment versions here

        return self.root.name.split("_")[2].strip("_")

    @property
    def pca_t_lims(self):
        """Time slicing for the calculation of PCA, excluding the initial part of the
        trace (where drifts/things related to the beginning of the experiment might be
        happening) and the parts with strong stimuli in protocols with dir motion.
        """

        exp_end = int(self["stimulus"]["log"][-1]["t_stop"])
        if self.exp_type == "cwccw":
            return PCA_TIME_PAD_S, int(self["stimulus"]["log"][1]["t_start"])
        elif self.exp_type == "2dvr":
            return PCA_TIME_PAD_S, int(self["stimulus"]["log"][2]["t_start"])
        else:
            return PCA_TIME_PAD_S, exp_end - PCA_TIME_PAD_S

    # TODO all this part on rPC calculation should be refactored somewhere else
    @property
    def pca_t_slice(self):
        t_lims = self.pca_t_lims
        return slice(*[t * self.fs for t in t_lims])

    def _compute_rpc_scores(self, idxs):
        """For a tutorial on this calculation, have a look at
        'Anatomical organization of the network.ipynb'
        """
        # 1. compute PCs:
        pca_scores, angles, _, circle_params = pca_and_phase(
            self.traces[self.pca_t_slice, idxs].T
        )
        # 2. center on 0:
        centered_pca_scores = pca_scores[:, :2] - circle_params[:2]

        # 3. Find transformation to match anatomy
        # Normalize coords (we don't care about z here)
        w_coords = get_zero_mean_weights(self.coords[self.hdn_indexes, 1:])

        # Find transformation to have at 0 angle rostral ROIs:
        return reorient_pcs(centered_pca_scores, w_coords)

    @property
    def rpc_scores(self):
        if self._rpc_scores is None:
            self._rpc_scores = self._compute_rpc_scores(self.hdn_indexes)
        return self._rpc_scores

    @property
    def rpc_scores_shuf(self):
        if self._rpc_scores_shuf is None:
            self._rpc_scores_shuf = self._compute_rpc_scores(self.rndcnt_indexes)
        return self._rpc_scores_shuf

    @property
    def stim_trials_df(self):
        if self._stim_trials_df is None:
            self._stim_trials_df = get_all_trials_df(self)

        return self._stim_trials_df

    @property
    def fictive_heading(self):
        return get_fictive_heading(self.n_pts, self.bouts_df)

    @staticmethod
    def _get_pc_scores_angles(pc_scores):
        """For a tutorial on this calculation, have a look at
        'Anatomical organization of the network.ipynb'
        """
        return np.arctan2(pc_scores[:, 1], -pc_scores[:, 0])

    @property
    def rpc_angles(self):
        if self._rpc_angles is None:
            self._rpc_angles = self._get_pc_scores_angles(self.rpc_scores)

        return self._rpc_angles

    @property
    def rpc_angles_shuf(self):
        if self._rpc_angles_shuf is None:
            self._rpc_angles_shuf = self._get_pc_scores_angles(self.rpc_scores_shuf)

        return self._rpc_angles_shuf

    def get_network_phase(self, indexes, rpc_scores):
        """For a tutorial on this calculation, have a look at
        'Anatomical organization of the network.ipynb'
        """
        # TODO write as matrix multiplication
        norm_activity = get_zero_mean_weights(self.traces[:, indexes].T).T
        avg_vects = np.einsum("ij,ik->jk", norm_activity.T, rpc_scores)

        # This choice of signs ensures that network phase correspond to angle of
        # max activation or ROIs over rPC space:
        return np.arctan2(avg_vects[:, 1], -avg_vects[:, 0])

    @property
    def network_phase(self):
        if self._network_phase is None:
            self._network_phase = self.get_network_phase(
                self.hdn_indexes, self.rpc_scores
            )

        return self._network_phase

    @property
    def network_phase_shuf(self):
        if self._network_phase_shuf is None:
            self._network_phase_shuf = self.get_network_phase(
                self.rndcnt_indexes, self.rpc_scores_shuf
            )

        return self._network_phase_shuf

    def find_mirror_dir(self, parent_folder):
        """Find homonym directory in a new parent folder, for file mirroring."""
        parent_folder = Path(parent_folder)
        matches = list(parent_folder.rglob(self.dir_name))
        if len(matches) > 1:
            raise OSError("Multiple folders found for this experiment")
        return matches[0]

    # TODO might have to go somewhere else, not too coherent with class
    def color_rois_by(self, values, indexes=None, **kwargs):
        """Color ROI stack by some convention (phase, activity, etc.). If no indexes
        are passed, we assume we are coloring either all ROIs or HDNs.

        Parameters
        ----------
        values : np.array
            Array over which to color the stack
        indexes : np.array (optional)
            Index of ROIs to color (if different from either all or HDNs only)
        kwargs : dict
            Additional kwargs for the color_stack function. Most importantly,
            color_scheme and vlims might be handy to set.

        Returns
        -------

        """
        if indexes is None:
            if len(values) == self.n_rois:
                indexes = np.arange(self.n_rois)
            elif len(values) == len(self.hdn_indexes):
                indexes = self.hdn_indexes
            else:
                raise ValueError(f"Can't infer index for variable of len {len(values)}")

        full_val_arr = np.full(self.n_rois, np.nan)
        full_val_arr[indexes] = values
        return color_stack(self.rois_stack, variable=full_val_arr, **kwargs)

    @property
    def exp_code(self):
        return "f" + str(hash(self.root.name))[-5:]


def compute_resolution_for_2p(zoom, size_px):
    # Calibration data:
    dist_in_um = 10
    dist_in_px = np.array([21.13, 19.62, 8.93])
    zooms = np.array([1.5, 3, 4.5])
    image_max_sizes = np.array([330, 610, 410])

    return np.mean(
        (dist_in_um / dist_in_px) * (zoom / zooms) * (image_max_sizes / size_px)
    )

import numpy as np
import pandas as pd
from tqdm import tqdm

from lotr import dataset_folders
from lotr.default_vals import DEFAULT_FN, POST_BOUT_WND_S, PRE_BOUT_WND_S
from lotr.experiment_class import LotrExperiment
from lotr.utils import crop, interpolate, resample_matrix


def crop_shifts_all_dataset(crop_stimulus=False):
    """Crop fictive heading and network phase around bouts from all fish.
    in the dataset. For a demo of what is happening, "4. Phase dynamics.ipynb" notebook.

    Returns
    -------
    (all_phase_cropped, all_head_cropped, events_df)
        The first two returns are the cropped n_tpts x n_bouts matrices, the third is
        the dataframe that contains the info about all events.

    """
    fn = DEFAULT_FN

    all_phase_cropped = []
    all_head_cropped = []
    all_stim_cropped = []
    # We will create a dataframe to keep track of events from all fish.
    # Mostly a way of keeping together the crop and the bouts:
    events_df = []

    # Define temporal array for the resampling:
    time_arr = (
        np.arange(1, ((PRE_BOUT_WND_S + POST_BOUT_WND_S) * fn) + 1) / fn
        - PRE_BOUT_WND_S
    )
    for path in tqdm(dataset_folders):
        exp = LotrExperiment(path)
        # TODO recompute to avoid this bugfix
        exp.bouts_df["fid"] = path.name

        stim_interp = np.full(exp.n_pts, np.nan)
        try:
            stim_df = exp.stimulus_log
            if "cl2D_theta" in stim_df.columns and crop_stimulus:
                stim_interp = interpolate(
                    stim_df["t"], stim_df["cl2D_theta"], exp.time_arr
                )
        except AttributeError:
            pass

        # Crop both the fictive heading (cumulative tail theta sum) and network phase
        # in the same way:
        for dest_list, to_crop in zip(
            [all_phase_cropped, all_head_cropped, all_stim_cropped],
            [np.unwrap(exp.network_phase), exp.fictive_heading, stim_interp],
        ):
            # Crop around events:
            cropped = crop(
                to_crop,
                exp.bouts_df["idx_imaging"],
                pre_int=PRE_BOUT_WND_S * exp.fs,
                post_int=POST_BOUT_WND_S * exp.fs,
            )

            # Subtract baseline:
            cropped = cropped - np.mean(cropped[: PRE_BOUT_WND_S * exp.fs, :], 0)

            # Interpolate if necessary:
            if exp.fs != fn:
                fish_time_arr = (
                    np.arange(1, cropped.shape[0] + 1) / exp.fs - PRE_BOUT_WND_S
                )
                cropped = resample_matrix(time_arr, fish_time_arr, cropped)

            dest_list.append(cropped)
        events_df.append(exp.bouts_df.reindex())

    # Concatenate all the results:
    all_phase_cropped = np.concatenate(all_phase_cropped, axis=1)
    all_head_cropped = np.concatenate(all_head_cropped, axis=1)
    try:
        all_stim_cropped = np.concatenate(all_stim_cropped, axis=1)
    except np.AxisError:
        all_stim_cropped = None
    events_df = pd.concat(events_df, ignore_index=True)

    if crop_stimulus:
        return (
            all_phase_cropped,
            all_head_cropped,
            all_stim_cropped,
            events_df,
            time_arr,
        )
    else:
        return all_phase_cropped, all_head_cropped, events_df, time_arr

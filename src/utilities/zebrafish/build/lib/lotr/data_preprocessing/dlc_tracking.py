import flammkuchen as fl
import numpy as np
import pandas as pd
from bouter.utilities import (
    polynomial_tail_coefficients,
    polynomial_tailsum,
    predictive_tail_fill,
)
from scipy.signal import medfilt


def _get_angles(df1, df2, norm=False, lh_thr=1.0):
    th = np.arctan2(
        df2["y"] - df1["y"],
        df2["x"] - df1["x"],
    )
    if norm:
        th = th.values - np.mean(th)

    if lh_thr < 1:
        th[(df1["likelihood"] < lh_thr) | (df2["likelihood"] < lh_thr)] = np.nan

    return th


def export_dlc_behavior(dlc_file_dir, tail_lh_thr=0.8, medfilt_wnd_s=0.8):
    # Load DLC file:
    dlc_filename = next(dlc_file_dir.glob("*DLC*.h5"))
    raw_data_df = fl.load(dlc_filename, "/df_with_missing")

    # Load stytra log, we'll need this to sync time:
    stytra_beh_log = fl.load(next(dlc_file_dir.glob("*behavior*.hdf5")), "/data")
    stytra_time_arr = stytra_beh_log["t"].values
    dt = np.diff(stytra_time_arr).mean()
    fs = 1 / dt

    # We used only one scorer, drop that level in the multiindex columns:
    df = raw_data_df.droplevel("scorer", axis=1)

    # Calculate eye angles.
    # Keys of the eyes as they were named in the model:
    eyes_ks_dict = dict(
        lf_eye=[f"l_eye_{k}" for k in ["0", "1", "3", "4"]],
        rt_eye=[f"r_eye_{k}" for k in ["0", "1", "3", "4"]],
    )

    data_dict = dict()
    for s in eyes_ks_dict.keys():
        k0 = eyes_ks_dict[s][0]  # first key of the eye:
        th = np.zeros((len(df[k0]), 3))  # array to fill with angles

        # loop over remaining points of the eye, and for each find angle with first
        # point:
        for i, k in enumerate(eyes_ks_dict[s][1:]):
            th[:, i] = _get_angles(df[k0], df[k])

        # reduce noise by taking median of the (3) angles series
        median_th = np.median(th, 1)
        data_dict[s] = median_th

        # median filter eye theta:
        # Perform median filtering in a window to remove 5 Hz microscope oscillation
        medfilt_wnd_pts = int(medfilt_wnd_s * fs)
        medfilt_wnd_pts += 1 - medfilt_wnd_pts % 2  # ensure it's odd
        data_dict[s + "_medfilt"] = medfilt(median_th, medfilt_wnd_pts)

    # Calculate tail angles:
    # Keys of the tail as they were named in the model:
    tail_keys = ["swim_b"] + [f"tail_{i}" for i in range(10)]
    thetas = []
    for tail_key0, tail_key1 in zip(tail_keys[:-1], tail_keys[1:]):
        thetas.append(_get_angles(df[tail_key0], df[tail_key1], lh_thr=tail_lh_thr))
    thetas = np.array(thetas).T

    # Use bouter function to fill missing tail segments and computing tail sum:
    thetas_fixes = predictive_tail_fill(thetas.copy())
    data_dict["tail_sum"] = polynomial_tailsum(
        polynomial_tail_coefficients(thetas_fixes)
    )

    # Get time array syncing it with the behavior log:
    data_dict["t"] = stytra_time_arr[data_dict["tail_sum"].shape[0] :]

    # Wrap together data and put it in dataframe with columns:
    data_df = pd.DataFrame(data_dict)

    # There are some arbitrary sign changes here. This is require to match stytra angles
    data_df["tail_sum"] = -data_df["tail_sum"]
    fl.save(dlc_file_dir / "behavior_from_dlc.h5", dict(data=data_df))

    return data_df


if __name__ == "__main__":
    from tqdm import tqdm

    from lotr import DATASET_LOCATION

    to_convert = [f.parent for f in DATASET_LOCATION.glob("*/*/*DLC*.h5")]

    for f in tqdm(to_convert):
        export_dlc_behavior(f)

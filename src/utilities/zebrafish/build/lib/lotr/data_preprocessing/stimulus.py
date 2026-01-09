import pandas as pd


def get_all_trials_df(exp):
    """Quite ugly function that tries to create a stimulus dataframe for an experiment
    containing the essential quantities that recapitulate it. So far such
    contained description has:
        - condition : tags lumping together comparable stimuli (see code)
        - t_start : stimulus start
        - t_stop : stimulus end
        - gain_theta : gain factor for closed- and open- loop data. Defined for cl2d.
        - exp_type : type of the experiment that was run (natmov, gainmod, 2dvr, etc.)
        - name : name of the stimulus
        - ... all other stimulus properties that got saved in the stimulus log

    Parameters
    ----------
    exp : a LotrExperiment object
        Experiment to preprocess.
    Returns
    -------
    pd.DataFrame
        Dataframe creaded from the stimulus log stripped of the param_df entry with
        some additional entries for cross-experiment comparisons.

    """
    # condition key should be either:
    #  - darkness: black screen
    #  - natural_mot: natural motion (replay or inverted feedback)
    #  - directional_mot: long motion drifts from the 8 directions
    #  - cl2d: closed loop 2D

    STIM_MAPPING_DICT = dict(
        darkness=[
            (k, "pause")
            for k in ["clol", "2dvr", "cwccw", "natmov", "gainmod", "spont"]
        ],
        closed_loop=[
            ("clol", "closed_loop"),
            ("2dvr", "seamless_image"),
            ("gainmod", "cl2d"),
            ("cl", "cl2D"),
        ],
        natural_motion=[("clol", "open_loop"), ("natmov", "bg")],
        directional_motion=[("cwccw", "bg"), ("2dvr", "bg"), ("spont", "bg")],
    )

    stim_log = exp["stimulus"]["log"]

    if exp.exp_type != "gainmod":
        for s in stim_log:
            try:
                s.pop("df_param")
            except KeyError:
                pass
            s["fid"] = exp.dir_name
            s["exp_type"] = exp.exp_type
        # print(stim_log[0].keys())

    # For the gainmod experiments, replace stim log by splitting different trials:
    else:
        stim_dict = stim_log[0]
        gain, t = [stim_dict["df_param"][k] for k in ["gain_theta", "t"]]

        stim_log = []
        for start, end, gain in zip(t[::2], t[1::2], gain[::2]):
            stim_log.append(
                dict(
                    t_start=stim_dict["t_start"] + start,
                    t_stop=stim_dict["t_start"] + end,
                    name="cl2d",  # overwrite inconsistent naming
                    gain_theta=gain,
                    fid=exp.dir_name,
                    exp_type=exp.exp_type,
                )
            )

    # Dataframe from params dict list:
    annotated_df = pd.DataFrame(stim_log)
    # print(annotated_df.columns)
    annotated_df["condition"] = "-"  # initialize main condition entry

    # Closed pause, loop, natural motion, and directional motion of different kinds:
    for stim_condition, sel_criteria in STIM_MAPPING_DICT.items():
        for exp_type, name in sel_criteria:
            annotated_df.loc[
                (annotated_df["exp_type"] == exp_type) & (annotated_df["name"] == name),
                "condition",
            ] = stim_condition

            # Add gain 1 for closed-loop stimuli in experiments other than the gain
            # modulation one:
            if stim_condition == "closed_loop" and not exp.exp_type == "gainmod":
                # print("- set theta", stim_condition, exp_type)
                annotated_df["gain_theta"] = 1

    # Gain -1 condition gets compared to external motion:
    annotated_df.loc[
        (annotated_df["name"] == "cl2d") & (annotated_df["gain_theta"] > 0.0),
        "condition",
    ] = "closed_loop"

    # Make sure that all stimuli got mapped somehow:
    try:
        assert set(annotated_df["condition"]).issubset(set(STIM_MAPPING_DICT.keys()))
    except AssertionError:
        raise ValueError(f"No map found for some stimuli in exp {exp.dir_name}")

    return annotated_df

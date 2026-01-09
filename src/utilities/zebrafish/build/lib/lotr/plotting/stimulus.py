from matplotlib import pyplot as plt

from lotr.plotting.default_colors import COLS


def plot_exp_condition(exp_stim_df, ax=None, alpha=0.3, **kwargs):
    """Plot experiment conditions as vertical ranges on some axes.

    Parameters
    ----------
    exp_stim_df : pd.DataFrame
        Dataframe containing the "t_start", "t_stop" and "condition" entries.
    ax : plt.Axis (optional)
        Axis over which to plot, by default current.
    alpha : float (optional)
        Transparency (default=0.3).
    kwargs : dict
        Additional arguments for plt.axvspan function.

    Returns
    -------
    list
        All the vspan function outputs.

    """
    stim_colors = COLS["stim_conditions"]
    if ax is None:
        ax = plt.gca()

    vspan = list()
    for i in exp_stim_df.index:
        t_s, t_e = [exp_stim_df.loc[i, k] for k in ["t_start", "t_stop"]]
        if exp_stim_df.loc[i, "condition"] == "closed_loop":
            col = stim_colors["closed_loop"][exp_stim_df.loc[i, "gain_theta"]]
        else:
            col = stim_colors[exp_stim_df.loc[i, "condition"]]
        vspan.append(ax.axvspan(t_s, t_e, fc=col, lw=0, alpha=alpha, **kwargs))

    return vspan

import numpy as np


def get_pi_labels(d=1, coefs=None, ax="x", style="notex"):
    """Handy way to generate labels for plots with radiants.
    Ugly numerical keys in dictionary, but should not give troubles.
    """
    coefs_mapping = {2: (-1, 1), 1: (-1, 0, 1), 0.5: (-1, -0.5, 0, 0.5, 1)}

    labels_mappings = {
        "tex": {
            -4: "$-4π$",
            -2: "$-2π$",
            -1: "$-π$",
            -0.75: r"$\dfrac{-3π}{4}$",
            -0.5: r"$\dfrac{-π}{2}$",
            -1 / 3: r"$\dfrac{-π}{3}$",
            -0.25: r"$\dfrac{-π}{4}$",
            0: "0",
            0.25: r"$\dfrac{π}{4}$",
            1 / 3: r"$\dfrac{π}{3}$",
            0.5: r"$\dfrac{π}{2}$",
            0.75: r"$\dfrac{3π}{4}$",
            1: "$π$",
            2: "$2π$",
            4: "$4π$",
        },
        "notex": {
            -4: "-4π",
            -2: "-2π",
            -1: "-π",
            -0.75: "-3/4π",
            -0.5: "-π/2",
            -1 / 3: "-π/3",
            -0.25: "-π/4",
            -0.125: "-π/8",
            -1 / 16: "-π/16",
            0: "0",
            1 / 16: "π/16",
            0.125: "π/8",
            0.25: r"π/4",
            1 / 3: "π/3",
            0.5: "π/2",
            0.75: "3π/4",
            1: "π",
            2: "2π",
            4: "4π",
        },
    }

    if coefs is None:
        coefs = coefs_mapping[d]

    ticks = []
    tick_labels = []
    for i in coefs:
        ticks.append(i * np.pi)
        try:
            tick_labels.append(labels_mappings[style][i])
        except KeyError:
            if style == "tex":
                tick_labels.append("$" + str(i) + "π$")
            else:
                tick_labels.append(str(i) + "π")
    tickname = ax + "ticks"
    labname = ax + "ticklabels"
    return {tickname: ticks, labname: tick_labels}


def get_pval_stars(test_result):
    """Get number of stars or n.s. from p-values. Convention:
        - p < 0.001: ***
        - p < 0.01: **
        - p < 0.5: *
        - p > 0.5: n.s.

    Parameters
    ----------
    test_result : float or scipy stats Result with pval attribute
        Number or test to label with stars

    Returns
    -------
    str
        string describing the result.

    """
    if type(test_result) not in [float, np.float64]:
        test_result = test_result.pvalue

    if test_result <= 0.0001:
        return "****"
    if test_result <= 0.001:
        return "***"
    elif test_result <= 0.01:
        return "**"
    elif test_result <= 0.05:
        return "*"
    else:
        return "n.s."

import numpy as np
from matplotlib import pyplot as plt

from lotr.plotting.general import add_cbar, despine


def add_anatomy_scalebar(
    ax=None,
    length=50,
    pos=(10, 30),
    units="μm",
    plane="horizontal",
    lw=0.5,
    c=(0.2,) * 3,
    fontsize=8,
    line_params=None,
    text_params=None,
    disable_axis=True,
    equalize_axis=True,
    spacing_coef=0.06,
    cartesian=False,
    switchlabels=False,
):
    """"""

    if ax is None:
        ax = plt.gca()

    line_params_def = dict(lw=lw, c=c)
    text_params_def = dict(fontsize=fontsize, c=c)

    for default_params, params_in in zip(
        [line_params_def, text_params_def], [line_params, text_params]
    ):
        if params_in is not None:
            default_params.update(params_in)

    xlen, ylen = length, length
    if cartesian:
        ylen = -ylen
        labels_dict = dict(
            horizontal=("ant.", "rt."),
            sagittal=("post.", "dors."),
            frontal=("dors.", "rt."),
        )
    else:
        labels_dict = dict(
            horizontal=("ant.", "rt."),
            sagittal=("ant.", "dors."),
            frontal=("dors.", "rt."),
        )
    xpos, ypos = pos

    ax.plot([xpos, xpos, xpos + xlen], [ypos + ylen, ypos, ypos], **line_params_def)

    # invert labels if required:
    labels = [labels_dict[plane][i] for i in (int(switchlabels), 1 - switchlabels)]

    ax.text(
        xpos - xlen * spacing_coef,
        ypos + ylen,
        labels[0],
        ha="right",
        va="top",
        **text_params_def,
    )
    ax.text(
        xpos + xlen,
        ypos + ylen * spacing_coef,
        labels[1],
        ha="center",
        va="bottom",
        **text_params_def,
    )
    ax.text(
        xpos + xlen / 2,
        ypos - ylen * spacing_coef * 2,
        f"{length:3.0f} {units}",
        ha="center",
        va="top",
        **text_params_def,
    )

    if disable_axis:
        despine(ax, "all")

    if equalize_axis:
        ax.axis("equal")


def add_phase_cbar(*args, **kwargs):
    return add_cbar(
        *args,
        ticks=(-np.pi + 0.1, np.pi - 0.1),
        ticklabels=(["-π", "π"]),
        title="phase",
        **kwargs,
    )


def add_dff_cbar(
    *args,
    flims=1,
    labelsize=6,
    titlesize=8,
    ticklabels=["$-$", "$+$"],
    title="ΔF",
    **kwargs,
):
    try:
        _ = iter(flims)
    except TypeError:
        flims = (-flims, flims)
    return add_cbar(
        *args,
        ticks=flims,
        ticklabels=ticklabels,
        title=title,
        titlesize=titlesize,
        labelsize=labelsize,
        **kwargs,
    )

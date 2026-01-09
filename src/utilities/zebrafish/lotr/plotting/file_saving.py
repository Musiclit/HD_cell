from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from lotr.file_utils import get_figures_location, get_nb_name


def get_nb_figures_location():
    """Using the ipynbname library here could be a bit brittle - but let's give this
    a try.
    """
    notebook_name = get_nb_name()
    if notebook_name is None:
        return get_figures_location()

    fig_location = get_figures_location() / notebook_name
    fig_location.mkdir(exist_ok=True)

    return fig_location


def savefig(name, fig=None, folder=None, format="pdf"):
    """Function to centralize figure saving."""
    if fig is None:
        fig = plt.gcf()

    if folder is not None:
        name = name.split(".")[0]
        folder = get_figures_location() / folder
        if not folder.exists():
            folder.mkdir(exist_ok=True)
    else:
        if type(name) == str:
            folder = get_nb_figures_location()
            name = name.split(".")[0]
        else:  # assuming Path object here
            folder = name.parent
            name = name.stem  # remove format if specified

    fig.savefig(folder / f"{name}.{format}", dpi=300)


def save_multiplot_to_pdf(plot_func, args_list, filename, **kwargs):
    """Function to save a list of pdfs from a plotting function and a list of arguments.

    Parameters
    ----------
    plot_func : fun
        The function to be used for plotting. Should return the Figure obj.
    args_list : list
        List of arguments with which the plotting function will be called, once per page
    filename : str or Path
        Location where the plot will be saved.

    Returns
    -------

    """
    with PdfPages(filename) as pdf:
        for args in args_list:
            fig = plot_func(*args)
            pdf.savefig(fig, **kwargs)
            plt.close(fig)

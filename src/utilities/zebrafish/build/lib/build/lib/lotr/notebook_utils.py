from inspect import getsource

from IPython.core.display import HTML
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


def print_source(function):
    """For use inside an IPython notebook: given a module and a function,
    print the source code.
    Code from https://stackoverflow.com/questions/20665118/how-to-show-source-code-of-a-package-function-in-ipython-notebook
    """

    return HTML(highlight(getsource(function), PythonLexer(), HtmlFormatter(full=True)))

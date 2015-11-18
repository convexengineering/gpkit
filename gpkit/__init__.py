# -*- coding: utf-8 -*-
"""Lightweight GP Modeling Package

    For examples please see the examples folder.

    Requirements
    ------------
    numpy
    MOSEK or CVXOPT
    scipy(optional): for complete sparse matrix support
    sympy(optional): for latex printing in iPython Notebook

    Attributes
    ----------
    settings : dict
        Contains settings loaded from ``./env/settings``
"""

__version__ = "0.3.3"
UNIT_REGISTRY = None
SIGNOMIALS_ENABLED = False


def enable_units():
    """Enables units support in a particular instance of GPkit.

    Posynomials created after calling this are incompatible with those created
    before.

    If gpkit is imported multiple times, this needs to be run each time."""
    global units, DimensionalityError, UNIT_REGISTRY
    try:
        import pint
        if UNIT_REGISTRY is None:
            UNIT_REGISTRY = pint.UnitRegistry()
        units = UNIT_REGISTRY
        DimensionalityError = pint.DimensionalityError
    except ImportError:
        print("Optional Python units library (Pint) not installed;"
              " unit support disabled.")
        disable_units()


def disable_units():
    """Disables units support in a particular instance of GPkit.

    Posynomials created after calling this are incompatible with those created
    before.

    If gpkit is imported multiple times, this needs to be run each time.

    The correct way to call this is:
        import gpkit
        gpkit.disable_units()

    The following will *not* have the intended effect:
        from gpkit import disable_units
        disable_units()
    """
    global units, DimensionalityError

    class DummyUnits(object):
        "Dummy class to replace missing pint"
        class Quantity(object):
            pass

        def __nonzero__(self):
            return 0

        def __bool__(self):
            return False

        def __getattr__(self, attr):
            return 1

    units = DummyUnits()
    DimensionalityError = ValueError

enable_units()


class SignomialsEnabled(object):
    """Class to put up and tear down signomial support in an instance of GPkit.

    Example
    -------
    >>> import gpkit
    >>> x = gpkit.Variable("x")
    >>> y = gpkit.Variable("y", 0.1)
    >>> with SignomialsEnabled():
    >>>     constraints = [x >= 1-y]
    >>> gpkit.Model(x, constraints).localsolve()
    """

    def __enter__(self):
        global SIGNOMIALS_ENABLED
        SIGNOMIALS_ENABLED = True

    def __exit__(self, type_, val, traceback):
        global SIGNOMIALS_ENABLED
        SIGNOMIALS_ENABLED = False


from .nomials import Monomial, Posynomial, Signomial
from .variables import Variable, VectorVariable, ArrayVariable
from .geometric_program import GeometricProgram
from .signomial_program import SignomialProgram
from .varkey import VarKey
from .posyarray import PosyArray
from .model import Model
from .shortcuts import GP, SP

if units:
    def _subvert_pint():
        """
        When gpkit objects appear in mathematical operations with pint
        Quantity objects, let the gpkit implementations determine what to do
        """
        def skip_if_gpkit_objects(fallback, objects=(PosyArray, Signomial)):
            def _newfn(self, other):
                if isinstance(other, objects):
                    return NotImplemented
                else:
                    return getattr(self, fallback)(other)
            return _newfn

        for op in "eq ge le add mul div truediv floordiv".split():
            dunder = "__%s__" % op
            trunder = "___%s___" % op
            original = getattr(units.Quantity, dunder)
            setattr(units.Quantity, trunder, original)
            newfn = skip_if_gpkit_objects(fallback=trunder)
            setattr(units.Quantity, dunder, newfn)

# Load settings
from os import sep as os_sep
from os.path import dirname as os_path_dirname
SETTINGS_PATH = os_sep.join([os_path_dirname(__file__), "env", "settings"])
def load_settings(path=SETTINGS_PATH):
    try:
        with open(path) as settingsfile:
            lines = [line[:-1].split(" : ") for line in settingsfile
                     if len(line.split(" : ")) == 2]
            settings_ = {name: value.split(", ") for name, value in lines}
            for name, value in settings_.items():
                # hack to flatten 1-element lists,
                # unlesss they're the solver list
                if len(value) == 1 and name != "installed_solvers":
                    settings_[name] = value[0]
    except IOError:
        print("Could not load settings file.")
        settings_ = {"installed_solvers": [""]}
    return settings_
settings = load_settings()

# Load IPython CSS
try:
    from IPython.display import HTML, display

    class InvisibleHTML(HTML):
        "HTML which shows on import only if the viewer supports HTML"

        def __repr__(self):
            return ""

        def __str__(self):
            return ""

    display(InvisibleHTML("""<style>
    [style="font-size: 1.15em;"] { padding-top: 0.25em; }
    .widget-numeric-text { width: auto; }
    .widget-numeric-text .widget-label { width: 15ex; }
    .widget-numeric-text .form-control { background: #fbfbfb; width: 10ex; }
    .widget-slider .widget-label { width: 15ex; }
    .widget-checkbox .widget-label { width: 15ex; }
    .form-control { border: none; box-shadow: none; }
    </style>"""))
except:
    pass

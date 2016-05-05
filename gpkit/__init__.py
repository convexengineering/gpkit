"""GP and SP Modeling Package

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
from os import sep as os_sep
from os.path import dirname as os_path_dirname
SETTINGS_PATH = os_sep.join([os_path_dirname(__file__), "env", "settings"])

__version__ = "0.4.0"
UNIT_REGISTRY = None
SIGNOMIALS_ENABLED = False

# global variable initializations
DimensionalityError = ValueError
units = None


def enable_units(path=None):
    """Enables units support in a particular instance of GPkit.

    Posynomials created after calling this are incompatible with those created
    before.

    If gpkit is imported multiple times, this needs to be run each time."""
    # pylint: disable=invalid-name,global-statement
    global units, DimensionalityError, UNIT_REGISTRY
    try:
        import pint
        if path:
            # let user load their own unit definitions
            UNIT_REGISTRY = pint.UnitRegistry(path)
        if UNIT_REGISTRY is None:
            UNIT_REGISTRY = pint.UnitRegistry() # use pint default
            path = os_sep.join([os_path_dirname(__file__), "pint"])
            UNIT_REGISTRY.load_definitions(os_sep.join([path, "usd_cpi.txt"]))
            # next line patches https://github.com/hgrecco/pint/issues/366
            UNIT_REGISTRY.define("nautical_mile = 1852 m = nmi")
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
    global units  # pylint: disable=global-statement

    class DummyUnits(object):
        "Dummy class to replace missing pint"
        class Quantity(object):
            "Dummy Quantity instead of pint"
            pass

        def __nonzero__(self):
            return 0

        def __bool__(self):
            return False

        def __getattr__(self, attr):
            return 1

        def __call__(self, arg):
            return 1

    units = DummyUnits()

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
    # pylint: disable=global-statement
    def __enter__(self):
        global SIGNOMIALS_ENABLED
        SIGNOMIALS_ENABLED = True

    def __exit__(self, type_, val, traceback):
        global SIGNOMIALS_ENABLED
        SIGNOMIALS_ENABLED = False


# pylint: disable=wrong-import-position
from .varkey import VarKey
from .nomials import Nomial, NomialArray
from .nomials import Monomial, Posynomial, Signomial
from .nomials import Variable, VectorVariable, ArrayVariable
from .geometric_program import GeometricProgram
from .constraints.signomial_program import SignomialProgram
from .constraints.set import ConstraintSet
from .constraints.model import Model
from .constraints.linked import LinkedConstraintSet

if units:
    def _subvert_pint():
        """
        When gpkit objects appear in mathematical operations with pint
        Quantity objects, let the gpkit implementations determine what to do
        """
        def skip_if_gpkit_objects(fallback, objects=(Nomial, NomialArray)):
            """Returned method calls self.fallback(other) if other is
            not in objects, and otherwise returns NotImplemented.
            """
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

    _subvert_pint()


def load_settings(path=SETTINGS_PATH):
    """Load the settings file at SETTINGS_PATH; return settings dict"""
    try:
        with open(path) as settingsfile:
            lines = [line[:-1].split(" : ") for line in settingsfile
                     if len(line.split(" : ")) == 2]
            settings_ = {name: value.split(", ") for name, value in lines}
            for name, value in settings_.items():
                # hack to flatten 1-element lists,
                # unless they're the solver list
                if len(value) == 1 and name != "installed_solvers":
                    settings_[name] = value[0]
    except IOError:
        print("Could not load settings file.")
        settings_ = {"installed_solvers": [""]}
    settings_["default_solver"] = settings_["installed_solvers"][0]
    settings_["latex_modelname"] = True
    return settings_
settings = load_settings()

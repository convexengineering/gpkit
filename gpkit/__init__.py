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
from collections import defaultdict
from os import sep as os_sep
from os.path import dirname as os_path_dirname
SETTINGS_PATH = os_sep.join([os_path_dirname(__file__), "env", "settings"])

__version__ = "0.5.1"
UNIT_REGISTRY = None
SIGNOMIALS_ENABLED = False

# global variable initializations
DimensionalityError = ValueError
ureg, units = None, None  # pylint: disable=invalid-name


class GPkitUnits(object):
    "Return monomials instead of Quantitites"

    def __init__(self):
        self.Quantity = ureg.Quantity  # pylint: disable=invalid-name
        if hasattr(ureg, "__nonzero__"):
            # that is, if it's a DummyUnits object
            self.__nonzero__ = ureg.__nonzero__
            self.__bool__ = ureg.__bool__

    def __getattr__(self, attr):
        return Monomial(self.Quantity(1, getattr(ureg, attr)))

    def __call__(self, arg):
        return Monomial(self.Quantity(1, ureg(arg)))


def enable_units(path=None):
    """Enables units support in a particular instance of GPkit.

    Posynomials created after calling this are incompatible with those created
    before.

    If gpkit is imported multiple times, this needs to be run each time."""
    # pylint: disable=invalid-name,global-statement
    global DimensionalityError, UNIT_REGISTRY, ureg, units
    try:
        import pint
        if path:
            # let user load their own unit definitions
            UNIT_REGISTRY = pint.UnitRegistry(path)
        if UNIT_REGISTRY is None:
            UNIT_REGISTRY = pint.UnitRegistry()  # use pint default
            path = os_sep.join([os_path_dirname(__file__), "pint"])
            UNIT_REGISTRY.load_definitions(os_sep.join([path, "usd_cpi.txt"]))
            # next line patches https://github.com/hgrecco/pint/issues/366
            UNIT_REGISTRY.define("nautical_mile = 1852 m = nmi")

        ureg = UNIT_REGISTRY
        DimensionalityError = pint.DimensionalityError
        units = GPkitUnits()
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
    global ureg, units  # pylint: disable=invalid-name,global-statement

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

    ureg = DummyUnits()
    units = ureg

enable_units()

# the current vectorization shape
VECTORIZATION = []
# the current model hierarchy
MODELS = []
# modelnumbers corresponding to MODELS, above
MODELNUMS = []
# lookup table for the number of models of each name that have been made
MODELNUM_LOOKUP = defaultdict(int)
# the list of variables named in the current MODELS/MODELNUM environment
NAMEDVARS = defaultdict(list)


def begin_variable_naming(model):
    "Appends a model name and num to the environment."
    MODELS.append(model)
    num = MODELNUM_LOOKUP[model]
    MODELNUMS.append(num)
    MODELNUM_LOOKUP[model] += 1
    return num, (tuple(MODELS), tuple(MODELNUMS))


def end_variable_naming():
    "Pops a model name and num from the environment."
    NAMEDVARS.pop((tuple(MODELS), tuple(MODELNUMS)), None)
    MODELS.pop()
    MODELNUMS.pop()


class NamedVariables(object):
    """Creates an environment in which all variables have
       a model name and num appended to their varkeys.
    """
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        "Enters a named environment."
        begin_variable_naming(self.model)

    def __exit__(self, type_, val, traceback):
        "Leaves a named environment."
        end_variable_naming()


class Vectorize(object):
    """Creates an environment in which all variables are
       exended in an additional dimension.
    """
    def __init__(self, dimension_length):
        self.dimension_length = dimension_length

    def __enter__(self):
        "Enters a vectorized environment."
        VECTORIZATION.insert(0, self.dimension_length)

    def __exit__(self, type_, val, traceback):
        "Leaves a vectorized environment."
        VECTORIZATION.pop(0)


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
from .nomials import VectorVariable, ArrayVariable
# note: the Variable the user sees is not the Variable used internally
from .nomials import VectorizableVariable as Variable
from .constraints.geometric_program import GeometricProgram
from .constraints.signomial_program import SignomialProgram
from .constraints.sigeq import SignomialEquality
from .constraints.set import ConstraintSet
from .constraints.model import Model
from .constraints.linked import LinkedConstraintSet


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

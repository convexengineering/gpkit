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

__version__ = "0.3.0"


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


def enable_units():
    """Enables units support in a particular instance of GPkit.

    Posynomials created after calling this are incompatible with those created
    before.

    If gpkit is imported multiple times, this needs to be run each time."""
    global units, DimensionalityError
    try:
        import pint
        units = pint.UnitRegistry()
        DimensionalityError = pint.DimensionalityError
    except ImportError:
        print("Optional Python units library (Pint) not installed;"
              " unit support disabled.")
        disable_units()

enable_units()

SIGNOMIALS_ENABLED = False


class SignomialsEnabled(object):
    """Class to put up and tear down signomial support in an instance of GPkit.

    Example
    -------
    >>> import gpkit
    >>> x = gpkit.Variable("x")
    >>> y = gpkit.Variable("y", 0.1)
    >>> with enable_signomials():
    >>>     constraints = [x >= 1-y]
    >>> gpkit.Model(x, constraints).localsolve()
    """

    def __enter__(self):
        global SIGNOMIALS_ENABLED
        SIGNOMIALS_ENABLED = True

    def __exit__(self, type_, val, traceback):
        global SIGNOMIALS_ENABLED
        SIGNOMIALS_ENABLED = False


def enable_signomials():
    """Enables signomial support in a particular instance of GPkit."""
    global SIGNOMIALS_ENABLED
    SIGNOMIALS_ENABLED = True
    print("'enable_signomials()' has been replaced by 'SignomialsEnabled'"
          " in a 'with' statement (e.g. 'with SignomialsEnabled():  constraints"
          " = [1-x]'). enable_signomials() will be removed in the next point"
          " release, so please update your code!")


def disable_signomials():
    """Disables signomial support in a particular instance of GPkit."""
    global SIGNOMIALS_ENABLED
    SIGNOMIALS_ENABLED = False
    print("'disable_signomials()' has been replaced by 'SignomialsEnabled'"
          " in a 'with' statement (e.g. 'with SignomialsEnabled():  constraints"
          " = [1-x]'). enable_signomials() will be removed in the next point"
          " release, so please update your code!")


from .nomials import Monomial, Posynomial, Signomial
from .variables import Variable, VectorVariable, ArrayVariable
from .geometric_program import GeometricProgram
from .signomial_program import SignomialProgram
from .varkey import VarKey
from .posyarray import PosyArray
from .model import Model
from .shortcuts import GP, SP

if units:
    # regain control of Quantities' interactions with Posynomials
    Posynomial = Posynomial
    import operator

    def Qadd(self, other):
        if isinstance(other, (PosyArray, Signomial)):
            return NotImplemented
        return self._add_sub(other, operator.add)

    def Qmul(self, other):
        if isinstance(other, (PosyArray, Signomial)):
            return NotImplemented
        else:
            return self._mul_div(other, operator.mul)

    def Qtruediv(self, other):
        if isinstance(other, (PosyArray, Signomial)):
            return NotImplemented
        else:
            return self._mul_div(other, operator.truediv)

    def Qfloordiv(self, other):
        if isinstance(other, (PosyArray, Signomial)):
            return NotImplemented
        else:
            return self._mul_div(other, operator.floordiv,
                                 units_op=operator.truediv)

    for oper in ["eq"]:
        #TODO: this should all be abstractable like this, but fails on lambdas?
        fname = "__"+oper+"__"
        oldf = getattr(units.Quantity, fname)
        setattr(units.Quantity, "__"+fname, oldf)

        def newf(self, other):
            if isinstance(other, (PosyArray, Signomial)):
                return NotImplemented
            else:
                getattr(units.Quantity, "__"+fname)(self, other)

        setattr(units.Quantity, fname, newf)

    def Qle(self, other):
        if isinstance(other, (PosyArray, Signomial)):
            return NotImplemented
        else:
            return self.compare(other, op=operator.le)

    def Qge(self, other):
        if isinstance(other, (PosyArray, Signomial)):
            return NotImplemented
        else:
            return self.compare(other, op=operator.ge)

    units.Quantity.__add__ = Qadd
    units.Quantity.__mul__ = Qmul
    units.Quantity.__div__ = Qtruediv
    units.Quantity.__truediv__ = Qtruediv
    units.Quantity.__floordiv__ = Qfloordiv
    units.Quantity.__le__ = Qle
    units.Quantity.__ge__ = Qge

# Load settings
from os import sep as os_sep
from os.path import dirname as os_path_dirname
settings_path = os_sep.join([os_path_dirname(__file__), "env", "settings"])
try:
    with open(settings_path) as settingsfile:
        lines = [line[:-1].split(" : ") for line in settingsfile
                 if len(line.split(" : ")) == 2]
        settings = {name: value.split(", ") for name, value in lines}
        for name, value in settings.items():
            # hack to flatten 1-element lists, unlesss they're the solver list
            if len(value) == 1 and name != "installed_solvers":
                settings[name] = value[0]
    try:
        del lines
        del line
    except NameError:
        pass
except IOError:
    print("Could not load settings file.")
    settings = {"installed_solvers": [""]}

try:
    cfg = get_ipython().config
    if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
        import interactive
except NameError:
    pass

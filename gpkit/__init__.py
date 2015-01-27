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


def disableUnits():
    """Disables units support in a particular instance of GPkit.

    Posynomials created after this is run are incompatible with those created before.
    If gpkit is imported multiple times, this needs to be run each time."""
    global units, DimensionalityError

    class DummyUnits(object):
        "Dummy class to replace missing pint"
        class Quantity(object):
            pass

        def __nonzero__(self):
            return 0

    units = DummyUnits()
    DimensionalityError = ValueError


def enableUnits():
    """Enables units support in a particular instance of GPkit.

    Posynomials created after this is run are incompatible with those created before.
    If gpkit is imported multiple times, this needs to be run each time."""
    global units, DimensionalityError
    try:
        import pint
        units = pint.UnitRegistry()
        DimensionalityError = pint.DimensionalityError
    except ImportError:
        print "Optional Python units library (Pint) not installed; unit support disabled."
        disableUnits()

enableUnits()

from .nomials import Monomial, Posynomial, Variable, VectorVariable, VarKey
from .posyarray import PosyArray
from .geometric_program import GP

if units:
    # regain control of Quantities' interactions with Posynomials
    Posynomial = Posynomial
    import operator

    def Qadd(self, other):
        if isinstance(other, (PosyArray, Posynomial)):
            return NotImplemented
        return self._add_sub(other, operator.add)

    def Qmul(self, other):
        if isinstance(other, (PosyArray, Posynomial)):
            return NotImplemented
        else:
            return self._mul_div(other, operator.mul)

    def Qtruediv(self, other):
        if isinstance(other, (PosyArray, Posynomial)):
            return NotImplemented
        else:
            return self._mul_div(other, operator.truediv)

    def Qfloordiv(self, other):
        if isinstance(other, (PosyArray, Posynomial)):
            return NotImplemented
        else:
            return self._mul_div(other, operator.floordiv, units_op=operator.truediv)

    for oper in ["eq"]:
        #TODO: this should all be abstractable like this, but fails on lambdas?
        fname = "__"+oper+"__"
        oldf = getattr(units.Quantity, fname)
        setattr(units.Quantity, "__"+fname, oldf)

        def newf(self, other):
            if isinstance(other, (PosyArray, Posynomial)):
                return NotImplemented
            else:
                getattr(units.Quantity, "__"+fname)(self, other)

        setattr(units.Quantity, fname, newf)

    def Qle(self, other):
        if isinstance(other, (PosyArray, Posynomial)):
            return NotImplemented
        else:
            return self.compare(other, op=operator.le)

    def Qge(self, other):
        if isinstance(other, (PosyArray, Posynomial)):
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
        settings = {name: value.split(", ") for (name, value) in lines}
except IOError:
    print "Could not load settings file."
    settings = {"installed_solvers": [""]}

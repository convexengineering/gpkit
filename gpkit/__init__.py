# -*- coding: utf-8 -*-
"""Lightweight GP Modeling Package

    For examples please see the examples folder.

    Requirements
    ------------
    numpy
    MOSEK or CVXOPT
    scipy(optional): for full sparse matrix support
    sympy(optional): for latex printing in iPython Notebook

    Attributes
    ----------
    settings : dict
        Contains settings loaded from ``./env/settings``
"""

try:
    import pint
    units = pint.UnitRegistry()
    DimensionalityError = pint.DimensionalityError
except ImportError:
    print "Unable to load pint; unit support disabled."

    class Units(object):
        "Dummy class to replace missing pint"
        class Quantity(object): pass
        def __nonzero__(self): return 0

    units = Units()
    DimensionalityError = ValueError

from .nomials import Monomial, Posynomial, Variable
from .nomial_interfaces import mon, vecmon
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

    def Qeq(self, other):
        # We compare to the base class of Quantity because
        # each Quantity class is unique.
        if isinstance(other, (PosyArray, Posynomial)):
            return NotImplemented
        else:
            if not isinstance(other, _Quantity):
                return (self.dimensionless and
                        _eq(self._convert_magnitude(UnitsContainer()), other, False))

            if _eq(self._magnitude, 0, True) and _eq(other._magnitude, 0, True):
                return self.dimensionality == other.dimensionality

            if self._units == other._units:
                return _eq(self._magnitude, other._magnitude, False)

            try:
                return _eq(self.to(other).magnitude, other._magnitude, False)
            except DimensionalityError:
                return False

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
    units.Quantity.__eq__ = Qeq
    units.Quantity.__le__ = Qle
    units.Quantity.__ge__ = Qge

# Load settings
from os import sep as os_sep
from os.path import dirname as os_path_dirname
settings_path = os_sep.join([os_path_dirname(__file__), "env", "settings"])
with open(settings_path) as settingsfile:
    lines = [line[:-1].split(" : ") for line in settingsfile
             if len(line.split(" : ")) == 2]
    settings = {name: value.split(", ") for (name, value) in lines}

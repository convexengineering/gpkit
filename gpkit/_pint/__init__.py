"wraps pint in gpkit monomials"
from __future__ import unicode_literals
import os
try:
    from pint import UnitRegistry, DimensionalityError
    ureg = UnitRegistry()  # pylint: disable=invalid-name
    ureg.load_definitions(os.sep.join([os.path.dirname(__file__),
                                       "usd_cpi.txt"]))
    Quantity = ureg.Quantity
except ImportError:  # pint is not installed; provide dummy imports
    ureg = None  # pylint: disable=invalid-name
    Quantity = lambda a, b: None
    DimensionalityError = None

QTY_CACHE = {}


def qty(unit):
    "Returns a Quantity, caching the result for future retrievals"
    if unit not in QTY_CACHE:
        QTY_CACHE[unit] = Quantity(1, unit)
    return QTY_CACHE[unit]


class GPkitUnits(object):
    "Return Monomials instead of Quantitites"
    division_cache = {}
    multiplication_cache = {}
    monomial_cache = {}

    def __call__(self, arg):
        "Returns a unit Monomial, caching the result for future retrievals"
        from .. import Monomial
        if arg not in self.monomial_cache:
            self.monomial_cache[arg] = Monomial(qty(arg))
        return self.monomial_cache[arg]

    def __getattr__(self, attr):
        "Turns an attribute get into a function call"
        return self(attr)

    def of_division(self, numerator, denominator):
        "Cached unit division. Requires Quantity inputs."
        if numerator.units is denominator.units:
            return 1
        key = (id(numerator.units), id(denominator.units))
        if key not in self.division_cache:
            if numerator.units and denominator.units:
                conversion = numerator.units/denominator.units
            else:
                conversion = numerator.units or 1/denominator.units
            try:
                self.division_cache[key] = float(conversion)
            except DimensionalityError:
                raise DimensionalityError(numerator, denominator)
        return self.division_cache[key]

    def of_product(self, thing1, thing2):
        "Cached unit division. Requires united inputs."
        # TODO: qty shouldn't be necessary below
        mul_units = qty((thing1*thing2).units)
        key = id(mul_units)
        if key not in self.multiplication_cache:
            try:
                self.multiplication_cache[key] = (None, float(mul_units))
            except DimensionalityError:
                self.multiplication_cache[key] = (mul_units, None)
        return self.multiplication_cache[key]


units = GPkitUnits()

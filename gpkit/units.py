"wraps pint in gpkit monomials"
import pint
ureg = pint.UnitRegistry()  # pylint: disable=invalid-name
ureg.define("USD = [money] = $")
pint.set_application_registry(ureg)
Quantity = ureg.Quantity
DimensionalityError = pint.DimensionalityError
QTY_CACHE = {}


def qty(unit):
    "Returns a Quantity, caching the result for future retrievals"
    if unit not in QTY_CACHE:
        QTY_CACHE[unit] = Quantity(1, unit)
    return QTY_CACHE[unit]


class GPkitUnits:
    "Return Monomials instead of Quantitites"
    division_cache = {}
    multiplication_cache = {}
    monomial_cache = {}

    def __call__(self, unity):
        "Returns a unit Monomial, caching the result for future retrievals"
        from . import Monomial
        if unity not in self.monomial_cache:
            self.monomial_cache[unity] = Monomial(qty(unity))
        return self.monomial_cache[unity]

    __getattr__ = __call__

    def of_division(self, numerator, denominator):
        "Cached unit division. Requires Quantity inputs."
        if numerator.units is denominator.units:
            return 1
        key = (id(numerator.units), id(denominator.units))
        try:
            return self.division_cache[key]
        except KeyError:
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
        key = (thing1.units, thing2.units)
        try:
            return self.multiplication_cache[key]
        except KeyError:
            mul_units = qty((thing1*thing2).units)
            try:
                self.multiplication_cache[key] = (None, float(mul_units))
            except DimensionalityError:
                self.multiplication_cache[key] = (mul_units, None)
        return self.multiplication_cache[key]


units = GPkitUnits()

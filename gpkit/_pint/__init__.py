"wraps pint in gpkit monomials"
import os
try:
    from pint import UnitRegistry, DimensionalityError

    ureg = UnitRegistry()  # pylint: disable=invalid-name
    ureg.load_definitions(os.sep.join([os.path.dirname(__file__),
                                       "usd_cpi.txt"]))
    # next line patches https://github.com/hgrecco/pint/issues/366
    ureg.define("nautical_mile = 1852 m = nmi")
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
    "Return monomials instead of Quantitites"

    def __call__(self, arg):
        from .. import Monomial
        return Monomial(qty(arg))

    def __getattr__(self, attr):
        return self(attr)


units = GPkitUnits()

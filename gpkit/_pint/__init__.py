"wraps pint in gpkit monomials"
import os
from pint import UnitRegistry, DimensionalityError

ureg = UnitRegistry()  # pylint: disable=invalid-name
ureg.load_definitions(os.sep.join([os.path.dirname(__file__), "usd_cpi.txt"]))
# next line patches https://github.com/hgrecco/pint/issues/366
ureg.define("nautical_mile = 1852 m = nmi")


class GPkitUnits(object):
    "Return monomials instead of Quantitites"

    def __init__(self):
        self.Quantity = ureg.Quantity  # pylint: disable=invalid-name
        if hasattr(ureg, "__nonzero__"):
            # that is, if it's a DummyUnits object
            self.__nonzero__ = ureg.__nonzero__
            self.__bool__ = ureg.__bool__

    def __getattr__(self, attr):
        from .. import Monomial
        return Monomial(self.Quantity(1, getattr(ureg, attr)))

    def __call__(self, arg):
        from .. import Monomial
        return Monomial(self.Quantity(1, ureg(arg)))


units = GPkitUnits()

"The shared non-mathematical backbone of all Nomials"
from .data import NomialData
from ..small_classes import Numbers
from ..small_scripts import nomial_latex_helper
from ..repr_conventions import _str, _repr, _repr_latex_, unitstr


class Nomial(NomialData):
    "Shared non-mathematical properties of all nomials"
    __div__ = None
    sub = None

    __str__ = _str
    __repr__ = _repr
    _repr_latex_ = _repr_latex_
    unitstr = unitstr

    def str_without(self, excluded=None):
        "String representation, excluding fields ('units', varkey attributes)"
        if excluded is None:
            excluded = []
        mstrs = []
        for exp, c in self.hmap.items():
            varstrs = []
            for (var, x) in exp.items():
                if x != 0:
                    varstr = var.str_without(excluded)
                    if x != 1:
                        varstr += "**%.2g" % x
                    varstrs.append(varstr)
            varstrs.sort()
            cstr = "%.3g" % c
            if cstr == "-1" and varstrs:
                mstrs.append("-" + "*".join(varstrs))
            else:
                cstr = [cstr] if (cstr != "1" or not varstrs) else []
                mstrs.append("*".join(cstr + varstrs))
        if "units" not in excluded:
            units = self.unitstr(" [%s]")
        else:
            units = ""
        return " + ".join(sorted(mstrs)) + units

    def latex(self, excluded=None):
        "Latex representation, parsing `excluded` just as .str_without does"
        if excluded is None:
            excluded = []
        mstrs = []
        for exp, c in self.hmap.items():
            pos_vars, neg_vars = [], []
            for var, x in exp.items():
                if x > 0:
                    pos_vars.append((var.latex(excluded), x))
                elif x < 0:
                    neg_vars.append((var.latex(excluded), x))
            mstrs.append(nomial_latex_helper(c, pos_vars, neg_vars))

        if "units" in excluded:
            return " + ".join(sorted(mstrs))

        units = self.unitstr(r"\mathrm{~\left[ %s \right]}", ":L~")
        units_tf = units.replace("frac", "tfrac").replace(r"\cdot", r"\cdot ")
        return " + ".join(sorted(mstrs)) + units_tf

    __hash__ = NomialData.__hash__  # required by Python 3

    @property
    def value(self):
        """Self, with values substituted for variables that have values

        Returns
        -------
        float, if no symbolic variables remain after substitution
        (Monomial, Posynomial, or Nomial), otherwise.
        """
        p = self.sub(self.values)  # pylint: disable=not-callable
        if len(p.hmap) == 1 and not p.vks:
            return p.c
        return p

    def prod(self):
        "Return self for compatibility with NomialArray"
        return self

    def sum(self):
        "Return self for compatibility with NomialArray"
        return self

    def to(self, units):
        "Create new Signomial converted to new units"
         # pylint: disable=no-member
        return self.__class__(self.hmap.to(units))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        "True if self and other are algbraically identical."
        if isinstance(other, Numbers):
            return (len(self.hmap) == 1 and  # single term
                    not self.vks and         # constant
                    self.cs[0] == other)     # the right constant
        return super(Nomial, self).__eq__(other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        "For the / operator in Python 3.x"
        return self.__div__(other)   # pylint: disable=not-callable

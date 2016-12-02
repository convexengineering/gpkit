"The shared non-mathematical backbone of all Nomials"
from .data import NomialData
from ..small_classes import Numbers
from ..small_scripts import nomial_latex_helper
from ..small_scripts import mag, unitstr
from ..repr_conventions import _str, _repr, _repr_latex_


class Nomial(NomialData):
    "Shared non-mathematical properties of all nomials"
    __div__ = None
    sub = None
    c = None

    __str__ = _str
    __repr__ = _repr
    _repr_latex_ = _repr_latex_

    def str_without(self, excluded=None):
        "String representation excluding fields ('units', varkey attributes)"
        if excluded is None:
            excluded = []
        mstrs = []
        for c, exp in zip(self.cs, self.exps):
            varstrs = []
            for (var, x) in exp.items():
                if x != 0:
                    varstr = var.str_without(excluded)
                    if x != 1:
                        varstr += "**%.2g" % x
                    varstrs.append(varstr)
            varstrs.sort()
            c = mag(c)
            cstr = "%.3g" % c
            if cstr == "-1" and varstrs:
                mstrs.append("-" + "*".join(varstrs))
            else:
                cstr = [cstr] if (cstr != "1" or not varstrs) else []
                mstrs.append("*".join(cstr + varstrs))
        if "units" not in excluded:
            units = unitstr(self.units, " [%s]")
        else:
            units = ""
        return " + ".join(sorted(mstrs)) + units

    def latex(self, excluded=None):
        "For pretty printing with Sympy"
        if excluded is None:
            excluded = []
        mstrs = []
        for c, exp in zip(self.cs, self.exps):
            pos_vars, neg_vars = [], []
            for var, x in exp.items():
                if x > 0:
                    pos_vars.append((var.latex(excluded), x))
                elif x < 0:
                    neg_vars.append((var.latex(excluded), x))

            mstrs.append(nomial_latex_helper(c, pos_vars, neg_vars))

        if "units" in excluded:
            return " + ".join(sorted(mstrs))

        units = unitstr(self.units, r"\mathrm{~\left[ %s \right]}", "L~")
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
        if len(p.exps) == 1:
            if not p.exp:
                return p.c
        return p

    def to(self, arg):
        "Create new Signomial converted to new units"
         # pylint: disable=no-member
        return self.__class__(self.exps, self.cs.to(arg).tolist())

    def convert_to(self, arg):
        "Convert this signomial to new units"
        self.cs = self.cs.to(arg)   # pylint: disable=no-member

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        """Equality test

        Returns
        -------
        bool
        """
        if isinstance(other, Numbers):
            return (len(self.exps) == 1 and  # single term
                    not self.exps[0] and     # constant
                    self.cs[0] == other)     # the right constant
        return super(Nomial, self).__eq__(other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        """Support the / operator in Python 3.x"""
        return self.__div__(other)   # pylint: disable=not-callable

"The shared non-mathematical backbone of all Nomials"
from __future__ import unicode_literals, print_function
from .data import NomialData
from ..small_classes import Numbers, FixedScalar
from ..small_scripts import nomial_latex_helper


class Nomial(NomialData):
    "Shared non-mathematical properties of all nomials"
    __div__ = None
    sub = None

    def str_without(self, excluded=()):
        "String representation, excluding fields ('units', varkey attributes)"
        units = "" if "units" in excluded else self.unitstr(" [%s]")
        if hasattr(self, "key"):
            return self.key.str_without(excluded) + units  # pylint: disable=no-member
        elif self.ast:
            return self.parse_ast(excluded) + units
        else:
            mstrs = []
            for exp, c in self.hmap.items():
                varstrs = []
                for (var, x) in exp.items():
                    if x != 0:
                        varstr = var.str_without(excluded)
                        if x != 1:
                            varstr += "^%.2g" % x
                        varstrs.append(varstr)
                varstrs.sort()
                cstr = "%.3g" % c
                if cstr == "-1" and varstrs:
                    mstrs.append("-" + "*".join(varstrs))
                else:
                    cstr = [cstr] if (cstr != "1" or not varstrs) else []
                    mstrs.append("*".join(cstr + varstrs))
        return " + ".join(sorted(mstrs)) + units

    def latex(self, excluded=()):
        "Latex representation, parsing `excluded` just as .str_without does"
        # TODO: add ast parsing here
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

    def prod(self):
        "Return self for compatibility with NomialArray"
        return self

    def sum(self):
        "Return self for compatibility with NomialArray"
        return self

    @property
    def value(self):
        """Self, with values substituted for variables that have values

        Returns
        -------
        float, if no symbolic variables remain after substitution
        (Monomial, Posynomial, or Nomial), otherwise.
        """
        if isinstance(self, FixedScalar):
            return self.cs[0]
        p = self.sub(self.varkeyvalues())  # pylint: disable=not-callable
        return p.cs[0] if isinstance(p, FixedScalar) else p

    def __eq__(self, other):
        "True if self and other are algebraically identical."
        if isinstance(other, Numbers):
            return isinstance(self, FixedScalar) and self.value == other
        return super(Nomial, self).__eq__(other)

    # pylint: disable=multiple-statements
    def __ne__(self, other): return not Nomial.__eq__(self, other)

    # required by Python 3
    __hash__ = NomialData.__hash__
    def __truediv__(self, other): return self.__div__(other)   # pylint: disable=not-callable

    # for arithmetic consistency
    def __radd__(self, other): return self.__add__(other, rev=True)   # pylint: disable=no-member
    def __rmul__(self, other): return self.__mul__(other, rev=True)   # pylint: disable=no-member

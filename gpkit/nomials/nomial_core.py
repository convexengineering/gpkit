"The shared non-mathematical backbone of all Nomials"
from .data import NomialData
from ..small_classes import Numbers
from ..small_scripts import latex_num
from ..small_scripts import mag, unitstr
from ..repr_conventions import _str, _repr, _repr_latex_


def fast_monomial_str(exp, c):
    "Quickly generates a unitless monomial string."
    varstrs = []
    for (var, x) in exp.items():
        if x != 0:
            varstr = str(var)
            if x != 1:
                varstr += "**%.2g" % x
            varstrs.append(varstr)
    c = mag(c)
    cstr = "%.3g" % c
    if cstr == "-1" and varstrs:
        return "-" + "*".join(varstrs)
    else:
        cstr = [cstr] if (cstr != "1" or not varstrs) else []
        return "*".join(cstr + varstrs)


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
        # pylint: disable=too-many-locals
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

            pvarstrs = ['%s^{%.2g}' % (varl, x) if "%.2g" % x != "1" else varl
                        for (varl, x) in pos_vars]
            nvarstrs = ['%s^{%.2g}' % (varl, -x)
                        if "%.2g" % -x != "1" else varl
                        for (varl, x) in neg_vars]
            pvarstrs.sort()
            nvarstrs.sort()
            pvarstr = ' '.join(pvarstrs)
            nvarstr = ' '.join(nvarstrs)
            c = mag(c)
            cstr = "%.2g" % c
            if pos_vars and (cstr == "1" or cstr == "-1"):
                cstr = cstr[:-1]
            else:
                cstr = latex_num(c)

            if not pos_vars and not neg_vars:
                mstrs.append("%s" % cstr)
            elif pos_vars and not neg_vars:
                mstrs.append("%s%s" % (cstr, pvarstr))
            elif neg_vars and not pos_vars:
                mstrs.append("\\frac{%s}{%s}" % (cstr, nvarstr))
            elif pos_vars and neg_vars:
                mstrs.append("%s\\frac{%s}{%s}" % (cstr, pvarstr, nvarstr))

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

    def prod(self):
        "base case: Product of a Nomial is itself"
        return self

    def sum(self):
        "base case: Sum of a Nomial is itself"
        return self

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

    def __float__(self):
        if len(self.exps) == 1:
            if not self.exps[0]:
                return mag(self.c)
        else:
            raise AttributeError("float() can only be called on"
                                 " monomials with no variable terms")

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        """Support the / operator in Python 3.x"""
        return self.__div__(other)   # pylint: disable=not-callable

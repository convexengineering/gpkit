"The shared non-mathematical backbone of all Nomials"
from .data import NomialData
from ..small_classes import Numbers
from ..small_scripts import latex_num
from ..small_scripts import mag, unitstr
from ..repr_conventions import _str, _repr, _repr_latex_


class Nomial(NomialData):
    "Shared non-mathematical properties of all nomials"

    __str__ = _str
    __repr__ = _repr
    _repr_latex_ = _repr_latex_

    def str_without(self, excluded=[]):
        mstrs = []
        for c, exp in zip(self.cs, self.exps):
            varstrs = []
            for (var, x) in exp.items():
                if x != 0:
                    varstr = var.str_without(*excluded)
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
        showunits = "units" not in excluded
        units = unitstr(self.units, " [%s]") if showunits else ""
        return " + ".join(sorted(mstrs)) + units

    def latex(self, showunits=True):
        "For pretty printing with Sympy"
        mstrs = []
        for c, exp in zip(self.cs, self.exps):
            pos_vars, neg_vars = [], []
            for var, x in exp.items():
                if x > 0:
                    pos_vars.append((var.latex(), x))
                elif x < 0:
                    neg_vars.append((var.latex(), x))

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

        if not showunits:
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
        p = self.sub(self.values)
        if len(p.exps) == 1:
            if not p.exp:
                return p.c
        return p

    def prod(self):
        return self

    def sum(self):
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
        return self.__div__(other)

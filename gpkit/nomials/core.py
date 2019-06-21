# -*- coding: utf-8 -*-
"The shared non-mathematical backbone of all Nomials"
import numpy as np
from .data import NomialData
from ..small_classes import Numbers, FixedScalar
from ..small_scripts import nomial_latex_helper, try_str_without


class Nomial(NomialData):
    "Shared non-mathematical properties of all nomials"
    __div__ = None
    sub = None
    aststr = None

    def str_without(self, excluded=()):
        "String representation, excluding fields ('units', varkey attributes)"
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
        if "units" not in excluded:
            units = self.unitstr(" [%s]")
        else:
            units = ""
        if self.ast:
            aststr = None
            oper, values = self.ast
            values_ = []
            for val in values:
                excluded = set(excluded)
                excluded.add("units")
                if isinstance(val, Numbers):
                    if val > np.pi/12 and val < 100*np.pi and abs(12*val/np.pi % 1) <= 1e-2:
                        if val > 3.1:
                            val = "%.3gPI" % (val/np.pi)
                            if val == "1PI":
                                val = "PI"
                        else:
                            val = "(PI/%.3g)" % (np.pi/val)
                    else:
                        val = "%.3g" % val
                values_.append(val)
            left, right = tuple(values_)
            if oper == "add":
                left = try_str_without(left, excluded)
                right = try_str_without(right, excluded)
                if right[0] == "-":
                    aststr = "%s - %s" % (left, right[1:])
                else:
                    aststr = "%s + %s" % (left, right)
            elif oper == "mul":
                if left == "1":
                    aststr = try_str_without(right, excluded)
                elif right == "1":
                    aststr = try_str_without(left, excluded)
                else:
                    if len(getattr(left, "hmap", [])) > 1:
                        left = "(%s)" % try_str_without(left, excluded)
                    if len(getattr(right, "hmap", [])) > 1:
                        right = "(%s)" % try_str_without(right, excluded)
                    left = try_str_without(left, excluded)
                    right = try_str_without(right, excluded)
                    aststr = "%s*%s" % (left, right)
            elif oper == "div":
                left = try_str_without(left, excluded)
                right = try_str_without(right, excluded)
                if right == "1":
                    aststr = left
                else:
                    aststr = "%s/%s" % (left, right)
            elif oper == "neg":
                aststr = "-%s" % try_str_without(left, excluded)
            else:
                raise ValueError(oper)
            return aststr + units
        return " + ".join(sorted(mstrs)) + units

    def latex(self, excluded=()):
        "Latex representation, parsing `excluded` just as .str_without does"
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

    def to(self, units):
        "Create new Signomial converted to new units"
        return self.__class__(self.hmap.to(units))  # pylint: disable=no-member

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
    def __radd__(self, other): return self.__add__(other, rev=True)
    def __rmul__(self, other): return self.__mul__(other, rev=True)

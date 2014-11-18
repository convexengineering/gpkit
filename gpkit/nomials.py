from collections import defaultdict
import itertools
import numpy as np
from types import NoneType

from .small_classes import Strings, Numbers
from .posyarray import PosyArray

from .small_scripts import latex_num
from .small_scripts import sort_and_simplify
from .small_scripts import locate_vars
from .small_scripts import invalid_types_for_oper
from .small_scripts import mag, unitstr

from pint import DimensionalityError
from . import units as ureg
Quantity = ureg.Quantity
Numbers += (Quantity,)


class Variable(object):
    """A key that Monomial and Posynomial exp dicts can be indexed by.

    Parameters
    ----------
    k : object (usually str)
        The variable's name attribute is derived from str(k).
    **kwargs
        Any additional attributes, which become the descr attribute (a dict).

    Returns
    -------
    Variable with the given name and descr.
    """
    new_unnamed_id = itertools.count().next

    def __init__(self, k=None, **kwargs):
        if isinstance(k, Variable):
            self.name = k.name
            self.descr = k.descr
        elif isinstance(k, Monomial):
            if mag(k.c) == 1 and len(k.exp) == 1:
                var = k.exp.keys()[0]
                self.name = var.name
                self.descr = var.descr
            else:
                raise TypeError("variables can only be formed from monomials"
                                " with a c of 1 and a single variable")
        else:
            if "name" in kwargs:
                k = kwargs["name"]
            elif k is None:
                k = "\\fbox{%s}" % Variable.new_unnamed_id()
            self.name = str(k)
            self.descr = dict(kwargs)
            self.descr["name"] = self.name

        if "value" in self.descr:
            value = self.descr["value"]
            if isinstance(value, Quantity):
                self.descr["value"] = value.magnitude
                self.descr["units"] = value/value.magnitude
        if ureg and "units" in self.descr:
            units = self.descr["units"]
            if isinstance(units, Strings):
                units = units.replace("-", "dimensionless")
                self.descr["units"] = ureg.parse_expression(units)
            elif isinstance(units, Quantity):
                self.descr["units"] = units/units.magnitude
            else:
                raise ValueError("units must be either a string"
                                 " or a Quantity from gpkit.units.")
        self.units = self.descr.get("units", None)
        self._hashvalue = hash(str(self))

    def __repr__(self):
        s = self.name
        for subscript in ["idx", "model"]:
            if subscript in self.descr:
                s = "%s_%s" % (s, self.descr[subscript])
        return s

    def _latex(self):
        s = self.name
        for subscript in ["idx", "model"]:
            if subscript in self.descr:
                s = "{%s}_{%s}" % (s, self.descr[subscript])
        return s

    def __hash__(self):
        return self._hashvalue

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.descr == other.descr
        elif isinstance(other, Strings):
            return str(self) == other
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class Posynomial(object):
    """A representation of a posynomial.

        Parameters
        ----------
        exps: tuple of dicts
            Exponent dicts for each monomial term
        cs: tuple
            Coefficient values for each monomial term
        var_locs: dict
            mapping from variable name to list of indices of monomial terms
            that variable appears in

        Returns
        -------
        Posynomial (if the input has multiple terms)
        Monomial   (if the input has one term)
    """
    def __init__(self, exps=None, cs=1, var_locs=None,
                 allow_negative=False, **descr):
        units = None
        if isinstance(exps, Numbers):
            cs = exps
            exps = {}
        if (isinstance(cs, Numbers)
           and isinstance(exps, Strings + (Variable, NoneType, dict))):
            # building a Monomial
            if isinstance(exps, Variable):
                exp = {exps: 1}
                units = exps.descr["units"] if "units" in exps.descr else None
            elif exps is None or isinstance(exps, Strings):
                exp = {Variable(exps, **descr): 1}
                descr = exp.keys()[0].descr
                units = descr["units"] if "units" in descr else None
            elif isinstance(exps, dict):
                exp = dict(exps)
                for key in exps:
                    if isinstance(key, Strings):
                        exp[Variable(key)] = exp.pop(key)
            else:
                raise TypeError("could not make Monomial with %s" % type(exps))
            if isinstance(units, Quantity):
                cs = cs * units
            cs = [cs]
            exps = [exp]
        elif isinstance(exps, Posynomial):
            cs = exps.cs
            var_locs = exps.var_locs
            exps = exps.exps
        else:
            # test for presence of length and identical lengths
            try:
                assert len(cs) == len(exps)
                exps_ = range(len(exps))
                if not isinstance(cs[0], Quantity):
                    try:
                        cs = np.array(cs, dtype='float')
                    except ValueError:
                        raise ValueError("cannot add dimensioned and"
                                         " dimensionless monomials together.")
                else:
                    units = cs[0]/cs[0].magnitude
                    if units == ureg.dimensionless:
                        cs = [c * ureg.dimensionless for c in cs]
                    cs = [c.to(units).magnitude for c in cs] * units
                    if not all([c.dimensionality == units.dimensionality
                                for c in cs]):
                        raise ValueError("cannot add monomials of"
                                         " different units together")
                for i in range(len(exps)):
                    exps_[i] = dict(exps[i])
                    for key in exps_[i]:
                        if isinstance(key, Strings+(Monomial,)):
                            exps_[i][Variable(key)] = exps_[i].pop(key)
                exps = exps_
            except AssertionError:
                raise TypeError("cs and exps must have the same length.")

        exps, cs = sort_and_simplify(exps, cs)
        if not allow_negative:
            if isinstance(cs, Quantity):
                any_negative = any((c.magnitude <= 0 for c in cs))
            else:
                any_negative = any((c <= 0 for c in cs))
            if any_negative:
                raise ValueError("each c must be positive.")

        self.cs = cs
        self.exps = exps
        self.units = cs[0]/cs[0].magnitude if isinstance(cs[0], Quantity) else None
        if len(exps) == 1:
            if self.__class__ is Posynomial:
                self.__class__ = Monomial
            self.exp = exps[0]
            self.c = cs[0]

        if var_locs is None:
            self.var_locs = locate_vars(exps)
        else:
            self.var_locs = var_locs

        self._hashvalue = hash(tuple(zip(self.exps, tuple(self.cs))))

    def to(self, arg):
        return Posynomial(self.exps, self.cs.to(arg).tolist())

    def sub(self, substitutions, val=None, allow_negative=False):
        var_locs, exps, cs, subs = substitution(self.var_locs,
                                                self.exps, self.cs,
                                                substitutions, val)
        return Posynomial(exps, cs, var_locs, allow_negative)

    # hashing, immutability, Posynomial inequality
    def __hash__(self):
        return self._hashvalue

    def __ne__(self, other):
        if isinstance(other, Posynomial):
            return not (self.exps == other.exps and self.cs == other.cs)
        else:
            return False

    # constraint generation
    def __eq__(self, other):
        # if at least one is a monomial, return a constraint
        mons = Numbers+(Monomial,)
        if isinstance(other, mons) and isinstance(self, mons):
            return MonoEQConstraint(self, other)
        elif isinstance(other, Posynomial) and isinstance(self, Posynomial):
            if self.exps == other.exps:
                if isinstance(self.cs, Quantity):
                    return all(self.cs.magnitude <= other.cs)
                else:
                    return all(self.cs <= other.cs)
            else:
                return False
        else:
            return False

    def __le__(self, other):
        return Constraint(self, other)

    def __ge__(self, other):
        return Constraint(other, self)

    def __lt__(self, other):
        invalid_types_for_oper("<", self, other)

    def __gt__(self, other):
        invalid_types_for_oper(">", self, other)

    def __str__(self, mult_symbol='*'):
        mstrs = []
        for c, exp in zip(self.cs, self.exps):
            varstrs = ['%s**%.2g' % (var, x) if x != 1 else "%s" % var
                       for (var, x) in sorted(exp.items()) if x != 0]
            c = mag(c)
            cstr = ["%.2g" % c] if c != 1 or not varstrs else []
            mstrs.append(mult_symbol.join(cstr + varstrs))
        return " + ".join(sorted(mstrs)) + unitstr(self.units, ", units='%s'")

    def descr(self, descr):
        self.descr = descr
        return self

    def __repr__(self):
        return "gpkit.%s(%s)" % (self.__class__.__name__, str(self))

    def _latex(self, unused=None):
        "For pretty printing with Sympy"
        mstrs = []
        for c, exp in zip(self.cs, self.exps):
            pos_vars, neg_vars = [], []
            for var, x in sorted(exp.items()):
                if x > 0:
                    pos_vars.append((var._latex(), x))
                elif x < 0:
                    neg_vars.append((var._latex(), x))

            pvarstrs = ['%s^{%.2g}' % (varl, x) if "%.2g" % x != "1" else varl
                        for (varl, x) in pos_vars]
            nvarstrs = ['%s^{%.2g}' % (varl, -x)
                        if "%.2g" % -x != "1" else varl
                        for (varl, x) in neg_vars]
            pvarstr = ' '.join(pvarstrs)
            nvarstr = ' '.join(nvarstrs)
            c = mag(c)
            if pos_vars and c == 1:
                cstr = ""
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

        units = unitstr(self.units, "\mathrm{\\left[ %s \\right]}", "L~")
        units_smallfrac = units.replace("frac", "tfrac")
        return " + ".join(sorted(mstrs)) + units_smallfrac

    # posynomial arithmetic
    def __add__(self, other):
        if isinstance(other, Numbers):
            if other == 0:
                return Posynomial(self.exps, self.cs, self.var_locs)
            else:
                return Posynomial(self.exps + ({},), self.cs.tolist() + [other],
                                  self.var_locs)
        elif isinstance(other, Posynomial):
            return Posynomial(self.exps + other.exps, self.cs.tolist() + other.cs.tolist())
            # TODO: automatically parse var_locs here
        elif isinstance(other, PosyArray):
            return np.array(self)+other
        else:
            invalid_types_for_oper("+", self, other)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, Numbers):
            return Posynomial(self.exps,
                              other*self.cs,
                              self.var_locs)
        elif isinstance(other, Posynomial):
            C = np.outer(self.cs, other.cs)
            if isinstance(self.cs, Quantity) or isinstance(other.cs, Quantity):
                if not isinstance(self.cs, Quantity):
                    sunits = ureg.dimensionless
                else:
                    sunits = self.cs[0]/self.cs[0].magnitude
                if not isinstance(other.cs, Quantity):
                    ounits = ureg.dimensionless
                else:
                    ounits = other.cs[0]/other.cs[0].magnitude
                # hack fix for pint not working with np.outer
                C = C * sunits * ounits
            Exps = np.empty((len(self.exps), len(other.exps)), dtype="object")
            for i, exp_s in enumerate(self.exps):
                for j, exp_o in enumerate(other.exps):
                    Exps[i, j] = exp_s + exp_o
            return Posynomial(Exps.flatten(), C.flatten())
        elif isinstance(other, PosyArray):
            return np.array(self)*other
        else:
            invalid_types_for_oper("*", self, other)

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        if isinstance(other, Numbers):
            return Posynomial(self.exps,
                              self.cs/other,
                              self.var_locs)
        elif isinstance(other, Monomial):
            exps = [exp - other.exp for exp in self.exps]
            return Posynomial(exps, self.cs/other.c)
        elif isinstance(other, PosyArray):
            return np.array(self)/other
        else:
            invalid_types_for_oper("/", self, other)

    def __pow__(self, x):
        if isinstance(x, int):
            if x >= 0:
                p = Monomial({}, 1)
                while x > 0:
                    p *= self
                    x -= 1
                return p
            else:
                raise ValueError("Posynomials are only closed under"
                                 " nonnegative integer exponents.")
        else:
            invalid_types_for_oper("** or pow()", self, x)


class Monomial(Posynomial):

    def __rdiv__(self, other):
        if isinstance(other, Numbers+(Posynomial,)):
            return other * self**-1
        else:
            invalid_types_for_oper("/", other, self)

    def __pow__(self, other):
        if isinstance(other, Numbers):
            return Monomial(self.exp*other, self.c**other)
        else:
            invalid_types_for_oper("** or pow()", self, x)


class Constraint(Posynomial):

    def _set_operator(self, p1, p2):
        if self.left is p1:
            self.oper_s = " <= "
            self.oper_l = " \\leq "
        else:
            self.oper_s = " >= "
            self.oper_l = " \\geq "

    def __str__(self):
        return str(self.left) + self.oper_s + str(self.right)

    def _latex(self, unused=None):
        return self.left._latex() + self.oper_l + self.right._latex()

    def __init__(self, p1, p2):
        p1 = Posynomial(p1)
        p2 = Posynomial(p2)
        p = p1 / p2
        if isinstance(p.cs, Quantity):
            try:
                p = p.to('dimensionless')
            except DimensionalityError:
                raise ValueError("constraints must have the same units"
                                 " on both sides.")
        p1.units = None
        p2.units = None

        self.cs = p.cs
        self.exps = p.exps
        self.var_locs = p.var_locs

        if len(str(p1)) == len(str(p2)):
            if str(p1) <= str(p2):
                self.left, self.right = p1, p2
            else:
                self.left, self.right = p2, p1
        elif len(str(p1)) < len(str(p2)):
            self.left, self.right = p1, p2
        else:
            self.left, self.right = p2, p1

        self._set_operator(p1, p2)

    def __nonzero__(self):
        # a constraint not guaranteed to be satisfied
        # evaluates as "False"
        return bool(self.c == 1 and self.exp == {})


class MonoEQConstraint(Constraint):
    def _set_operator(self, p1, p2):
        self.oper_l = " = "
        self.oper_s = " = "
        self.leq = Constraint(p2, p1)
        self.geq = Constraint(p1, p2)

from .substitution import substitution

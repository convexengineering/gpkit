from collections import defaultdict
from itertools import chain
import numpy as np
from posyarray import PosyArray

latex_symbol_dict = {
    "omega": "\\omega",
    "rho": "\\rho",
    "mu": "\\mu",
    "pi": "\\pi",
    "nu": "\\nu",
    "eta": "\\eta",
    "tau": "\\tau",
    "th": "\\theta",
}


def sort_and_simplify(exps, cs):
    "Reduces the number of monomials, and casts them to a sorted form."
    matches = defaultdict(float)
    for i, exp in enumerate(exps):
        exp = HashVector({var: x for (var, x) in exp.iteritems() if x != 0})
        matches[exp] += cs[i]
    return tuple(matches.keys()), tuple(matches.values())


class Posynomial(object):
    '''A representation of a posynomial.

    exps: a tuple of exp vectors, each of which corresponds to a monomial
    cs: a tuple of coefficient values, each of which corresponds to a monomial

    exp: a HashVector of each variable's power in a monomial
    c: the scalar multiplier of a monomial (positive)

    var_locs: a dict containing for each variable in the posynomial a list
        of each monomial index that variable is in

    var_descrs: a dict containing strings describing variables
    '''

    def __init__(self, exps, cs=1, var_descrs=[], var_locs=None):
        if isinstance(exps, (int, float)):
            cs = exps
            exps = {}
        if isinstance(cs, (int, float)) and isinstance(exps, (dict, str)):
            # building a Monomial
            if isinstance(exps, str):
                exps = {exps: 1}
            cs = [cs]
            exps = [exps]
        elif isinstance(exps, Posynomial):
            cs = exps.cs
            var_descrs = exps.var_descrs
            exps = exps.exps
        else:
            # test for presence of length and identical lengths
            try: assert len(cs) == len(exps)
            except:
                raise TypeError("cs and exps must have the same length.")

        exps, cs = sort_and_simplify(exps, cs)
        if any((c <= 0 for c in cs)):
            raise ValueError("each c must be positive.")
        elif any((any((len(var.split()) > 1 for var in exp)) for exp in exps)):
            raise ValueError("variable names may not contain spaces.")

        self.exps = exps
        self.cs = cs
        if len(exps) == 1:
            self.__class__ = Monomial
            self.exp = exps[0]
            self.c = cs[0]

        if var_locs is None:
            self.var_locs = locate_vars(exps)
        else:
            self.var_locs = var_locs

        self.var_descrs = defaultdict(str)
        if var_descrs:
            if isinstance(var_descrs, dict):
                    self.var_descrs.update(var_descrs)
            else:
                try:
                    assert isinstance(var_descrs[0], str)
                    assert len(self.var_locs) == 1
                    # if we only have one variable, a string can describe it
                    var_descr = {self.var_locs.keys()[0]: [None, var_descrs[0]]}
                    self.var_descrs.update(var_descr)
                except:
                    for var_descr in var_descrs:
                        if isinstance(var_descr, dict):
                            self.var_descrs.update(var_descr)
                        else:
                            raise ValueError("invalid variable descriptions: "
                                             + str(var_descr))

    def sub(self, substitutions, val=None):
        var_locs, exps, cs, newdescrs, subs = substitution(self.var_locs,
                                                           self.exps, self.cs,
                                                           substitutions, val)
        return Posynomial(exps, cs, [self.var_descrs, newdescrs], var_locs)

    # hashing, immutability, Posynomial inequality
    def __hash__(self):
        if not hasattr(self, "exps"):
            print "hihihi"
            print type(self)
            print dir(self)
        if not hasattr(self, "_hashvalue"):
            self._hashvalue = hash(tuple(zip(self.exps, tuple(self.cs))))
        return self._hashvalue

    def __ne__(self, other):
        if isinstance(other, Posynomial):
            return not (self.exps == other.exps and self.cs == other.cs)
        else:
            return False

    # constraint generation
    def __eq__(self, other):
        if not isinstance(other, Posynomial):
            return False
        else:
            if (self.exps == other.exps and self.cs <= other.cs):
                return True
            else:
                if (isinstance(other, Monomial) and isinstance(self, Monomial)):
                    return MonoEQConstraint(self, other)
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

    # string translations
    def _string(self, mult_symbol='*'):
        mstrs = []
        for c, exp in zip(self.cs, self.exps):
            expsorted = sorted(exp.iteritems(), key=lambda x: x[1])
            varstrs = ['%s**%.2g' % (var, x) if x != 1 else var
                       for (var, x) in expsorted if x != 0]
            cstr = ["%.2g" % c] if c != 1 or not varstrs else []
            mstrs.append(mult_symbol.join(cstr + varstrs))
        return " + ".join(mstrs)

    def __repr__(self):
        return self.__class__.__name__+"("+self._string()+")"

    def _latex(self, unused=None):
        "For pretty printing with Sympy"
        mstrs = []
        for c, exp in zip(self.cs, self.exps):
            pos_vars, neg_vars = [], []
            for var, x in exp.iteritems():
                if '_' in var:
                    idx = var.index("_")
                    varbase = var[:idx]
                    if varbase in latex_symbol_dict:
                        varbase = latex_symbol_dict[varbase]
                    var = varbase+"_{"+var[idx+1:]+"}"
                else:
                    if var in latex_symbol_dict:
                        var = latex_symbol_dict[var]
                if x > 0:
                    pos_vars.append((var, x))
                elif x < 0:
                    neg_vars.append((var, x))

            pvarstrs = ['%s^{%.4g}' % (var, x) if x != 1 else var
                        for (var, x) in pos_vars]
            nvarstrs = ['%s^{%.4g}' % (var, -x) if -x != 1 else var
                        for (var, x) in neg_vars]
            pvarstr = ' '.join(pvarstrs)
            nvarstr = ' '.join(nvarstrs)
            if pos_vars and c == 1:
                cstr = ""
            else:
                cstr = "%.4g" % c
                if 'e' in cstr:
                    idx = cstr.index('e')
                    cstr = "%s\\times 10^{%i}" % (
                           cstr[:idx], int(cstr[idx+1:]))

            if not pos_vars and not neg_vars:
                mstrs.append("%s" % cstr)
            elif pos_vars and not neg_vars:
                mstrs.append("%s%s" % (cstr, pvarstr))
            elif neg_vars and not pos_vars:
                mstrs.append("\\frac{%s}{%s}" % (cstr, nvarstr))
            elif pos_vars and neg_vars:
                mstrs.append("%s\\frac{%s}{%s}" % (cstr, pvarstr, nvarstr))

        return " + ".join(mstrs)

    # posynomial arithmetic
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Posynomial(self.exps + ({},), self.cs + (other,),
                              self.var_descrs, self.var_locs)
        elif isinstance(other, Posynomial):
            return Posynomial(self.exps + other.exps, self.cs + other.cs,
                              [self.var_descrs, other.var_descrs])
        elif isinstance(other, PosyArray):
            return np.array(self)+other
        else:
            invalid_types_for_oper("+", self, other)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Posynomial(self.exps, other*np.array(self.cs),
                              self.var_descrs, self.var_locs)
        elif isinstance(other, Posynomial):
            C = np.outer(self.cs, other.cs)
            Exps = np.empty((len(self.exps), len(other.exps)), dtype="object")
            for i, exp_s in enumerate(self.exps):
                for j, exp_o in enumerate(other.exps):
                    Exps[i,j] = exp_s + exp_o
            return Posynomial(Exps.flatten(), C.flatten(),
                              [self.var_descrs, other.var_descrs])
        elif isinstance(other, PosyArray):
            return np.array(self)*other
        else:
            invalid_types_for_oper("*", self, other)

    def __rmul__(self, other): return self * other

    def __div__(self, other):
        if isinstance(other, Posynomial):
            if self.exps == other.exps:
                div_cs = np.array(self.cs)/np.array(other.cs)
                if all(div_cs == div_cs[0]):
                    return Monomial({}, div_cs[0])
        if isinstance(other, (int, float)):
            return Posynomial(self.exps, np.array(self.cs)/other,
                              self.var_descrs, self.var_locs)
        elif isinstance(other, Monomial):
            exps = [exp - other.exp for exp in self.exps]
            return Posynomial(exps, np.array(self.cs)/other.c,
                              [self.var_descrs, other.var_descrs])
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
        if isinstance(other, (int, float, Posynomial)):
            return other * self**-1
        else:
            invalid_types_for_oper("/", other, self)

    def __pow__(self, other):
        if isinstance(other, (int,float)):
            return Monomial(self.exp*other, self.c**other,
                            self.var_descrs)
        else:
            invalid_types_for_oper("** or pow()", self, x)


class Constraint(Posynomial):

    def _set_operator(self, p1, p2):
        if self.left is p1:
            self.oper_s = " <= "
            self.oper_l = " \leq "
        else:
            self.oper_s = " >= "
            self.oper_l = " \geq "

    def _string(self):
        return self.left._string() + self.oper_s + self.right._string()

    def _latex(self, unused=None):
        return self.left._latex() + self.oper_l + self.right._latex()

    def __init__(self, p1, p2):
        p1 = Posynomial(p1)
        p2 = Posynomial(p2)
        p = p1 / p2

        self.cs = p.cs
        self.var_descrs = p.var_descrs
        self.exps = p.exps
        self.var_locs = p.var_locs
        if len(self.exps) == 1:
            self.exp = self.exps[0]
            self.c = self.cs[0]

        if len(p1._string()) == len(p2._string()):
            if p1._string() <= p2._string():
                self.left, self.right = p1, p2
            else:
                self.left, self.right = p2, p1
        elif len(p1._string()) < len(p2._string()):
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

from internal_utils import *

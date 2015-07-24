"""Signomial, Posynomial, Monomial, Constraint, & MonoEQCOnstraint classes"""
import numpy as np

from .small_classes import Strings, Numbers, Quantity
from .posyarray import PosyArray
from .varkey import VarKey

from .small_scripts import diff, mono_approx
from .small_scripts import latex_num
from .small_scripts import sort_and_simplify
from .small_scripts import locate_vars
from .small_scripts import invalid_types_for_oper
from .small_scripts import mag, unitstr

from . import units as ureg
from . import DimensionalityError


class Signomial(object):
    """A representation of a signomial.

        Arguments
        ---------
        exps: tuple of dicts
            Exponent dicts for each monomial term
        cs: tuple
            Coefficient values for each monomial term
        varlocsandkeys: dict
            mapping from variable name to list of indices of monomial terms
            that variable appears in
        require_positive: bool
            If True and signomials not enabled, c <= 0 will raise ValueError

        Returns
        -------
        Signomial
        Posynomial (if the input has only positive cs)
        Monomial   (if the input has one term and only positive cs)
    """

    def __init__(self, exps=None, cs=1, varlocsandkeys=None,
                 require_positive=True, **descr):
        units = None
        if isinstance(exps, Numbers):
            cs = exps
            exps = {}
        if (isinstance(cs, Numbers) and
                (exps is None or isinstance(exps, Strings + (VarKey, dict)))):
            # building a Monomial
            if isinstance(exps, VarKey):
                exp = {exps: 1}
                units = exps.units
            elif exps is None or isinstance(exps, Strings):
                exp = ({VarKey(**descr): 1} if exps is None else
                       {VarKey(exps, **descr): 1})
                descr = list(exp)[0].descr
                units = descr["units"] if "units" in descr else None
            elif isinstance(exps, dict):
                exp = dict(exps)
                for key in exps:
                    if isinstance(key, Strings):
                        exp[VarKey(key)] = exp.pop(key)
            else:
                raise TypeError("could not make Monomial with %s" % type(exps))
            if isinstance(units, Quantity):
                cs = cs * units
            cs = [cs]
            exps = [exp]
        elif isinstance(exps, Signomial):
            cs = exps.cs
            varlocs = exps.varlocs
            exps = exps.exps
        else:
            # test for presence of length and identical lengths
            try:
                assert len(cs) == len(exps)
                exps_ = list(range(len(exps)))
                if not isinstance(cs[0], Quantity):
                    try:
                        cs = np.array(cs, dtype='float')
                    except ValueError:
                        raise ValueError("cannot add dimensioned and"
                                         " dimensionless monomials together.")
                else:
                    units = cs[0]/cs[0].magnitude
                    if units.dimensionless:
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
                            exps_[i][VarKey(key)] = exps_[i].pop(key)
                exps = exps_
            except AssertionError:
                raise TypeError("cs and exps must have the same length.")

        exps, cs = sort_and_simplify(exps, cs)
        if isinstance(cs, Quantity):
            any_negative = any((c.magnitude <= 0 for c in cs))
        else:
            any_negative = any((c <= 0 for c in cs))
        if any_negative:
            from . import SIGNOMIALS_ENABLED
            if require_positive and not SIGNOMIALS_ENABLED:
                raise ValueError("each c must be positive.")
        else:
            self.__class__ = Posynomial

        if isinstance(cs[0], Quantity):
            units = cs[0]/cs[0].magnitude
        elif "units" in descr:
            units = descr["units"]
            if isinstance(units, Quantity):
                cs = cs*units
        else:
            units = None
        self.cs = cs
        self.exps = exps
        self.units = units

        if len(exps) == 1:
            if self.__class__ is Posynomial:
                self.__class__ = Monomial
            self.exp = exps[0]
            self.c = cs[0]

        if varlocsandkeys is None:
            varlocsandkeys = locate_vars(exps)
        self.varlocs, self.varkeys = varlocsandkeys

        self._hashvalue = hash(tuple(zip(self.exps, tuple(self.cs))))

    @property
    def value(self):
        """Self, with values substituted for variables that have values

        Returns
        -------
        float, if no symbolic variables remain after substitution
        (Monomial, Posynomial, or Signomial), otherwise.
        """
        values = {vk: vk.descr["value"] for vk in self.varlocs.keys()
                  if "value" in vk.descr}
        p = self.sub(values)
        if isinstance(p, Monomial):
            if not p.exp:
                return p.c
        return p

    def to(self, arg):
        return Signomial(self.exps, self.cs.to(arg).tolist())

    def diff(self, wrt):
        """Derivative of this with respect to a Variable

        Arguments
        ---------
        wrt (Variable):
            Variable to take derivative with respect to

        Returns
        -------
        Signomial (or Posynomial or Monomial)
        """
        if wrt in self.varkeys:
            wrt = self.varkeys[wrt]
        elif isinstance(wrt, Monomial):
            vks = list(wrt.exp)
            if len(vks) == 1:
                wrt = vks[0]
        exps, cs = diff(self, wrt)
        return Signomial(exps, cs, require_positive=False)

    def mono_approximation(self, x0):
        if isinstance(self, Monomial):
            raise TypeError("making a Monomial approximation of %s"
                            " is unnecessary; it's already a Monomial."
                            "" % str(self))
        else:
            c, exp = mono_approx(self, getsubs(self.varkeys, self.varlocs, x0))
            return Monomial(exp, c)

    def sub(self, substitutions, val=None, require_positive=True):
        """Returns a nomial with substitued values.

        Usage
        -----
        3 == (x**2 + y).sub({'x': 1, y: 2})
        3 == (x).gp.sub(x, 3)

        Arguments
        ---------
        substitutions : dict or key
            Either a dictionary whose keys are strings, Variables, or VarKeys,
            and whose values are numbers, or a string, Variable or Varkey.
        val : number (optional)
            If the substitutions entry is a single key, val holds the value
        require_positive : boolean (optional, default is True)
            Controls whether the returned value can be a signomial.

        Returns
        -------
        Returns substituted nomial.
        """
        varlocs, exps, cs, subs = substitution(self.varlocs, self.varkeys,
                                               self.exps, self.cs,
                                               substitutions, val)
        return Signomial(exps, cs, units=self.units,
                         require_positive=require_positive)

    def subcmag(self, substitutions, val=None):
        varlocs, exps, cs, subs = substitution(self.varlocs, self.varkeys,
                                               self.exps, mag(self.cs),
                                               substitutions, val)
        if any(exps):
            raise ValueError("could not substitute for all variables.")
        return mag(cs).sum()

    def prod(self):
        return self

    def sum(self):
        return self

    # hashing, immutability, Signomial inequality
    def __hash__(self):
        return self._hashvalue

    def __ne__(self, other):
        if isinstance(other, Signomial):
            return not (self.exps == other.exps and self.cs == other.cs)
        else:
            return False

    # constraint generation
    def __eq__(self, other):
        # if at least one is a monomial, return a constraint
        mons = Numbers+(Monomial,)
        if isinstance(other, mons) and isinstance(self, mons):
            return MonoEQConstraint(self, other)
        elif isinstance(other, Signomial) and isinstance(self, Signomial):
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
        if isinstance(other, PosyArray):
            return NotImplemented
        else:
            return Constraint(self, other)

    def __ge__(self, other):
        if isinstance(other, PosyArray):
            return NotImplemented
        else:
            return Constraint(other, self)

    def __lt__(self, other):
        invalid_types_for_oper("<", self, other)

    def __gt__(self, other):
        invalid_types_for_oper(">", self, other)

    def __str__(self, mult_symbol='*'):
        mstrs = []
        for c, exp in zip(self.cs, self.exps):
            varstrs = ['%s**%.2g' % (var, x) if x != 1 else "%s" % var
                       for (var, x) in exp.items() if x != 0]
            varstrs.sort()
            c = mag(c)
            cstr = "%.2g" % c
            if cstr == "-1":
                mstrs.append("-" + mult_symbol.join(varstrs))
            else:
                cstr = [cstr] if cstr != "1" or not varstrs else []
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
            for var, x in exp.items():
                if x > 0:
                    pos_vars.append((var._latex(), x))
                elif x < 0:
                    neg_vars.append((var._latex(), x))

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

        units = unitstr(self.units, r"\mathrm{\left[ %s \right]}", "L~")
        units_tf = units.replace("frac", "tfrac").replace(r"\cdot", r"\cdot ")
        return " + ".join(sorted(mstrs)) + units_tf

    # posynomial arithmetic
    def __add__(self, other):
        if isinstance(other, Numbers):
            if other == 0:
                return Signomial(self.exps, self.cs,
                                 (self.varlocs, self.varkeys))
            else:
                return Signomial(self.exps + ({},),
                                 self.cs.tolist() + [other])
        elif isinstance(other, Signomial):
            return Signomial(self.exps + other.exps,
                             self.cs.tolist() + other.cs.tolist())
        elif isinstance(other, PosyArray):
            return np.array(self)+other
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, Numbers):
            if not other:
                # assume other is multiplicative zero
                return other
            return Signomial(self.exps, other*self.cs,
                             (self.varlocs, self.varkeys))
        elif isinstance(other, Signomial):
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
            return Signomial(Exps.flatten(), C.flatten())
        elif isinstance(other, PosyArray):
            return np.array(self)*other
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        """Support the / operator in Python 2.x"""
        if isinstance(other, Numbers):
            return Signomial(self.exps, self.cs/other)
        elif isinstance(other, Monomial):
            exps = [exp - other.exp for exp in self.exps]
            return Signomial(exps, self.cs/other.c)
        elif isinstance(other, PosyArray):
            return np.array(self)/other
        else:
            return NotImplemented

    def __truediv__(self, other):
        """Support the / operator in Python 3.x"""
        return self.__div__(other)

    def __pow__(self, x):
        if isinstance(x, int):
            if x >= 0:
                p = Monomial({}, 1)
                while x > 0:
                    p *= self
                    x -= 1
                return p
            else:
                raise ValueError("Signomials are only closed under"
                                 " nonnegative integer exponents.")
        else:
            return NotImplemented

    def __neg__(self):
        from . import SIGNOMIALS_ENABLED

        if not SIGNOMIALS_ENABLED:
            return NotImplemented
        else:
            return -1*self

    def __sub__(self, other):
        from . import SIGNOMIALS_ENABLED

        if not SIGNOMIALS_ENABLED:
            return NotImplemented
        else:
            return self + -other

    def __rsub__(self, other):
        from . import SIGNOMIALS_ENABLED

        if not SIGNOMIALS_ENABLED:
            return NotImplemented
        else:
            return other + -self

    def __float__(self):
        if len(self.exps) == 1:
            if not self.exps[0]:
                return mag(self.c)

        raise AttributeError("float() can only be called"
                             " on monomials with no variable terms")


class Posynomial(Signomial):
    """A Signomial with strictly positive cs

    Arguments
    ---------
    Same as Signomial.
    Note: Posynomial historically supported several different init formats
          These will be deprecated in the future, replaced with a single
          __init__ syntax, same as Signomial.
    """
    pass


class Monomial(Posynomial):
    """A Posynomial with only one term

    Arguments
    ---------
    Same as Signomial.
    Note: Monomial historically supported several different init formats
          These will be deprecated in the future, replaced with a single
          __init__ syntax, same as Signomial.
    """
    def __rdiv__(self, other):
        """Divide other by this Monomial"""
        if isinstance(other, Numbers+(Posynomial,)):
            return other * self**-1
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """rdiv for python 3.x"""
        return self.__rdiv__(other)

    def __pow__(self, other):
        if isinstance(other, Numbers):
            return Monomial(self.exp*other, self.c**other)
        else:
            return NotImplemented


class Constraint(Posynomial):
    """A constraint of the general form monomial > posynomial
    Stored internally (exps, cs) as a single Posynomial (1 >= self)
    Usually initialized via operator overloading, e.g. cc = y**2 >= 1 + x
    Additionally stores input format (lhs vs rhs) in self.right and self.left
    Form is self.left >= self.right.

    TODO: this documentation needs to address Signomial Constraints.
    """
    def _set_operator(self, p1, p2):
        if self.left is p1:
            self.oper_s = " <= "
            self.oper_l = " \\leq "
        else:
            self.oper_s = " >= "
            self.oper_l = " \\geq "

    def __str__(self):
        return str(self.left) + self.oper_s + str(self.right)

    def __repr__(self):
        return repr(self.left) + self.oper_s + repr(self.right)

    def _latex(self, unused=None):
        return self.left._latex() + self.oper_l + self.right._latex()

    def __init__(self, p1, p2):
        """Initialize a constraint of the form p2 >= p1.

        Arguments
        ---------
        p1 (Signomial)
        p2 (Signomial)

        TODO: clarify how this __init__ handles Signomial constraints
        TODO: change p1 and p2 to left and right instead of auto-selecting?
        TODO: call super() from this __init__
        """
        p1 = Signomial(p1)
        p2 = Signomial(p2)
        from . import SIGNOMIALS_ENABLED

        if SIGNOMIALS_ENABLED and not isinstance(p2, Monomial):
            if p1.units:
                p = (p1 - p2)/p1.units + 1.0
            else:
                p = (p1 - p2) + 1.0
        else:
            p = p1 / p2
        if isinstance(p.cs, Quantity):
            try:
                p = p.to('dimensionless')
            except DimensionalityError:
                raise ValueError("constraints must have the same units"
                                 " on both sides: '%s' and '%s' can not"
                                 " be converted into each other."
                                 "" % (p1.units.units, p2.units.units))

        p1.units = None if all(p1.exps) else p1.units
        p2.units = None if all(p2.exps) else p2.units

        for i, exp in enumerate(p.exps):
            if not exp:
                if p.cs[i] < 1:
                    if SIGNOMIALS_ENABLED:
                        const = p.cs[i]
                        p -= const
                        p /= (1-const)
                    else:
                        coeff = float(1 - p.cs[i])
                        p.cs = np.hstack((p.cs[:i], p.cs[i+1:]))
                        p.exps = p.exps[:i] + p.exps[i+1:]
                        p = p/coeff
                elif p.cs[i] > 1 and not SIGNOMIALS_ENABLED:
                    raise ValueError("infeasible constraint:"
                                     "constant term too large.")

        self.cs = p.cs
        self.exps = p.exps
        self.varlocs = p.varlocs

        if len(p1.exps) == len(p2.exps):
            if len(p1.exps[0]) <= len(p2.exps[0]):
                self.left, self.right = p1, p2
            else:
                self.left, self.right = p2, p1
        elif len(p1.exps) < len(p2.exps):
            self.left, self.right = p1, p2
        else:
            self.left, self.right = p2, p1

        self._set_operator(p1, p2)


class MonoEQConstraint(Constraint):
    """
    A Constraint of the form Monomial == Monomial.
    Stored internally as a Monomial Constraint, (1 == self).
    """
    def _set_operator(self, p1, p2):
        self.oper_l = " = "
        self.oper_s = " == "
        self.leq = Constraint(p2, p1)
        self.geq = Constraint(p1, p2)

    def __nonzero__(self):
        # a constraint not guaranteed to be satisfied
        # evaluates as "False"
        return bool(mag(self.cs[0]) == 1.0 and self.exps[0] == {})

    def __bool__(self):
        return self.__nonzero__()


from .substitution import substitution, getsubs

"""Signomial, Posynomial, Monomial, Constraint, & MonoEQCOnstraint classes"""
import numpy as np

from .small_classes import Strings, Numbers, Quantity, HashVector
from .posyarray import PosyArray
from .varkey import VarKey
from .nomial_data import NomialData

from .small_scripts import latex_num
from .small_scripts import invalid_types_for_oper
from .small_scripts import mag, unitstr

from . import units as ureg
from . import DimensionalityError


class Signomial(NomialData):
    """A representation of a Signomial.

        Arguments
        ---------
        exps: tuple of dicts
            Exponent dicts for each monomial term
        cs: tuple
            Coefficient values for each monomial term
        require_positive: bool
            If True and Signomials not enabled, c <= 0 will raise ValueError

        Returns
        -------
        Signomial
        Posynomial (if the input has only positive cs)
        Monomial   (if the input has one term and only positive cs)
    """

    def __init__(self, exps=None, cs=1, require_positive=True, simplify=True,
                 **descr):
        # this is somewhat deprecated, used for Variables and subbing Monomials
        units = descr.get("units", None)
        # If cs has units, then they will override this setting.
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
                vk = VarKey(**descr) if exps is None else VarKey(exps, **descr)
                descr = vk.descr
                units = vk.units
                exp = {vk: 1}
            elif isinstance(exps, dict):
                exp = dict(exps)
                for key in exps:
                    if isinstance(key, Strings):
                        exp[VarKey(key)] = exp.pop(key)
            else:
                raise TypeError("could not make Monomial with %s" % type(exps))
            #simplify = False #TODO: this shouldn't require simplification
            cs = [cs]
            exps = [HashVector(exp)]
        elif isinstance(exps, Signomial):
            simplify = False
            cs = exps.cs
            exps = exps.exps
        else:
            try:
                # test for presence of length and identical lengths
                assert len(cs) == len(exps)
                exps_ = list(range(len(exps)))
                if not all(isinstance(c, Quantity) for c in cs):
                    try:
                        cs = np.array(cs, dtype='float')
                    except ValueError:
                        raise ValueError("cannot add dimensioned and"
                                         " dimensionless monomials together.")
                else:
                    units = Quantity(1, cs[0].units)
                    if units.dimensionless:
                        cs = [c * ureg.dimensionless for c in cs]
                        units = ureg.dimensionless
                    try:
                        cs = [c.to(units).magnitude for c in cs] * units
                    except DimensionalityError:
                        raise ValueError("cannot add monomials of"
                                         " different units together")
                for i in range(len(exps)):
                    exps_[i] = HashVector(exps[i])
                    for key in exps_[i]:
                        if isinstance(key, Strings+(Monomial,)):
                            exps_[i][VarKey(key)] = exps_[i].pop(key)
                exps = tuple(exps_)
            except AssertionError:
                raise TypeError("cs and exps must have the same length.")

        if isinstance(units, Quantity):
            if not isinstance(cs, Quantity):
                cs = cs*units
            else:
                cs = cs.to(units)

        # init NomialData to create self.exps, self.cs, and so on
        super(Signomial, self).__init__(exps, cs, simplify=simplify)

        if self.any_nonpositive_cs:
            from . import SIGNOMIALS_ENABLED
            if require_positive and not SIGNOMIALS_ENABLED:
                raise ValueError("each c must be positive.")
        else:
            self.__class__ = Posynomial

        if len(self.exps) == 1:
            if self.__class__ is Posynomial:
                self.__class__ = Monomial
            self.exp = self.exps[0]
            self.c = self.cs[0]

    __hash__ = NomialData.__hash__

    @property
    def value(self):
        """Self, with values substituted for variables that have values

        Returns
        -------
        float, if no symbolic variables remain after substitution
        (Monomial, Posynomial, or Signomial), otherwise.
        """
        p = self.sub(self.values)
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
        deriv = super(Signomial, self).diff(wrt)
        return Signomial(exps=deriv.exps, cs=deriv.cs)

    def mono_approximation(self, x_0):
        """Monomial approximation about a point x_0

        Arguments
        ---------
        x_0 (dict):
            point to monomialize about

        Returns
        -------
        Monomial (unless self(x_0) < 0, in which case a Signomial is returned)
        """
        if not x_0:
            for i, exp in enumerate(self.exps):
                if exp == {}:
                    return Monomial({}, self.cs[i])
        x_0 = get_constants(self, x_0)
        exp = HashVector()
        psub = self.sub(x_0)
        if psub.varlocs:
            raise ValueError("Variables %s remained after substituting x_0=%s"
                             % (list(psub.varlocs), x_0))
        p0 = psub.value  # includes any units
        m0 = 1
        for vk in self.varlocs:
            e = mag(x_0[vk]*self.diff(vk).sub(x_0, require_positive=False).c/p0)
            exp[vk] = e
            m0 *= (x_0[vk])**e
        return Monomial(exp, p0/mag(m0))


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
            Controls whether the returned value can be a Signomial.

        Returns
        -------
        Returns substituted nomial.
        """
        _, exps, cs, _ = substitution(self, substitutions, val)
        return Signomial(exps, cs, require_positive=require_positive)

    def subsummag(self, substitutions, val=None):
        "Returns the sum of the magnitudes of the substituted Signomial."
        _, exps, cs, _ = substitution(self, substitutions, val)
        if any(exps):
            raise ValueError("could not substitute for all variables.")
        return mag(cs).sum()

    def prod(self):
        return self

    def sum(self):
        return self

    def __ne__(self, other):
        return not Signomial.__eq__(self, other)

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
        return super(Signomial, self).__eq__(other)

    def __le__(self, other):
        if isinstance(other, PosyArray):
            return NotImplemented
        else:
            return SignomialConstraint(other, self, oper_ge=True)

    def __ge__(self, other):
        if isinstance(other, PosyArray):
            return NotImplemented
        else:
            # by default all constraints take the form left >= right
            return SignomialConstraint(self, other, oper_ge=True)

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
            cstr = "%.3g" % c
            if cstr == "-1" and varstrs:
                mstrs.append("-" + mult_symbol.join(varstrs))
            else:
                cstr = [cstr] if cstr != "1" or not varstrs else []
                mstrs.append(mult_symbol.join(cstr + varstrs))
        return " + ".join(sorted(mstrs)) + unitstr(self.units, " [%s]")

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

    def posy_negy(self):
        """Get the positive and negative parts, both as Posynomials

        Returns
        -------
        Posynomial, Posynomial:
            p_pos and p_neg in (self = p_pos - p_neg) decomposition,
        """
        p_exp, p_cs, n_exp, n_cs = [], [], [], []
        assert len(self.cs) == len(self.exps)   # assert before calling zip
        for c, exp in zip(self.cs, self.exps):
            if mag(c) > 0:
                p_exp.append(exp)
                p_cs.append(c)
            elif mag(c) < 0:
                n_exp.append(exp)
                n_cs.append(-c)  # -c to keep posynomial
            else:
                raise ValueError("Unexpected c=%s in %s" % (c, self))
        return (Posynomial(p_exp, p_cs) if p_cs else 0,
                Posynomial(n_exp, n_cs) if n_cs else 0)

    # posynomial arithmetic
    def __add__(self, other):
        if isinstance(other, Numbers):
            if other == 0:
                return Signomial(self.exps, self.cs)
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
            return Signomial(self.exps, other*self.cs)
        elif isinstance(other, Signomial):
            C = np.outer(self.cs, other.cs)
            if isinstance(self.cs, Quantity) or isinstance(other.cs, Quantity):
                if not isinstance(self.cs, Quantity):
                    sunits = ureg.dimensionless
                else:
                    sunits = Quantity(1, self.cs[0].units)
                if not isinstance(other.cs, Quantity):
                    ounits = ureg.dimensionless
                else:
                    ounits = Quantity(1, other.cs[0].units)
                # HACK: fix for pint not working with np.outer
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
            return other.__rdiv__(self)
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
        if SIGNOMIALS_ENABLED:
            return -1*self
        else:
            return NotImplemented

    def __sub__(self, other):
        from . import SIGNOMIALS_ENABLED
        if SIGNOMIALS_ENABLED:
            return self + -other
        else:
            return NotImplemented

    def __rsub__(self, other):
        from . import SIGNOMIALS_ENABLED
        if SIGNOMIALS_ENABLED:
            return other + -self
        else:
            return NotImplemented

    def __float__(self):
        if len(self.exps) == 1:
            if not self.exps[0]:
                return mag(self.c)
        else:
            raise AttributeError("float() can only be called on"
                                 " monomials with no variable terms")


class Posynomial(Signomial):
    """A Signomial with strictly positive cs

    Arguments
    ---------
    Same as Signomial.
    Note: Posynomial historically supported several different init formats
          These will be deprecated in the future, replaced with a single
          __init__ syntax, same as Signomial.
    """
    def __le__(self, other):
        if isinstance(other, Numbers + (Monomial,)):
            return Constraint(other, self, oper_ge=True)
        else:
            # fall back on other's __ge__
            return NotImplemented

    # Posynomial.__ge__ falls back on Signomial.__ge__

    def mono_lower_bound(self, x_0):
        """Monomial lower bound at a point x0

        Arguments
        ---------
        x_0 (dict):
            point to make lower bound exact

        Returns
        -------
        Monomial
        """
        return self.mono_approximation(x_0)


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
        "Divide other by this Monomial"
        if isinstance(other, Numbers + (Signomial,)):
            return other * self**-1
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        "__rdiv__ for python 3.x"
        return self.__rdiv__(other)

    def __pow__(self, other):
        if isinstance(other, Numbers):
            return Monomial(self.exp*other, self.c**other)
        else:
            return NotImplemented

    # inherit __ne__ from Signomial

    def __eq__(self, other):
        mons = Numbers + (Monomial,)
        if isinstance(other, mons):
            # if both are monomials, return a constraint
            return MonoEQConstraint(self, other)
        return super(Monomial, self).__eq__(other)

    # Monomial.__le__ falls back on Posynomial.__le__

    def __ge__(self, other):
        if isinstance(other, Numbers + (Monomial,)):
            return Constraint(self, other, oper_ge=True)
        else:
            # fall back on other's __ge__
            return NotImplemented

    def mono_approximation(self, x0):
        raise TypeError("Monomial approximation of %s is unnecessary - "
                        "it's already a Monomial." % str(self))


class Constraint(Posynomial):
    """A constraint of the general form monomial >= posynomial
    Stored internally (exps, cs) as a single Posynomial (1 >= self)
    Usually initialized via operator overloading, e.g. cc = (y**2 >= 1 + x)
    Additionally retains input format (lhs vs rhs) in self.left and self.right
    Form is self.left >= self.right.
    """
    def __str__(self):
        return str(self.left) + self.oper_s + str(self.right)

    def __repr__(self):
        return repr(self.left) + self.oper_s + repr(self.right)

    def _latex(self, unused=None):
        return self.left._latex() + self.oper_l + self.right._latex()

    def __init__(self, left, right, oper_ge=True):
        """Initialize a constraint of the form left >= right
        (or left <= right, if oper_ge is False).

        Arguments
        ---------
        left: Monomial if oper_ge=True, else Posynomial
        right: Posynomial if oper_ge=True, else Monomial
        oper_ge: bool
            If true, form is left >= right; otherwise, left <= right.

        Note: Constraints initialized via operator overloading always take
              the form left >= right, e.g. (x <= y) becomes (y >= x).
        """
        pgt, plt = (left, right) if oper_ge else (right, left)
        plt = Posynomial(plt)
        pgt = Monomial(pgt)

        p = plt / pgt

        if isinstance(p.cs, Quantity):
            try:
                p = p.to('dimensionless')
            except DimensionalityError:
                raise ValueError("constraints must have the same units"
                                 " on both sides: '%s' and '%s' can not"
                                 " be converted into each other."
                                 "" % (plt.units, pgt.units))

        for i, exp in enumerate(p.exps):
            if not exp:
                if p.cs[i] < 1:
                    coeff = float(1 - p.cs[i])
                    p.cs = np.hstack((p.cs[:i], p.cs[i+1:]))
                    p.exps = p.exps[:i] + p.exps[i+1:]
                    p = p/coeff
                elif p.cs[i] > 1:
                    raise ValueError("infeasible constraint:"
                                     "constant term too large.")

        super(Constraint, self).__init__(p)
        self.__class__ = Constraint  # TODO should not have to do this

        self.left, self.right = left, right

        self.oper_s = " >= " if oper_ge else " <= "
        self.oper_l = r" \geq " if oper_ge else r" \leq "


class MonoEQConstraint(Constraint):
    """
    A Constraint of the form Monomial == Monomial.
    Stored internally as a Monomial Constraint, (1 == self).
    """
    def __init__(self, m1, m2):
        super(MonoEQConstraint, self).__init__(m1, m2)
        self.__class__ = MonoEQConstraint  # todo should not have to do this
        self.oper_l = " = "
        self.oper_s = " == "
        # next two lines would be clean implementation, but Constraint
        # inheritance from Posy won't allow division
        # self.leq = self
        # self.geq = 1 / self
        # instead, we'll do this -- TODO improve
        self.leq = m1/m2
        self.geq = m2/m1

    def __nonzero__(self):
        # a constraint not guaranteed to be satisfied
        # evaluates as "False"
        return bool(mag(self.cs[0]) == 1.0 and self.exps[0] == {})

    def __bool__(self):
        return self.__nonzero__()


class SignomialConstraint(Signomial):
    """A constraint of the general form posynomial >= posynomial
    Stored internally (exps, cs) as a single Signomial (0 >= self)
    Usually initialized via operator overloading, e.g. cc = (y**2 >= 1 + x - y)
    Additionally retains input format (lhs vs rhs) in self.left and self.right
    Form is self.left >= self.right.
    """
    def __str__(self):
        return str(self.left) + self.oper_s + str(self.right)

    def __repr__(self):
        return repr(self.left) + self.oper_s + repr(self.right)

    def _latex(self, unused=None):
        return self.left._latex() + self.oper_l + self.right._latex()

    def __init__(self, left, right, oper_ge=True):
        """Initialize a constraint of the form left >= right
        (or left <= right, if oper_ge is False).

        Arguments
        ---------
        left: Signomial
        right: Signomial
        oper_ge: bool
            If true, form is left >= right; otherwise, left <= right.

        Note: Constraints initialized via operator overloading always take
              the form left >= right, e.g. (x <= y) becomes (y >= x).

        Note: Unlike Constraints, SignomialConstraints have units.
        """
        left = Signomial(left)
        right = Signomial(right)
        pgt, plt = (left, right) if oper_ge else (right, left)

        from . import SIGNOMIALS_ENABLED
        if not SIGNOMIALS_ENABLED:
            raise TypeError("Cannot initialize SignomialConstraint "
                            "without SignomialsEnabled.")

        p = plt - pgt

        super(SignomialConstraint, self).__init__(p)
        self.__class__ = SignomialConstraint  # TODO should not have to do this
        self.left, self.right = left, right

        self.oper_s = " >= " if oper_ge else " <= "
        self.oper_l = r" \geq " if oper_ge else r" \leq "

from .substitution import substitution, get_constants

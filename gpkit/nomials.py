"""Signomial, Posynomial, Monomial, Constraint, & MonoEQCOnstraint classes"""
import numpy as np

from .small_classes import Strings, Numbers, Quantity
from .small_classes import HashVector, KeySet, KeyDict
from .posyarray import PosyArray
from .varkey import VarKey
from .nomial_data import NomialData

from .small_scripts import latex_num
from .small_scripts import invalid_types_for_oper
from .small_scripts import mag, unitstr, listify
from .nomial_data import simplify_exps_and_cs

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

    def mono_approximation(self, x0):
        """Monomial approximation about a point x0

        Arguments
        ---------
        x0 (dict):
            point to monomialize about

        Returns
        -------
        Monomial (unless self(x0) < 0, in which case a Signomial is returned)
        """
        if not x0:
            for i, exp in enumerate(self.exps):
                if exp == {}:
                    return Monomial({}, self.cs[i])
        x0, _, _ = parse_subs(self.varkeys, x0)
        exp = HashVector()
        psub = self.sub(x0)
        if psub.varlocs:
            raise ValueError("Variables %s remained after substituting x0=%s"
                             % (list(psub.varlocs), x0)
                             + " into %s" % self)
        p0 = psub.value  # includes any units
        m0 = 1
        for vk in self.varlocs:
            e = mag(x0[vk]*self.diff(vk).sub(x0, require_positive=False).c/p0)
            exp[vk] = e
            m0 *= (x0[vk])**e
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
            keys = set()
            for exp in exps:
                keys.update(exp)
            raise ValueError("could not substitute for %s" % keys)
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
            return SignomialConstraint(self, "<=", other)

    def __ge__(self, other):
        if isinstance(other, PosyArray):
            return NotImplemented
        else:
            # by default all constraints take the form left >= right
            return SignomialConstraint(self, ">=", other)

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

    def latex(self, unused=None, showunits=True):
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

    def _repr_latex_(self):
        return "$$"+self.latex()+"$$"

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
            return PosynomialConstraint(self, "<=", other)
        else:
            # fall back on other's __ge__
            return NotImplemented

    # Posynomial.__ge__ falls back on Signomial.__ge__

    def mono_lower_bound(self, x0):
        """Monomial lower bound at a point x0

        Arguments
        ---------
        x0 (dict):
            point to make lower bound exact

        Returns
        -------
        Monomial
        """
        return self.mono_approximation(x0)


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
            return MonoEQConstraint(self, "=", other)
        return super(Monomial, self).__eq__(other)

    # Monomial.__le__ falls back on Posynomial.__le__

    def __ge__(self, other):
        if isinstance(other, Numbers + (Posynomial,)):
            return PosynomialConstraint(self, ">=", other)
        else:
            # fall back on other's __ge__
            return NotImplemented

    def mono_approximation(self, x0):
        raise TypeError("Monomial approximation of %s is unnecessary - "
                        "it's already a Monomial." % str(self))


class Constraint(object):
    """Retains input format (lhs vs rhs) in self.left and self.right
    Calls self._constraint_init_ for child class initialization.
    """
    latex_opers = {"<=": "\\leq", ">=": "\\geq", "=": "="}

    def __init__(self, left, oper=None, right=None):
        if oper is None:
            oper = self.default_oper
        if right is None:
            right = self.default_right
        if not isinstance(oper, Strings):
            raise ValueError("operator must be string, not %s" % type(oper))
        self.left = Signomial(left)
        self.oper = oper
        self.right = Signomial(right)
        self.varkeys = KeySet(self.left.varkeys)
        self.varkeys.update(self.right.varkeys)
        self.substitutions = {}
        if hasattr(self, "_constraint_init_"):
            self._constraint_init_()

    def __str__(self):
        return "%s %s %s" % (self.left, self.oper, self.right)

    def __repr__(self):
        return "gpkit.%s(%s)" % (self.__class__.__name__, self)

    def latex(self):
        latex_oper = self.latex_opers[self.oper]
        return ("%s %s %s" % (self.left.latex(showunits=False), latex_oper,
                              self.right.latex(showunits=False)))

    def sub(self, subs, value=None):
        if value:
            subs = {subs: value}
        return self.__class__(self.left.sub(subs), self.oper,
                              self.right.sub(subs))

    def as_posyslt1(self):
        return [None]

    def as_localposyconstr(self, x0):
        return None

    def sensitivities(self, p_senss, m_sensss):
        constr_sens, var_senss = {}, {}
        assert False
        return constr_sens, var_senss

    def sp_sensitivities(self, posyapprox, posyapprox_sens, var_senss):
        constr_sens = {}
        assert False
        return constr_sens

    # optional: process_result


class PosynomialConstraint(Constraint):
    """A constraint of the general form monomial >= posynomial
    Stored in the posylt1_rep attribute as a single Posynomial (self <= 1)
    Usually initialized via operator overloading, e.g. cc = (y**2 >= 1 + x)
    """
    default_oper = "<="
    default_right = 1

    def _constraint_init_(self):
        if self.oper == "<=":
            p_lt, m_gt = self.left, self.right
        elif self.oper == ">=":
            m_gt, p_lt = self.left, self.right
        else:
            raise ValueError("operator %s is not supported by Posynomial"
                             "Constraint." % self.oper)

        p = p_lt / m_gt

        if isinstance(p.cs, Quantity):
            try:
                p = p.to('dimensionless')
            except DimensionalityError:
                raise ValueError("constraints must have the same units"
                                 " on both sides: '%s' and '%s' can not"
                                 " be converted into each other."
                                 "" % (p_lt.units, pgt.units))

        for i, exp in enumerate(p.exps):
            if not exp:
                if p.cs[i] < 1:
                    coeff = float(1 - p.cs[i])
                    p.cs = np.hstack((p.cs[:i], p.cs[i+1:]))
                    p.exps = p.exps[:i] + p.exps[i+1:]
                    p = p/coeff
                elif p.cs[i] > 1:
                    raise ValueError("infeasible constraint:"
                                     " constant term too large.")
        self.p_lt, self.m_gt = p_lt, m_gt
        self.posylt1_rep = p
        self.substitutions = p.values

    def as_posyslt1(self):
        posys = listify(self.posylt1_rep)
        if not self.substitutions:
            # just return the pre-generated posynomial representation
            return posys

        out = []
        for posy in posys:
            if hasattr(self, "m_gt"):
                m_gt = self.m_gt.sub(self.substitutions,
                                     require_positive=False)
                if m_gt.c == 0:
                    return []

            _, exps, cs, _ = substitution(posy, self.substitutions)
            # remove any cs that are just nans and/or 0s
            nans = np.isnan(cs)
            if np.all(nans) or np.all(cs[~nans] == 0):
                return []  # skip nan'd or 0'd constraint

            exps, cs, pmap = simplify_exps_and_cs(exps, cs, return_map=True)

            #  The monomial sensitivities from the GP/SP are in terms of this
            #  smaller post-substitution list of monomials, so we need to map
            #  back to the pre-substitution list.
            #
            #  A "pmap" is a list of HashVectors (mmaps), whose keys are
            #  monomial indexes pre-substitution, and whose values are the
            #  percentage of the simplified  monomial's coefficient that came
            #  from that particular parent.

            self.pmap = pmap
            p = Posynomial(exps, cs, simplify=False)
            if p.any_nonpositive_cs:
                raise RuntimeWarning("PosynomialConstraint %s became Signomial"
                                     " after substitution" % self)
            out.append(p)
        return out

    def sensitivities(self, p_senss, m_sensss):
        if not p_senss or not m_sensss:
            # as_posyslt1 created no inequalities
            return {}, {}
        p_sens, = p_senss
        m_senss, = m_sensss
        presub = self.posylt1_rep
        constr_sens = {"overall": p_sens}
        if hasattr(self, "pmap"):
            m_senss_ = np.zeros(len(presub.cs))
            counter = 0
            for i, mmap in enumerate(self.pmap):
                for idx, percentage in mmap.items():
                    m_senss_[idx] += percentage*m_senss[i]
            m_senss = m_senss_
        # Monomial sensitivities
        constr_sens[str(self.m_gt)] = p_sens
        for i, mono_sens in enumerate(m_senss):
            mono = Monomial(self.p_lt.exps[i], self.p_lt.cs[i])
            constr_sens[str(mono)] = mono_sens
        # Constant sensitivities
        var_senss = {var: sum([presub.exps[i][var]*m_senss[i] for i in locs])
                     for (var, locs) in presub.varlocs.items()
                     if var in self.substitutions}
        return constr_sens, var_senss


class MonoEQConstraint(PosynomialConstraint):
    """A Constraint of the form Monomial == Monomial.
    """
    default_oper = "="
    default_right = NotImplemented

    def _constraint_init_(self):
        if self.oper is not "=":
            raise ValueError("operator %s is not supported by"
                             " MonoEQConstraint." % self.oper)
        self.posylt1_rep = [self.left/self.right, self.right/self.left]
        self.substitutions = self.posylt1_rep[0].values

    def __nonzero__(self):
        # a constraint not guaranteed to be satisfied
        # evaluates as "False"
        return bool(mag(self.posylt1_rep[0].c) == 1.0
                    and self.posylt1_rep[0].exp == {})

    def __bool__(self):
        return self.__nonzero__()

    def sensitivities(self, p_senss, m_sensss):
        left, right = p_senss
        constr_sens = {str(self.left): left-right,
                       str(self.right): right-left}
        # Constant sensitivities
        var_senss = HashVector()
        for i, m_s in enumerate(m_sensss):
            presub = self.posylt1_rep[i]
            var_sens = {var: sum([presub.exps[i][var]*m_s[i] for i in locs])
                        for (var, locs) in presub.varlocs.items()
                        if var in self.substitutions}
            var_senss += HashVector(var_sens)
        return constr_sens, var_senss


class SignomialConstraint(Constraint):
    """A constraint of the general form posynomial >= posynomial
    Stored internally (exps, cs) as a single Signomial (0 >= self)
    Usually initialized via operator overloading, e.g. cc = (y**2 >= 1 + x - y)
    Additionally retains input format (lhs vs rhs) in self.left and self.right
    Form is self.left >= self.right.
    """
    default_oper = "<="
    default_right = 0

    def _constraint_init_(self):
        from . import SIGNOMIALS_ENABLED
        if not SIGNOMIALS_ENABLED:
            raise TypeError("Cannot initialize SignomialConstraint"
                            " outside of a SignomialsEnabled environment.")
        if self.oper == "<=":
            plt, pgt = self.left, self.right
        elif self.oper == ">=":
            pgt, plt = self.left, self.right
        else:
            raise ValueError("operator %s is not supported by Signomial"
                             "Constraint." % self.oper)
        self.sigy_lt0_rep = plt - pgt
        self.substitutions = self.sigy_lt0_rep.values

    def as_posyslt1(self):
        s = self.sigy_lt0_rep.sub(self.substitutions, require_positive=False)
        posy, negy = s.posy_negy()
        if len(negy.cs) != 1:
            return [None]
        else:
            self.__class__ = PosynomialConstraint
            self.__init__(posy, "<=", negy)
            return [posy/negy]

    def as_localposyconstr(self, x0):
        posy, negy = self.sigy_lt0_rep.posy_negy()
        if x0 is None:
            x0, _, _ = parse_subs(negy.varkeys, {})
             # TODO: don't all the equivalencies collide by now?
            sp_inits = {vk: vk.descr["sp_init"] for vk in negy.varlocs
                        if "sp_init" in vk.descr}
            x0.update(sp_inits)
            # HACK: initial guess for negative variables
            x0.update({var: 1 for var in negy.varlocs if var not in x0})
        else:
            x0 = dict(x0)
        x0.update(self.substitutions)
        return PosynomialConstraint(posy, "<=", negy.mono_lower_bound(x0))

    def sp_sensitivities(self, posyapprox, posyapprox_sens, var_senss):
        constr_sens = dict(posyapprox_sens)
        del constr_sens[str(posyapprox.m_gt)]
        _, negy = self.sigy_lt0_rep.posy_negy()
        constr_sens[str(negy)] = posyapprox_sens["overall"]
        posyapprox_sens[str(posyapprox)] = posyapprox_sens.pop("overall")
        constr_sens["posyapprox"] = posyapprox_sens
        return constr_sens

from .substitution import substitution, parse_subs

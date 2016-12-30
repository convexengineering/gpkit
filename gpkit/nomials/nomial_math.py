"""Signomial, Posynomial, Monomial, Constraint, & MonoEQCOnstraint classes"""
import numpy as np
from .data import simplify_exps_and_cs
from .nomial_core import Nomial
from .substitution import substitution, parse_subs
from ..constraints import SingleEquationConstraint
from ..small_classes import Strings, Numbers, Quantity
from ..small_classes import HashVector
from ..keydict import KeySet
from ..varkey import VarKey
from ..small_scripts import mag
from .. import ureg
from .. import DimensionalityError
from ..exceptions import InvalidGPConstraint


class Signomial(Nomial):
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
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-branches
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
                units = exps.units  # pylint: disable=no-member
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
            exps = [HashVector(exp)]  # pylint: disable=redefined-variable-type
        elif isinstance(exps, Nomial):
            simplify = False
            cs = exps.cs  # pylint: disable=no-member
            exps = exps.exps  # pylint: disable=no-member
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
                for i, k in enumerate(exps):
                    exps_[i] = HashVector(k)
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
            from .. import SIGNOMIALS_ENABLED
            if require_positive and not SIGNOMIALS_ENABLED:
                raise ValueError("each c must be positive.")
            self.__class__ = Signomial
        else:
            self.__class__ = Posynomial

        if len(self.exps) == 1:
            if self.__class__ is Posynomial:
                self.__class__ = Monomial
            self.exp = self.exps[0]
            self.c = self.cs[0]

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
        # pylint: disable=unexpected-keyword-arg
        return Signomial(deriv.exps, deriv.cs, require_positive=False)

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
        x0, _, _ = parse_subs(self.varkeys, x0)  # use only varkey keys
        exp = HashVector()
        psub = self.sub(x0)
        if psub.varlocs:
            raise ValueError("Variables %s remained after substituting x0=%s"
                             % (list(psub.varlocs), x0)
                             + " into %s" % self)
        p0 = psub.value  # includes any units
        m0 = 1
        for vk in self.varlocs:
            diff = self.diff(vk)
            e = x0[vk]*mag(diff.sub(x0, require_positive=False).c)/mag(p0)
            exp[vk] = e
            m0 *= (x0[vk])**e
        return Monomial(exp, p0/mag(m0))

    def sub(self, substitutions, require_positive=True):
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
        _, exps, cs, _ = substitution(self, substitutions)
        return Signomial(exps, cs, require_positive=require_positive)

    def subinplace(self, substitutions):
        "Substitutes in place."
        _, exps, cs, _ = substitution(self, substitutions)
        super(Signomial, self).__init__(exps, cs)

    def __le__(self, other):
        if isinstance(other, (Numbers, Signomial)):
            return SignomialInequality(self, "<=", other)
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, (Numbers, Signomial)):
            # by default all constraints take the form left >= right
            return SignomialInequality(self, ">=", other)
        else:
            return NotImplemented

    # posynomial arithmetic
    def __add__(self, other):
        if isinstance(other, Numbers):
            if other == 0:
                return Signomial(self.exps, self.cs)
            else:
                cs = self.cs.tolist() + [other]  # pylint: disable=no-member
                return Signomial(self.exps + ({},), cs)
        elif isinstance(other, Signomial):
             # pylint: disable=no-member
            cs = self.cs.tolist() + other.cs.tolist()
            return Signomial(self.exps + other.exps, cs)
        else:
            return NotImplemented

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
        else:
            return NotImplemented

    def __div__(self, other):
        """Support the / operator in Python 2.x"""
        if isinstance(other, Numbers):
            return Signomial(self.exps, self.cs/other)
        elif isinstance(other, Monomial):
            return other.__rdiv__(self)
        else:
            return NotImplemented

    def __pow__(self, expo):
        if isinstance(expo, int):
            if expo >= 0:
                p = Monomial({}, 1)
                while expo > 0:
                    p *= self
                    expo -= 1
                return p
            else:
                raise ValueError("Signomials are only closed under"
                                 " nonnegative integer exponents.")
        else:
            return NotImplemented

    def __neg__(self):
        from .. import SIGNOMIALS_ENABLED
        if SIGNOMIALS_ENABLED:
            return -1*self
        else:
            return NotImplemented

    def __sub__(self, other):
        from .. import SIGNOMIALS_ENABLED
        if SIGNOMIALS_ENABLED:
            return self + -other
        else:
            return NotImplemented

    def __rsub__(self, other):
        from .. import SIGNOMIALS_ENABLED
        if SIGNOMIALS_ENABLED:
            return other + -self
        else:
            return NotImplemented


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
            return PosynomialInequality(self, "<=", other)
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
            return MonomialEquality(self, "=", other)
        return super(Monomial, self).__eq__(other)

    # Monomial.__le__ falls back on Posynomial.__le__

    def __ge__(self, other):
        if isinstance(other, Numbers + (Posynomial,)):
            return PosynomialInequality(self, ">=", other)
        else:
            # fall back on other's __ge__
            return NotImplemented

    def mono_approximation(self, x0):
        return self


#######################################################
####### CONSTRAINTS ###################################
#######################################################


class ScalarSingleEquationConstraint(SingleEquationConstraint):
    "A SingleEquationConstraint with scalar left and right sides."
    nomials = []

    def __init__(self, left, oper, right):
        super(ScalarSingleEquationConstraint,
              self).__init__(Signomial(left), oper, Signomial(right))
        self.varkeys = KeySet(self.left.varlocs)
        self.varkeys.update(self.right.varlocs)

    def subinplace(self, substitutions):
        "Modifies the constraint in place with substitutions."
        for nomial in self.nomials:
            nomial.subinplace(substitutions)
        self.varkeys = KeySet(self.left.varlocs)
        self.varkeys.update(self.right.varlocs)


class PosynomialInequality(ScalarSingleEquationConstraint):
    """A constraint of the general form monomial >= posynomial
    Stored in the posylt1_rep attribute as a single Posynomial (self <= 1)
    Usually initialized via operator overloading, e.g. cc = (y**2 >= 1 + x)
    """

    def __init__(self, left, oper, right):
        ScalarSingleEquationConstraint.__init__(self, left, oper, right)
        if self.oper == "<=":
            p_lt, m_gt = self.left, self.right
        elif self.oper == ">=":
            m_gt, p_lt = self.left, self.right
        else:
            raise ValueError("operator %s is not supported by Posynomial"
                             "Constraint." % self.oper)

        self.p_lt, self.m_gt = p_lt, m_gt
        self.substitutions = dict(p_lt.values)
        self.substitutions.update(m_gt.values)
        self.unsubbed = self._gen_unsubbed()
        self.nomials = [self.left, self.right, self.p_lt, self.m_gt]
        self.nomials.extend(self.unsubbed)
        self._last_used_substitutions = None

    def _simplify_posy_ineq(self, exps, cs):
        "Simplify a posy <= 1 by moving constants to the right side."
        if len(exps) == 1:
            if not exps[0] or cs[0] == 0:
                if cs[0] > 1:
                    raise ValueError("infeasible constraint: %s" % self)
                else:
                    # allow tautological monomial constraints (cs[0] <= 1)
                    # because they allow models to impose requirements
                    return (), np.array([])
            return exps, cs
        coeff = 1.0
        exps_ = []
        nonzero_exp_ixs = []
        for i, exp in enumerate(exps):
            if exp:
                nonzero_exp_ixs.append(i)
                exps_.append(exp)
            else:
                coeff -= cs[i]
        if len(exps_) < len(exps):
            if coeff > 0:
                cs = cs[nonzero_exp_ixs]
            elif coeff < 0:
                raise ValueError("infeasible constraint: %s" % self)
            elif coeff == 0:
                raise ValueError("tautological constraint: %s" % self)
        return tuple(exps_), cs/coeff

    def _gen_unsubbed(self):
        "Returns the unsubstituted posys <= 1."
        p = self.p_lt / self.m_gt

        if isinstance(p.cs, Quantity):
            try:
                p.convert_to('dimensionless')
                p.cs = p.cs.magnitude
                p.units = None
            except DimensionalityError:
                raise ValueError("unit mismatch: units of %s cannot "
                                 "be converted to units of %s" %
                                 (self.p_lt, self.m_gt))

        p.exps, p.cs = self._simplify_posy_ineq(p.exps, p.cs)
        return [p]

    def as_posyslt1(self, substitutions=None):
        "Returns the posys <= 1 representation of this constraint."
        posys = self.unsubbed
        if not substitutions:
            self._last_used_substitutions = substitutions
            # just return the pre-generated posynomial representation
            return posys

        out = []
        for posy in posys:
            _, exps, cs, subs = substitution(posy, substitutions)
            self._last_used_substitutions = subs
            exps, cs = self._simplify_posy_ineq(exps, cs)
            if not exps and not cs:  # tautological constraint
                continue
            exps, cs, pmap = simplify_exps_and_cs(exps, cs, return_map=True)

            #  The monomial sensitivities from the GP/SP are in terms of this
            #  smaller post-substitution list of monomials, so we need to map
            #  back to the pre-substitution list.
            #
            #  A "pmap" is a list of HashVectors (mmaps), whose keys are
            #  monomial indexes pre-substitution, and whose values are the
            #  percentage of the simplified  monomial's coefficient that came
            #  from that particular parent.

            self.pmap = pmap  # pylint: disable=attribute-defined-outside-init
            try:
                p = Posynomial(exps, cs, simplify=False)
            except ValueError:
                raise RuntimeWarning("PosynomialInequality %s became Signomial"
                                     " after substitution %s"
                                     % (self, substitutions))
            out.append(p)
        return out

    def sens_from_dual(self, la, nu):
        "Returns the variable/constraint sensitivities from lambda/nu"
        if not la or not nu:
            # as_posyslt1 created no inequalities
            return {}
        la, = la
        nu, = nu
        presub, = self.unsubbed
        # constr_sens = {"overall": la}
        if hasattr(self, "pmap"):
            nu_ = np.zeros(len(presub.cs))
            for i, mmap in enumerate(self.pmap):
                for idx, percentage in mmap.items():
                    nu_[idx] += percentage*nu[i]
            nu = nu_
        # Monomial sensitivities
        # constr_sens[str(self.m_gt)] = la
        # for i, mono_sens in enumerate(nu):
        #     mono_str = fast_monomial_str(self.p_lt.exps[i], self.p_lt.cs[i])
        #     constr_sens[mono_str] = mono_sens
        # Constant sensitivities
        var_senss = {}
        for var in self._last_used_substitutions:
            locs = presub.varlocs[var]
            var_senss[var] = sum([presub.exps[i][var]*nu[i] for i in locs])
        return var_senss

    # pylint: disable=unused-argument
    def as_gpconstr(self, x0):
        "GP version of a Posynomial constraint is itself"
        return self


class MonomialEquality(PosynomialInequality):
    "A Constraint of the form Monomial == Monomial."

    # pylint: disable=super-init-not-called
    def __init__(self, left, oper, right):
        # pylint: disable=non-parent-init-called
        ScalarSingleEquationConstraint.__init__(self, left, oper, right)
        if self.oper is not "=":
            raise ValueError("operator %s is not supported by"
                             " MonomialEquality." % self.oper)
        self.substitutions = dict(self.left.values)
        self.substitutions.update(self.right.values)
        self.unsubbed = self._gen_unsubbed()
        self.nomials = [self.left, self.right]
        self.nomials.extend(self.unsubbed)

    def _gen_unsubbed(self):
        "Returns the unsubstituted posys <= 1."
        l_lt_r, r_lt_l = self.left/self.right, self.right/self.left
        if l_lt_r.units:
            try:
                l_lt_r.convert_to('dimensionless')
                l_lt_r.cs = l_lt_r.cs.magnitude
                l_lt_r.units = None
                r_lt_l.convert_to('dimensionless')
                r_lt_l.cs = r_lt_l.cs.magnitude
                r_lt_l.units = None
            except DimensionalityError:
                raise ValueError("unit mismatch: units of %s cannot "
                                 "be converted to units of %s" %
                                 (self.left, self.right))
        return [l_lt_r, r_lt_l]

    def __nonzero__(self):
        'A constraint not guaranteed to be satisfied  evaluates as "False".'
        return bool(self.left.c == self.right.c
                    and self.left.exp == self.right.exp)

    def __bool__(self):
        'A constraint not guaranteed to be satisfied  evaluates as "False".'
        return self.__nonzero__()

    def sens_from_dual(self, la, nu):
        "Returns the variable/constraint sensitivities from lambda/nu"
        if not la or not nu:
            # as_posyslt1 created no inequalities
            return {}
        # left, right = la
        # constr_sens = {str(self.left): left-right,
        #                str(self.right): right-left}
        # Constant sensitivities
        var_senss = HashVector()
        for i, m_s in enumerate(nu):
            presub = self.unsubbed[i]
            var_sens = {var: sum([presub.exps[i][var]*m_s[i] for i in locs])
                        for (var, locs) in presub.varlocs.items()
                        if var in self._last_used_substitutions}
            var_senss += HashVector(var_sens)
        return var_senss


class SignomialInequality(ScalarSingleEquationConstraint):
    """A constraint of the general form posynomial >= posynomial
    Stored internally (exps, cs) as a single Signomial (0 >= self)
    Usually initialized via operator overloading, e.g. cc = (y**2 >= 1 + x - y)
    Additionally retains input format (lhs vs rhs) in self.left and self.right
    Form is self.left >= self.right.
    """

    def __init__(self, left, oper, right):
        ScalarSingleEquationConstraint.__init__(self, left, oper, right)
        from .. import SIGNOMIALS_ENABLED
        if not SIGNOMIALS_ENABLED:
            raise TypeError("Cannot initialize SignomialInequality"
                            " outside of a SignomialsEnabled environment.")
        if self.oper == "<=":
            plt, pgt = self.left, self.right
        elif self.oper == ">=":
            pgt, plt = self.left, self.right
        else:
            raise ValueError("operator %s is not supported by Signomial"
                             "Constraint." % self.oper)
        self.nomials = [self.left, self.right]
        self.unsubbed = [plt - pgt]
        self.nomials.extend(self.unsubbed)
        self.substitutions = dict(self.left.values)
        self.substitutions.update(self.right.values)

    def as_posyslt1(self, substitutions=None):
        "Returns the posys <= 1 representation of this constraint."
        siglt0, = self.unsubbed
        siglt0 = siglt0.sub(substitutions, require_positive=False)
        posy, negy = siglt0.posy_negy()
        if posy is 0:
            raise ValueError("SignomialConstraint %s became the tautological"
                             " constraint %s %s %s after substitution." %
                             (self, posy, "<=", negy))
        elif negy is 0:
            raise ValueError("SignomialConstraint %s became the infeasible"
                             " constraint %s %s %s after substitution." %
                             (self, posy, "<=", negy))
        elif not hasattr(negy, "cs") or len(negy.cs) == 1:
            self.__class__ = PosynomialInequality
            self.__init__(posy, "<=", negy)
            return self.as_posyslt1(substitutions)

        else:
            raise InvalidGPConstraint("SignomialInequality could not simplify"
                                      " to a PosynomialInequality; try calling"
                                      " `.localsolve` instead of `.solve` to"
                                      " form your Model as a SignomialProgram")

    def as_gpconstr(self, x0):
        "Returns GP approximation of an SP constraint at x0"
        siglt0, = self.unsubbed
        posy, negy = siglt0.posy_negy()
        # assume unspecified negy variables have a value of 1.0
        x0.update({vk: 1.0 for vk in negy.varlocs if vk not in x0})
        pc = PosynomialInequality(posy, "<=", negy.mono_lower_bound(x0))
        pc.substitutions = self.substitutions
        return pc


class SingleSignomialEquality(SignomialInequality):
    "A constraint of the general form posynomial == posynomial"

    def __init__(self, left, right):
        SignomialInequality.__init__(self, left, "<=", right)
        self.oper = "="

    def as_posyslt1(self, substitutions=None):
        "Returns the posys <= 1 representation of this constraint."
        # todo deal with substitutions
        raise InvalidGPConstraint("SignomialEquality could not simplify"
                                  " to a PosynomialInequality; try calling"
                                  "`.localsolve` instead of `.solve` to"
                                  " form your Model as a SignomialProgram")

    def as_gpconstr(self, x0):
        "Returns GP approximation of an SP constraint at x0"
        siglt0, = self.unsubbed
        posy, negy = siglt0.posy_negy()
        # assume unspecified variables have a value of 1.0
        x0.update({vk: 1.0 for vk in siglt0.varlocs if vk not in x0})
        mec = (posy.mono_lower_bound(x0) == negy.mono_lower_bound(x0))
        mec.substitutions = self.substitutions
        return mec

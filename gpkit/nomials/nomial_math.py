"""Signomial, Posynomial, Monomial, Constraint, & MonoEQCOnstraint classes"""
import numpy as np
from .nomial_core import Nomial
from ..constraints import SingleEquationConstraint
from ..small_classes import Strings, Numbers, Quantity
from ..small_classes import HashVector
from ..keydict import KeySet
from ..varkey import VarKey
from ..small_scripts import mag
from .. import ureg
from .. import DimensionalityError
from ..exceptions import InvalidGPConstraint
from ..nomial_map import NomialMap, parse_subs


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
    def __init__(self, exps=None, cs=1, require_positive=True, **descr):
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-branches
        if isinstance(exps, NomialMap):
            hmap = exps
        else:
            hmap = None
            if isinstance(cs, Numbers):
                if exps is None:
                    exps = VarKey(**descr)
                elif isinstance(exps, Strings):
                    exps = VarKey(exps, **descr)
                elif isinstance(exps, dict):
                    exp = HashVector(exps)
                    for key in exps:
                        if isinstance(key, Strings):
                            exp[VarKey(key)] = exp.pop(key)
                    hmap = NomialMap({exp: mag(cs)})
                    hmap.set_units(cs)
                    hmap._remove_zeros()
            if hasattr(exps, "hmap"):
                hmap = exps.hmap  # should this be a copy?
                if not cs is 1:
                    raise ValueError("Nomial and cs cannot be input together.")
            elif isinstance(exps, Numbers):
                hmap = NomialMap([(HashVector(), mag(exps))])
                hmap.set_units(exps)
            elif not hmap:
                hmap = NomialMap()
                hmap.set_units(cs[0])
                for i, exp in enumerate(exps):
                    if isinstance(exp, Strings):
                        exp = VarKey(exp)
                    exp = HashVector(exp)
                    c = mag(cs[i].to(hmap.units)) if hmap.units else cs[i]
                    hmap[exp] = c + hmap.get(exp, 0)
                hmap._remove_zeros()

        super(Signomial, self).__init__(hmap)

        if self.any_nonpositive_cs:
            from .. import SIGNOMIALS_ENABLED
            if require_positive and not SIGNOMIALS_ENABLED:
                raise ValueError("each c must be positive.")
            self.__class__ = Signomial
        else:
            self.__class__ = Posynomial

        if len(self.hmap) == 1:
            if self.__class__ is Posynomial:
                self.__class__ = Monomial

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
        return Signomial(Nomial.diff(self, wrt), require_positive=False)

    def posy_negy(self):
        """Get the positive and negative parts, both as Posynomials

        Returns
        -------
        Posynomial, Posynomial:
            p_pos and p_neg in (self = p_pos - p_neg) decomposition,
        """
        py, ny = NomialMap(), NomialMap()
        py.units, ny.units = self.units, self.units
        for exp, c in self.hmap.items():
            if c > 0:
                py[exp] = c
            elif c < 0:
                ny[exp] = -c  # -c to keep it a posynomial
        return (Posynomial(py) if py else 0, Posynomial(ny) if ny else 0)

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
        x0 = parse_subs(self.varkeys, x0, sweeps=False)  # use only varkey keys
        psub = self.sub(x0)
        if psub.vks:
            raise ValueError("Variables %s remained after substituting x0=%s"
                             % (list(psub.vks), x0)
                             + " into %s" % self)
        # TODO: profile this code, return an hmap
        c0, = psub.hmap.values()
        exp = HashVector()
        c = c0
        for vk in self.vks:
            val = mag(x0[vk])
            diff, = self.diff(vk).sub(x0, require_positive=False).hmap.values()
            e = val*diff/c0
            exp[vk] = e
            c /= val**e
        if psub.hmap.units:
            c *= psub.hmap.units
        return Monomial(exp, c)

    def sub(self, substitutions, require_positive=True):
        """Returns a nomial with substitued values.

        Usage
        -----
        3 == (x**2 + y).sub({'x': 1, y: 2})
        3 == (x).gp.sub(x, 3)substitution

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
        return Signomial(self.hmap.sub(substitutions),
                         require_positive=require_positive)

    def subinplace(self, substitutions):
        "Substitutes in place."
        hmap = self.hmap.sub(substitutions)
        super(Signomial, self).__init__(hmap)
        self._reset()

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
        if isinstance(other, np.ndarray):
            return np.array(self)+other

        other_hmap = getattr(other, "hmap", None)
        if isinstance(other, Numbers):
            if not other:
                return Signomial(self.hmap)
            else:
                other_hmap = NomialMap({HashVector(): mag(other)})
                other_hmap.set_units(other)
        if other_hmap:
            return Signomial(self.hmap + other_hmap)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return np.array(self)*other

        if isinstance(other, Numbers):
            if not other:
                # assume other is multiplicative zero
                return other
            hmap = mag(other)*self.hmap
            hmap.units = self.hmap.units
            if isinstance(other, Quantity):
                if hmap.units:
                    hmap.set_units(hmap.units*other)
                else:
                    hmap.set_units(other)
            return Signomial(hmap)
        elif isinstance(other, Signomial):
            hmap = NomialMap()
            if not (self.hmap.units or other.hmap.units):
                hmap.units = None
            elif not self.hmap.units:
                hmap.units = other.hmap.units
            elif not other.hmap.units:
                hmap.units = self.hmap.units
            else:
                hmap.units = self.hmap.units*other.hmap.units
            for exp_s, c_s in self.hmap.items():
                for exp_o, c_o in other.hmap.items():
                    exp = exp_s + exp_o
                    hmap[exp] = c_s*c_o + hmap.get(exp, 0)
            hmap._remove_zeros()
            return Signomial(hmap)
        else:
            return NotImplemented

    def __div__(self, other):
        """Support the / operator in Python 2.x"""
        if isinstance(other, Numbers):
            return self*other**-1
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

    @property
    def exp(self):
        if not hasattr(self, "_exp"):
            self._exp, = self.hmap.keys()
        return self._exp

    @property
    def c(self):
        if not hasattr(self, "_c"):
            self._c, = self.cs
        return self._c

    def __rdiv__(self, other):
        "Divide other by this Monomial"
        if isinstance(other, Numbers + (Signomial,)):
            return other * self**-1
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        "__rdiv__ for python 3.x"
        return self.__rdiv__(other)

    def __pow__(self, x):
        if isinstance(x, Numbers):
            exp, c = self.hmap.items()[0]
            exp = exp*x if x else HashVector()
            # TODO: c should already be a float
            hmap = NomialMap({exp: float(c)**x})
            hmap.units = self.hmap.units
            if hmap.units:
                hmap.units = hmap.units**x
            return Monomial(hmap)
        else:
            return NotImplemented

    # inherit __ne__ from Signomial

    def __eq__(self, other):
        mons = Numbers + (Monomial,)
        if isinstance(other, mons):
            # if both are monomials, return a constraint
            try:
                return MonomialEquality(self, "=", other)
            except ValueError:
                return False
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
        self.varkeys = KeySet(self.left.vks)
        self.varkeys.update(self.right.vks)

    def subinplace(self, substitutions):
        "Modifies the constraint in place with substitutions."
        for nomial in self.nomials:
            nomial.subinplace(substitutions)
        self.varkeys = KeySet(self.left.vks)
        self.varkeys.update(self.right.vks)


# pylint: disable=too-many-instance-attributes
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
        self.unsubbed = self._gen_unsubbed(p_lt, m_gt)
        self.nomials = [self.left, self.right, self.p_lt, self.m_gt]
        self.nomials.extend(self.unsubbed)
        self._last_used_substitutions = {}

    def _simplify_posy_ineq(self, hmap, pmap=None, allow_tautological=True):
        "Simplify a posy <= 1 by moving constants to the right side."
        if HashVector() not in hmap:
            return hmap
        coeff = 1 - hmap[HashVector()]
        if pmap is not None:
            # move constant term's mmap to another
            const_idx = hmap.keys().index(HashVector())
            self.const_mmap = self.pmap.pop(const_idx)  # pylint: disable=attribute-defined-outside-init
            self.const_coeff = coeff  # pylint: disable=attribute-defined-outside-init
        if allow_tautological and len(hmap) == 1 and coeff >= 0:
            # don't error on tautological monomials (0 <= coeff)
            # because they allow models to impose requirements
            # raise ValueError("tautological constraint: %s" % self)
            return None
        elif coeff <= 0:
            raise ValueError("infeasible constraint: %s" % self)
        scaled = hmap/coeff
        scaled.units = hmap.units
        del scaled[HashVector()]
        return scaled

    def _gen_unsubbed(self, p_lt, m_gt):
        "Returns the unsubstituted posys <= 1."
        hmap = (p_lt / m_gt).hmap
        if hasattr(hmap, "units"):
            try:
                hmap = hmap.to(ureg.dimensionless)
                hmap.units = None
            except DimensionalityError:
                raise ValueError("unit mismatch: units of %s cannot "
                                 "be converted to units of %s" %
                                 (p_lt, m_gt))
        hmap = self._simplify_posy_ineq(hmap)
        if not hmap:
            return []
        return [Posynomial(hmap)]

    def as_posyslt1(self, substitutions=None):
        "Returns the posys <= 1 representation of this constraint."
        posys = self.unsubbed
        if not substitutions:
            # just return the pre-generated posynomial representation
            return posys

        out = []
        self._last_used_substitutions = {}
        for posy in posys:
            # 1) substitution
            fixed = parse_subs(posy.varkeys, substitutions, sweeps=False)
            self._last_used_substitutions.update(fixed)
            hmap = posy.hmap.sub(fixed)
            self.pmap, self.mfm = hmap.mmap(posy.hmap)
            # 2) constraint simpl. (subtracting the constant term from the RHS)
            hmap = self._simplify_posy_ineq(hmap, self.pmap)
            if not hmap:  # tautological constraint
                continue

            #  The monomial sensitivities from the GP/SP are in terms of this
            #  smaller post-substitution list of monomials, so we need to map
            #  back to the pre-substitution list.
            #
            #  A "pmap" is a list of HashVectors (mmaps), whose keys are
            #  monomial indexes pre-substitution, and whose values are the
            #  percentage of the simplified  monomial's coefficient that came
            #  from that particular parent.

            allnan_or_zero = True
            for c in hmap.values():
                if c != 0 and not np.isnan(c):
                    allnan_or_zero = False
                    break
            if allnan_or_zero:
                continue  # skip nan'd or 0'd constraint

            p = Posynomial(hmap)
            if p.any_nonpositive_cs:
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

            # TODO: why wrong in the aircraft example??
            # presubexps = presub.hmap.keys()
            # nu_2 = np.zeros(len(presub.cs))
            # for nu, exp in zip(nu, self.mfm):
            #     for presubexp, fraction in self.mfm[exp].items():
            #         idx = presubexps.index(presubexp)
            #         nu_2[idx] += fraction*nu

            if hasattr(self, "const_mmap"):
                scale = (1-self.const_coeff)/self.const_coeff
                for idx, percentage in self.const_mmap.items():
                    nu_[idx] += percentage * la*scale
            nu = nu_
        # Constant sensitivities
        var_senss = {}
        for var in self._last_used_substitutions:
            locs = presub.varlocs[var]
            var_senss[var] = sum([presub.exps[i][var]*nu[i] for i in locs])
        return var_senss

    # pylint: disable=unused-argument
    def as_gpconstr(self, x0, substitutions):
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
        self._last_used_substitutions = {}

    def _gen_unsubbed(self):
        "Returns the unsubstituted posys <= 1."
        unsubbed = PosynomialInequality._gen_unsubbed
        l_over_r = unsubbed(self, self.left, self.right)
        r_over_l = unsubbed(self, self.right, self.left)
        return l_over_r + r_over_l

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

    def as_gpconstr(self, x0, substitutions=None):
        "Returns GP approximation of an SP constraint at x0"
        siglt0, = self.unsubbed
        if substitutions:
            # check if it's a posynomial constraint after substitutions
            subsiglt0 = siglt0.sub(substitutions, require_positive=False)
            _, subnegy = subsiglt0.posy_negy()
            if not hasattr(subnegy, "cs") or len(subnegy.cs) == 1:
                return self
        posy, negy = siglt0.posy_negy()
        # assume unspecified negy variables have a value of 1.0
        x0.update({vk: 1.0 for vk in negy.varlocs if vk not in x0})
        pc = PosynomialInequality(posy, "<=", negy.mono_lower_bound(x0))
        pc.substitutions = self.substitutions
        return pc

    def as_approxslt(self):
        "Returns posynomial-less-than sides of a signomial constraint"
        siglt0, = self.unsubbed
        posy, self._negy = siglt0.posy_negy()  # pylint: disable=attribute-defined-outside-init
        return [posy]

    def as_approxsgt(self, x0):
        "Returns monomial-greater-than sides, to be called after as_approxlt1"
        return [self._negy.mono_lower_bound(x0)]


class SingleSignomialEquality(SignomialInequality):
    "A constraint of the general form posynomial == posynomial"

    def __init__(self, left, right):
        SignomialInequality.__init__(self, left, "<=", right)
        self.oper = "="

    def as_posyslt1(self, substitutions=None):
        "Returns the posys <= 1 representation of this constraint."
        # TODO: deal with substitutions
        raise InvalidGPConstraint("SignomialEquality could not simplify"
                                  " to a PosynomialInequality; try calling"
                                  "`.localsolve` instead of `.solve` to"
                                  " form your Model as a SignomialProgram")

    def as_gpconstr(self, x0, substitutions=None):
        "Returns GP approximation of an SP constraint at x0"
        # TODO: deal with substitutions
        siglt0, = self.unsubbed
        posy, negy = siglt0.posy_negy()
        # assume unspecified variables have a value of 1.0
        x0.update({vk: 1.0 for vk in siglt0.varlocs if vk not in x0})
        mec = (posy.mono_lower_bound(x0) == negy.mono_lower_bound(x0))
        mec.substitutions = self.substitutions
        return mec

    def as_approxslt(self):
        "Returns posynomial-less-than sides of a signomial constraint"
        self._siglt0, = self.unsubbed  # pylint: disable=attribute-defined-outside-init
        self._posy, self._negy = self._siglt0.posy_negy()  # pylint: disable=attribute-defined-outside-init
        return Monomial(1), Monomial(1)

    def as_approxsgt(self, x0):
        "Returns monomial-greater-than sides, to be called after as_approxlt1"
        lhs = self._posy.mono_lower_bound(x0)
        rhs = self._negy.mono_lower_bound(x0)
        return lhs/rhs, rhs/lhs

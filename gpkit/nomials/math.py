"""Signomial, Posynomial, Monomial, Constraint, & MonoEQCOnstraint classes"""
from collections import defaultdict
import numpy as np
from .core import Nomial
from .array import NomialArray
from .. import units
from ..constraints import SingleEquationConstraint
from ..globals import SignomialsEnabled
from ..small_classes import Numbers
from ..small_classes import HashVector, EMPTY_HV
from ..varkey import VarKey
from ..small_scripts import mag
from ..exceptions import (InvalidGPConstraint, InvalidPosynomial,
                          PrimalInfeasible, DimensionalityError)
from .map import NomialMap
from .substitution import parse_subs


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
    _c = _exp = None  # pylint: disable=invalid-name

    __hash__ = Nomial.__hash__

    def __init__(self, hmap=None, cs=1, require_positive=True):  # pylint: disable=too-many-statements,too-many-branches
        if not isinstance(hmap, NomialMap):
            if hasattr(hmap, "hmap"):
                hmap = hmap.hmap
            elif isinstance(hmap, Numbers):
                hmap_ = NomialMap([(EMPTY_HV, mag(hmap))])
                hmap_.units_of_product(hmap)
                hmap = hmap_
            elif isinstance(hmap, dict):
                exp = HashVector({VarKey(k): v for k, v in hmap.items() if v})
                hmap = NomialMap({exp: mag(cs)})
                hmap.units_of_product(cs)
            else:
                raise ValueError("Nomial construction accepts only NomialMaps,"
                                 " objects with an .hmap attribute, numbers,"
                                 " or *(exp dict of strings, number).")
        super().__init__(hmap)
        if self.any_nonpositive_cs:
            if require_positive and not SignomialsEnabled:
                raise InvalidPosynomial("each c must be positive.")
            self.__class__ = Signomial
        elif len(self.hmap) == 1:
            self.__class__ = Monomial
        else:
            self.__class__ = Posynomial

    def diff(self, var):
        """Derivative of this with respect to a Variable

        Arguments
        ---------
        var : Variable key
            Variable to take derivative with respect to

        Returns
        -------
        Signomial (or Posynomial or Monomial)
        """
        varset = self.varkeys[var]
        if len(varset) > 1:
            raise ValueError("multiple variables %s found for key %s"
                             % (list(varset), var))
        if not varset:
            diff = NomialMap({EMPTY_HV: 0.0})
            diff.units = None
        else:
            var, = varset
            diff = self.hmap.diff(var)
        return Signomial(diff, require_positive=False)

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
        return Posynomial(py) if py else 0, Posynomial(ny) if ny else 0

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
        psub = self.hmap.sub(x0, self.varkeys, parsedsubs=True)
        if EMPTY_HV not in psub or len(psub) > 1:
            raise ValueError("Variables %s remained after substituting x0=%s"
                             " into %s" % (psub, x0, self))
        c0, = psub.values()
        c, exp = c0, HashVector()
        for vk in self.vks:
            val = float(x0[vk])
            diff, = self.hmap.diff(vk).sub(x0, self.varkeys,
                                           parsedsubs=True).values()
            e = val*diff/c0
            if e:
                exp[vk] = e
            try:
                c /= val**e
            except OverflowError:
                raise OverflowError(
                    "While approximating the variable %s with a local value of"
                    " %s, %s/(%s**%s) overflowed. Try reducing the variable's"
                    " value by changing its unit prefix, or specify x0 values"
                    " for any free variables it's multiplied or divided by in"
                    " the posynomial %s whose expected value is far from 1."
                    % (vk, val, c, val, e, self))
        hmap = NomialMap({exp: c})
        hmap.units = self.units
        return Monomial(hmap)

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
        return Signomial(self.hmap.sub(substitutions, self.varkeys),
                         require_positive=require_positive)

    def __le__(self, other):
        if isinstance(other, (Numbers, Signomial)):
            return SignomialInequality(self, "<=", other)
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, (Numbers, Signomial)):
            return SignomialInequality(self, ">=", other)
        return NotImplemented

    def __add__(self, other, rev=False):
        other_hmap = getattr(other, "hmap", None)
        if isinstance(other, Numbers):
            if not other:  # other is zero
                return Signomial(self.hmap)
            other_hmap = NomialMap({EMPTY_HV: mag(other)})
            other_hmap.units_of_product(other)
        if other_hmap:
            astorder = (self, other)
            if rev:
                astorder = tuple(reversed(astorder))
            out = Signomial(self.hmap + other_hmap)
            out.ast = ("add", astorder)
            return out
        return NotImplemented

    def __mul__(self, other, rev=False):
        astorder = (self, other)
        if rev:
            astorder = tuple(reversed(astorder))
        if isinstance(other, np.ndarray):
            s = NomialArray(self)
            s.ast = self.ast
            return s*other
        if isinstance(other, Numbers):
            if not other:  # other is zero
                return other
            hmap = mag(other)*self.hmap
            hmap.units_of_product(self.hmap.units, other)
            out = Signomial(hmap)
            out.ast = ("mul", astorder)
            return out
        if isinstance(other, Signomial):
            hmap = NomialMap()
            for exp_s, c_s in self.hmap.items():
                for exp_o, c_o in other.hmap.items():
                    exp = exp_s + exp_o
                    new, accumulated = c_s*c_o, hmap.get(exp, 0)
                    if new != -accumulated:
                        hmap[exp] = accumulated + new
                    elif accumulated:
                        del hmap[exp]
            hmap.units_of_product(self.hmap.units, other.hmap.units)
            out = Signomial(hmap)
            out.ast = ("mul", astorder)
            return out
        return NotImplemented

    def __truediv__(self, other):
        "Support the / operator in Python 2.x"
        if isinstance(other, Numbers):
            out = self*other**-1
            out.ast = ("div", (self, other))
            return out
        if isinstance(other, Monomial):
            return other.__rtruediv__(self)
        return NotImplemented

    def __pow__(self, expo):
        if isinstance(expo, int) and expo >= 0:
            p = 1
            while expo > 0:
                p *= self
                expo -= 1
            p.ast = ("pow", (self, expo))
            return p
        return NotImplemented

    def __neg__(self):
        if SignomialsEnabled:  # pylint: disable=using-constant-test
            out = -1*self
            out.ast = ("neg", self)
            return out
        return NotImplemented

    def __sub__(self, other):
        return self + -other if SignomialsEnabled else NotImplemented  # pylint: disable=using-constant-test

    def __rsub__(self, other):
        return other + -self if SignomialsEnabled else NotImplemented  # pylint: disable=using-constant-test

    def chop(self):
        "Returns a list of monomials in the signomial."
        monmaps = [NomialMap({exp: c}) for exp, c in self.hmap.items()]
        for monmap in monmaps:
            monmap.units = self.hmap.units
        return [Monomial(monmap) for monmap in monmaps]

class Posynomial(Signomial):
    "A Signomial with strictly positive cs"

    __hash__ = Signomial.__hash__

    def __le__(self, other):
        if isinstance(other, Numbers + (Monomial,)):
            return PosynomialInequality(self, "<=", other)
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
    "A Posynomial with only one term"

    __hash__ = Posynomial.__hash__

    @property
    def exp(self):
        "Creates exp or returns a cached exp"
        if not self._exp:
            self._exp, = self.hmap.keys()  # pylint: disable=attribute-defined-outside-init
        return self._exp

    @property
    def c(self):  # pylint: disable=invalid-name
        "Creates c or returns a cached c"
        if not self._c:
            self._c, = self.cs  # pylint: disable=attribute-defined-outside-init, invalid-name
        return self._c

    def __rtruediv__(self, other):
        "Divide other by this Monomial"
        if isinstance(other, Numbers + (Signomial,)):
            out = other * self**-1
            out.ast = ("div", (other, self))
            return out
        return NotImplemented

    def __pow__(self, expo):
        if isinstance(expo, Numbers):
            (exp, c), = self.hmap.items()
            exp = exp*expo if expo else EMPTY_HV
            hmap = NomialMap({exp: c**expo})
            if expo and self.hmap.units:
                hmap.units = self.hmap.units**expo
            else:
                hmap.units = None
            out = Monomial(hmap)
            out.ast = ("pow", (self, expo))
            return out
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, MONS):
            try:  # if both are monomials, return a constraint
                return MonomialEquality(self, other)
            except (DimensionalityError, ValueError) as e:
                print("Infeasible monomial equality: %s" % e)
                return False
        return super().__eq__(other)

    def __ge__(self, other):
        if isinstance(other, Numbers + (Posynomial,)):
            return PosynomialInequality(self, ">=", other)
        # elif isinstance(other, np.ndarray):
        #     return other.__le__(self, rev=True)
        return NotImplemented

    # Monomial.__le__ falls back on Posynomial.__le__

    def mono_approximation(self, x0):
        return self


MONS = Numbers + (Monomial,)


#######################################################
####### CONSTRAINTS ###################################
#######################################################


class ScalarSingleEquationConstraint(SingleEquationConstraint):
    "A SingleEquationConstraint with scalar left and right sides."
    generated_by = v_ss = parent = None
    bounded = meq_bounded = {}

    def __init__(self, left, oper, right):
        lr = [left, right]
        self.varkeys = set()
        for i, sig in enumerate(lr):
            if isinstance(sig, Signomial):
                self.varkeys.update(sig.vks)
            else:
                lr[i] = Signomial(sig)
        from .. import NamedVariables
        self.lineage = tuple(NamedVariables.lineage)
        super().__init__(lr[0], oper, lr[1])

    def relaxed(self, relaxvar):
        "Returns the relaxation of the constraint in a list."
        if self.oper == ">=":
            return [relaxvar*self.left >= self.right]
        if self.oper == "<=":
            return [self.left <= relaxvar*self.right]
        if self.oper == "=":
            return [self.left <= relaxvar*self.right,
                    relaxvar*self.left >= self.right]
        raise ValueError(
            "Constraint %s had unknown operator %s." % self.oper, self)


# pylint: disable=too-many-instance-attributes, invalid-unary-operand-type
class PosynomialInequality(ScalarSingleEquationConstraint):
    """A constraint of the general form monomial >= posynomial
    Stored in the posylt1_rep attribute as a single Posynomial (self <= 1)
    Usually initialized via operator overloading, e.g. cc = (y**2 >= 1 + x)
    """
    feastol = 1e-3
    # NOTE: follows .check_result's max default, but 1e-3 seems a bit lax...

    def __init__(self, left, oper, right):
        ScalarSingleEquationConstraint.__init__(self, left, oper, right)
        if self.oper == "<=":
            p_lt, m_gt = self.left, self.right
        elif self.oper == ">=":
            m_gt, p_lt = self.left, self.right
        else:
            raise ValueError("operator %s is not supported." % self.oper)

        self.unsubbed = self._gen_unsubbed(p_lt, m_gt)
        self.bounded = set()
        for p in self.unsubbed:
            for exp in p.hmap:
                for vk, x in exp.items():
                    self.bounded.add((vk, "upper" if x > 0 else "lower"))

    def _simplify_posy_ineq(self, hmap, pmap=None, fixed=None):
        "Simplify a posy <= 1 by moving constants to the right side."
        if EMPTY_HV not in hmap:
            return hmap
        coeff = 1 - hmap[EMPTY_HV]
        if pmap is not None:  # note constant term's mmap
            const_idx = list(hmap.keys()).index(EMPTY_HV)
            self.const_mmap = self.pmap.pop(const_idx)  # pylint: disable=attribute-defined-outside-init
            self.const_coeff = coeff  # pylint: disable=attribute-defined-outside-init
        if coeff >= -self.feastol and len(hmap) == 1:
            return None   # a tautological monomial!
        if coeff < -self.feastol:
            msg = "'%s' is infeasible by %.2g%%" % (self, -coeff*100)
            if fixed:
                msg += " after substituting %s." % fixed
            raise PrimalInfeasible(msg)
        scaled = hmap/coeff
        scaled.units = hmap.units
        del scaled[EMPTY_HV]
        return scaled

    def _gen_unsubbed(self, p_lt, m_gt):
        """Returns the unsubstituted posys <= 1.

        Parameters
        ----------
        p_lt : posynomial
            the left-hand side of (posynomial < monomial)

        m_gt : monomial
            the right-hand side of (posynomial < monomial)

        """
        try:
            m_exp, = m_gt.hmap.keys()
            m_c, = m_gt.hmap.values()
        except ValueError:
            raise TypeError("greater-than side '%s' is not monomial." % m_gt)
        m_c *= units.of_division(m_gt, p_lt)
        hmap = p_lt.hmap.copy()
        for exp in list(hmap):
            hmap[exp-m_exp] = hmap.pop(exp)/m_c
        hmap = self._simplify_posy_ineq(hmap)
        return [Posynomial(hmap)] if hmap else []

    def as_hmapslt1(self, substitutions):
        "Returns the posys <= 1 representation of this constraint."
        out = []
        for posy in self.unsubbed:
            fixed, _, _ = parse_subs(posy.varkeys, substitutions, clean=True)
            hmap = posy.hmap.sub(fixed, posy.varkeys, parsedsubs=True)
            self.pmap, self.mfm = hmap.mmap(posy.hmap)  # pylint: disable=attribute-defined-outside-init
            hmap = self._simplify_posy_ineq(hmap, self.pmap, fixed)
            if hmap is not None:
                if any(c <= 0 for c in hmap.values()):
                    raise InvalidGPConstraint("'%s' became Signomial after sub"
                                              "stituting %s" % (self, fixed))
                hmap.parent = self
                out.append(hmap)
        return out

    def sens_from_dual(self, la, nu, _):
        "Returns the variable/constraint sensitivities from lambda/nu"
        presub, = self.unsubbed
        if hasattr(self, "pmap"):
            nu_ = np.zeros(len(presub.hmap))
            for i, mmap in enumerate(self.pmap):
                for idx, percentage in mmap.items():
                    nu_[idx] += percentage*nu[i]
            if hasattr(self, "const_mmap"):
                scale = (1-self.const_coeff)/self.const_coeff
                for idx, percentage in self.const_mmap.items():
                    nu_[idx] += percentage * la*scale
            nu = nu_
        self.v_ss = HashVector()
        for nu_i, exp in zip(nu, presub.hmap):
            for vk, x in exp.items():
                self.v_ss[vk] = nu_i*x + self.v_ss.get(vk, 0)
        return self.v_ss, la


class MonomialEquality(PosynomialInequality):
    "A Constraint of the form Monomial == Monomial."
    oper = "="

    def __init__(self, left, right):
        # pylint: disable=super-init-not-called,non-parent-init-called
        ScalarSingleEquationConstraint.__init__(self, left, self.oper, right)
        self.unsubbed = self._gen_unsubbed(self.left, self.right)
        self.bounded = set()
        self.meq_bounded = {}
        self._las = []
        if self.unsubbed and len(self.varkeys) > 1:
            exp, = self.unsubbed[0].hmap
            for key, e in exp.items():
                s_e = np.sign(e)
                ubs = frozenset((k, "upper" if np.sign(e) != s_e else "lower")
                                for k, e in exp.items() if k != key)
                lbs = frozenset((k, "lower" if np.sign(e) != s_e else "upper")
                                for k, e in exp.items() if k != key)
                self.meq_bounded[(key, "upper")] = frozenset([ubs])
                self.meq_bounded[(key, "lower")] = frozenset([lbs])

    def _gen_unsubbed(self, left, right):  # pylint: disable=arguments-differ
        "Returns the unsubstituted posys <= 1."
        unsubbed = PosynomialInequality._gen_unsubbed
        l_over_r = unsubbed(self, left, right)
        r_over_l = unsubbed(self, right, left)
        return l_over_r + r_over_l

    def as_hmapslt1(self, substitutions):
        "Tags posynomials for dual feasibility checking"
        out = super().as_hmapslt1(substitutions)
        for h in out:
            h.from_meq = True  # pylint: disable=attribute-defined-outside-init
        return out

    def __bool__(self):
        'A constraint not guaranteed to be satisfied evaluates as "False".'
        return bool(self.left.c == self.right.c
                    and self.left.exp == self.right.exp)

    def sens_from_dual(self, la, nu, _):
        "Returns the variable/constraint sensitivities from lambda/nu"
        self._las.append(la)
        if len(self._las) < 2:
            return {}, 0
        la = self._las[0] - self._las[1]
        self._las = []
        exp, = self.unsubbed[0].hmap
        self.v_ss = exp*la
        return self.v_ss, la


class SignomialInequality(ScalarSingleEquationConstraint):
    """A constraint of the general form posynomial >= posynomial

    Stored at .unsubbed[0] as a single Signomial (0 >= self)"""

    def __init__(self, left, oper, right):
        ScalarSingleEquationConstraint.__init__(self, left, oper, right)
        if not SignomialsEnabled:
            raise TypeError("Cannot initialize SignomialInequality"
                            " outside of a SignomialsEnabled environment.")
        if self.oper == "<=":
            plt, pgt = self.left, self.right
        elif self.oper == ">=":
            pgt, plt = self.left, self.right
        else:
            raise ValueError("operator %s is not supported." % self.oper)
        self.unsubbed = [plt - pgt]
        self.bounded = self.as_gpconstr({}).bounded

    def as_hmapslt1(self, substitutions):
        "Returns the posys <= 1 representation of this constraint."
        siglt0, = self.unsubbed
        siglt0 = siglt0.sub(substitutions, require_positive=False)
        posy, negy = siglt0.posy_negy()
        if posy is 0:  # pylint: disable=literal-comparison
            print("Warning: SignomialConstraint %s became the tautological"
                  " constraint 0 <= %s after substitution." % (self, negy))
            return []
        if negy is 0:  # pylint: disable=literal-comparison
            raise ValueError("%s became the infeasible constraint %s <= 0"
                             " after substitution." % (self, posy))
        if hasattr(negy, "cs") and len(negy.cs) > 1:
            raise InvalidGPConstraint(
                "%s did not simplify to a PosynomialInequality; try calling"
                " `.localsolve` instead of `.solve` to form your Model as a"
                " SequentialGeometricProgram." % self)
        # all but one of the negy terms becomes compatible with the posy
        p_ineq = PosynomialInequality(posy, "<=", negy)
        p_ineq.parent = self
        siglt0_us, = self.unsubbed
        siglt0_hmap = siglt0_us.hmap.sub(substitutions, siglt0_us.varkeys)
        negy_hmap = NomialMap()
        posy_hmaps = defaultdict(NomialMap)
        for o_exp, exp in siglt0_hmap.expmap.items():
            if exp == negy.exp:
                negy_hmap[o_exp] = -siglt0_us.hmap[o_exp]
            else:
                posy_hmaps[exp-negy.exp][o_exp] = siglt0_us.hmap[o_exp]
        # pylint: disable=attribute-defined-outside-init
        self._mons = [Monomial(NomialMap({k: v}))
                      for k, v in (posy/negy).hmap.items()]
        self._negysig = Signomial(negy_hmap, require_positive=False)
        self._coeffsigs = {exp: Signomial(hmap, require_positive=False)
                           for exp, hmap in posy_hmaps.items()}
        self._sigvars = {exp: (list(self._negysig.varkeys)
                               + list(sig.varkeys))
                         for exp, sig in self._coeffsigs.items()}
        return p_ineq.as_hmapslt1(substitutions)

    def sens_from_dual(self, la, nu, result):
        """ We want to do the following chain:
               dlog(Obj)/dlog(monomial[i])    = nu[i]
               * dlog(monomial)/d(monomial)   = 1/(monomial value)
               * d(monomial)/d(var)           = see below
               * d(var)/dlog(var)             = var
               = dlog(Obj)/dlog(var)
            each final monomial is really
               (coeff signomial)/(negy signomial)
            and by the chain rule d(monomial)/d(var) =
               d(coeff)/d(var)*1/negy + d(1/negy)/d(var)*coeff
               = d(coeff)/d(var)*1/negy - d(negy)/d(var)*coeff*1/negy**2
        """
        # pylint: disable=too-many-locals, attribute-defined-outside-init

        # pylint: disable=no-member
        def subval(posy):
            "Substitute solution into a posynomial and return the result"
            hmap = posy.sub(result["variables"],
                            require_positive=False).hmap
            (key, value), = hmap.items()
            assert not key  # constant
            return value

        var_senss = {}
        invnegy_val = 1/subval(self._negysig)
        for i, nu_i in enumerate(nu):
            mon = self._mons[i]
            inv_mon_val = 1/subval(mon)
            coeff = self._coeffsigs[mon.exp]
            for var in self._sigvars[mon.exp]:
                d_mon_d_var = (subval(coeff.diff(var))*invnegy_val
                               - (subval(self._negysig.diff(var))
                                  * subval(coeff) * invnegy_val**2))
                var_val = result["variables"][var]
                sens = (nu_i*inv_mon_val*d_mon_d_var*var_val)
                assert isinstance(sens, float)
                var_senss[var] = sens + var_senss.get(var, 0)
        return var_senss, la

    def as_gpconstr(self, x0):
        "Returns GP-compatible approximation at x0"
        siglt0, = self.unsubbed
        posy, negy = siglt0.posy_negy()
        # default guess of 1.0 for unspecified negy variables
        x0 = {vk: x0.get(vk, 1) for vk in negy.vks}
        pconstr = PosynomialInequality(posy, "<=", negy.mono_lower_bound(x0))
        pconstr.generated_by = self
        return pconstr


class SingleSignomialEquality(SignomialInequality):
    "A constraint of the general form posynomial == posynomial"

    def __init__(self, left, right):
        SignomialInequality.__init__(self, left, "<=", right)
        self.oper = "="
        self.meq_bounded = self.as_gpconstr({}).meq_bounded

    def as_hmapslt1(self, substitutions):
        "SignomialEquality is never considered GP-compatible"
        raise InvalidGPConstraint(self)

    def as_gpconstr(self, x0):
        "Returns GP-compatible approximation at x0"
        siglt0, = self.unsubbed
        posy, negy = siglt0.posy_negy()
        # default guess of 1.0 for unspecified negy variables
        x0 = {vk: x0.get(vk, 1) for vk in siglt0.vks}
        mec = (posy.mono_lower_bound(x0) == negy.mono_lower_bound(x0))
        mec.generated_by = self
        return mec

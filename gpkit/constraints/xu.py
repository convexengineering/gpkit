import numpy as np
from ..nomials.nomial_math import SignomialInequality
from ..nomials.variables import Variable
from ..nomials.array import NomialArray
from .costed import CostedConstraintSet
from .. import SignomialsEnabled


class XuConstraintSet(CostedConstraintSet):
    """ The objective function defined as y + sum of WiSi where Wi
        are the weights (defined apriori) and Si are the added
        slack variables """

    def __init__(self, objective, constraints, w=1, alpha=1, M=100):
        cost = Variable("J", unique=True)
        CostedConstraintSet.__init__(self, cost, constraints)
        # TODO: s might be too common of a name?
        self.s = NomialArray([c.s for c in self.flat() if hasattr(c, "s")])
        self.w = w*np.ones(self.s.shape)
        self.xi0 = Variable("\\xi_0", unique=True)
        self.alpha = alpha
        self.M = M
        self.objective = objective

    def as_gpconstr(self, x0):
        gpconstrs = CostedConstraintSet.as_gpconstr(self, x0)
        with SignomialsEnabled():
            f0p, f0m = self.objective.sub(self.substitutions).posy_negy()
        return [self.cost >= self.xi0 + (self.w*self.s).sum(),
                f0p + self.M <= (f0m + self.xi0).mono_lower_bound(x0),
                gpconstrs]

    def on_failedsolve(self):
        self.w *= self.alpha

    def on_successfulsolve(self, result):
        # normalize the cost by removing the value of unused slack variables
        result["cost"] -= self.w.sum()
        self.w *= self.alpha


XuConstraint_K11 = SignomialInequality
XuConstraint_K12 = SignomialInequality


class XuEqualityConstraint(SignomialInequality):
    def __init__(self, left, right):
        SignomialInequality.__init__(self, left, "<=", right)
        self.s = Variable("s", unique=True)

    def process_result(self, result):
        "Checks that all constraints are satisfied with equality"
        variables = result["variables"]
        leftsubbed = self.left.sub(variables).value
        rightsubbed = self.right.sub(variables).value
        rel_diff = abs(rightsubbed - leftsubbed)
        result["constr_viol"] = result.get("constr_viol", 0) + rel_diff

class XuConstraint_K21(XuEqualityConstraint):
    """An equality constraint of the general form monomial = monomial
    where posy and negy+1 are both monomials"""

    def as_gpconstr(self, x0):
        with SignomialsEnabled():
            posy, negy = self._unsubbed.sub(self.substitutions).posy_negy()
        # TODO: turn self into a MonomialEquality Constraint here
        return posy == negy


class XuConstraint_K22(XuEqualityConstraint):
    """An equality constraint of the general form posynomial = monomial
    where posy is a posynomial and negy+1 is a monomial"""

    def as_gpconstr(self, x0):
        with SignomialsEnabled():
            posy, negy = self._unsubbed.sub(self.substitutions).posy_negy()
        c1 = posy <= negy
        # s above 1 makes the approximated side easier
        c2 = negy <= self.s*posy.mono_lower_bound(x0)
        c3 = self.s >= 1
        return [c1, c2, c3]


class XuConstraint_K23(XuEqualityConstraint):
    """An equality constraint of the general form monomial = posynomial
    where the posy is a monomial and negy+1 is a posynomial"""

    def as_gpconstr(self, x0):
        with SignomialsEnabled():
            posy, negy = self._unsubbed.sub(self.substitutions).posy_negy()
        c1 = negy <= posy
        # s above 1 makes the approximated side easier
        c2 = posy <= self.s*negy.mono_lower_bound(x0)
        c3 = self.s >= 1
        return [c1, c2, c3]


class XuConstraint_K24(XuEqualityConstraint):
    """ A equality contraint which does not lie in K21, K22 or K23"""

    def as_gpconstr(self, x0):
        with SignomialsEnabled():
            posy, negy = self._unsubbed.sub(self.substitutions).posy_negy()
        # s above 1 makes one of the approximated sides easier
        c1 = negy <= self.s*posy.mono_lower_bound(x0)
        c2 = posy <= negy.mono_lower_bound(x0)
        c3 = self.s >= 1
        return [c1, c2, c3]

"Implements Bounded"
from collections import defaultdict
import numpy as np
from .. import Variable
from .set import ConstraintSet
from ..small_scripts import mag


def varkey_bounds(varkeys, lower, upper):
    """Returns constraints list bounding all varkeys.

    Arguments
    ---------
    varkeys : iterable
        list of varkeys to create bounds for

    lower : float
        lower bound for all varkeys

    upper : float
        upper bound for all varkeys
    """
    constraints = []
    for varkey in varkeys:
        variable = Variable(**varkey.descr)
        if variable.units:
            variable.hmap.units = None
            variable.units = None
        constraint = []
        if upper:
            constraint.append(upper >= variable)
        if lower:
            constraint.append(variable >= lower)
        constraints.append(constraint)
    return constraints


class Bounded(ConstraintSet):
    """Bounds contained variables so as to ensure dual feasibility.

    Arguments
    ---------
    constraints : iterable
        constraints whose varkeys will be bounded

    substitutions : dict
        as in ConstraintSet.__init__

    verbosity : int
        how detailed of a warning to print
            0: nothing
            1: print warnings

    eps : float
        default lower bound is eps, upper bound is 1/eps

    lower : float
        lower bound for all varkeys, replaces eps

    upper : float
        upper bound for all varkeys, replaces 1/eps
    """

    def __init__(self, constraints, verbosity=1,
                 eps=1e-30, lower=None, upper=None):
        if not isinstance(constraints, ConstraintSet):
            constraints = ConstraintSet(constraints)
        self.bound_las = None
        self.verbosity = verbosity
        self.lowerbound = lower if (lower or upper) else eps
        self.upperbound = upper if (lower or upper) else 1/eps
        constrained_varkeys = constraints.constrained_varkeys()
        self.bound_varkeys = frozenset(vk for vk in constrained_varkeys
                                       if vk not in constraints.substitutions)
        bounding_constraints = varkey_bounds(self.bound_varkeys,
                                             self.lowerbound, self.upperbound)
        super(Bounded, self).__init__([constraints, bounding_constraints])

    def sens_from_dual(self, las, nus, result):
        "Return sensitivities while capturing the relevant lambdas"
        n = bool(self.lowerbound) + bool(self.upperbound)
        self.bound_las = las[-n*len(self.bound_varkeys):]
        return super(Bounded, self).sens_from_dual(las, nus, result)

    def process_result(self, result):
        "Add boundedness to the model's solution"
        ConstraintSet.process_result(self, result)
        if "boundedness" not in result:
            result["boundedness"] = {}
        for key, value in self.check_boundaries(result).items():
            if key not in result["boundedness"]:
                result["boundedness"][key] = value
            else:
                result["boundedness"][key].update(value)

    def check_boundaries(self, result):
        "Creates (and potentially prints) a dictionary of unbounded variables."
        out = defaultdict(set)
        for i, varkey in enumerate(self.bound_varkeys):
            value = mag(result["variables"][varkey])
            if self.bound_las:
                # TODO: support sensitive-to bounds for SPs
                #       by using named variables, returning las,
                #       or pulling from self.las?
                if self.lowerbound and self.upperbound:
                    lam_gt, lam_lt = self.bound_las[2*i], self.bound_las[2*i+1]
                elif self.lowerbound:
                    lam_lt = self.bound_las[i]
                elif self.upperbound:
                    lam_gt = self.bound_las[i]
            if self.lowerbound:
                if self.bound_las:
                    if abs(lam_lt) >= 1e-7:  # arbitrary sens threshold
                        out["sensitive to lower bound"].add(varkey)
                distance_below = np.log(value/self.lowerbound)
                if distance_below <= 3:  # arbitrary dist threshold
                    out["value near lower bound"].add(varkey)
            if self.upperbound:
                if self.bound_las:
                    if abs(lam_gt) >= 1e-7:  # arbitrary sens threshold
                        out["sensitive to upper bound"].add(varkey)
                distance_above = np.log(self.upperbound/value)
                if distance_above <= 3:  # arbitrary dist threshold
                    out["value near upper bound"].add(varkey)
        if self.verbosity > 0 and out:
            print
            print "Solves with these variables bounded:"
            for key, value in out.items():
                print "% 25s: %s" % (key, ", ".join(map(str, value)))
            print
        return out

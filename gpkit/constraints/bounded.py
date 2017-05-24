"Implements Bounded"
from collections import defaultdict
import numpy as np

from .. import Variable
from .set import ConstraintSet
from ..small_scripts import mag
from ..small_classes import Quantity


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
        units = varkey.units if isinstance(varkey.units, Quantity) else 1
        constraint = []
        if upper:
            constraint.append(upper >= variable/units)
        if lower:
            constraint.append(variable/units >= lower)
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
        self.bounded_varkeys = tuple(vk for vk in constraints.varkeys
                                     if vk not in constraints.substitutions)
        bounding_constraints = varkey_bounds(self.bounded_varkeys,
                                             self.lowerbound, self.upperbound)
        super(Bounded, self).__init__([constraints, bounding_constraints])

    def sens_from_dual(self, las, nus):
        "Return sensitivities while capturing the relevant lambdas"
        n = bool(self.lowerbound) + bool(self.upperbound)
        self.bound_las = las[-n*len(self.bounded_varkeys):]
        return super(Bounded, self).sens_from_dual(las, nus)

    # pylint: disable=too-many-branches
    def process_result(self, result):
        "Creates (and potentially prints) a dictionary of unbounded variables."
        ConstraintSet.process_result(self, result)
        out = defaultdict(list)
        for i, varkey in enumerate(self.bounded_varkeys):
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
                        out["sensitive to lower bound"].append(varkey)
                distance_below = np.log(value/self.lowerbound)
                if distance_below <= 3:  # arbitrary dist threshold
                    out["value near lower bound"].append(varkey)
            if self.upperbound:
                if self.bound_las:
                    if abs(lam_gt) >= 1e-7:  # arbitrary sens threshold
                        out["sensitive to upper bound"].append(varkey)
                distance_above = np.log(self.upperbound/value)
                if distance_above <= 3:  # arbitrary dist threshold
                    out["value near upper bound"].append(varkey)
        if self.verbosity > 0 and out:
            print
            print "Solves with these variables bounded:"
            for key, value in out.items():
                print "% 25s: %s" % (key, value)
            print
        if "boundedness" not in result:
            result["boundedness"] = {}
        result["boundedness"].update(out)

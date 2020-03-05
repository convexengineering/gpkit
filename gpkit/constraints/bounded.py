"Implements Bounded"
from collections import defaultdict
import numpy as np
from .. import Variable
from .set import ConstraintSet


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
        if lower:
            constraint.append(lower <= variable)
        if upper:
            constraint.append(variable <= upper)
        constraints.append(constraint)
    return constraints


class Bounded(ConstraintSet):
    """Bounds contained variables so as to ensure dual feasibility.

    Arguments
    ---------
    constraints : iterable
        constraints whose varkeys will be bounded

    verbosity : int (default 1)
        how detailed of a warning to print
            0: nothing
            1: print warnings

    eps : float (default 1e-30)
        default lower bound is eps, upper bound is 1/eps

    lower : float (default None)
        lower bound for all varkeys, replaces eps

    upper : float (default None)
        upper bound for all varkeys, replaces 1/eps
    """
    sens_threshold = 1e-7
    logtol_threshold = 3

    def __init__(self, constraints, *, verbosity=1,
                 eps=1e-30, lower=None, upper=None):
        if not isinstance(constraints, ConstraintSet):
            constraints = ConstraintSet(constraints)
        self.verbosity = verbosity
        self.lowerbound = lower if (lower or upper) else eps
        self.upperbound = upper if (lower or upper) else 1/eps
        constrained_varkeys = constraints.constrained_varkeys()
        self.bound_varkeys = frozenset(vk for vk in constrained_varkeys
                                       if vk not in constraints.substitutions)
        bounding_constraints = varkey_bounds(self.bound_varkeys,
                                             self.lowerbound, self.upperbound)
        super().__init__({"original constraints": constraints,
                          "variable bounds": bounding_constraints})

    def process_result(self, result):
        "Add boundedness to the model's solution"
        ConstraintSet.process_result(self, result)
        if "boundedness" not in result:
            result["boundedness"] = {}
        result["boundedness"].update(
            self.check_boundaries(result, verbosity=self.verbosity))

    def check_boundaries(self, result, *, verbosity=0):
        "Creates (and potentially prints) a dictionary of unbounded variables."
        out = defaultdict(set)
        for i, varkey in enumerate(self.bound_varkeys):
            value = result["variables"][varkey]
            c_senss = [result["sensitivities"]["constraints"].get(c, 0)
                       for c in self["variable bounds"][i]]
            if self.lowerbound:
                if c_senss[0] >= self.sens_threshold:
                    out["sensitive to lower bound"].add(varkey)
                if np.log(value/self.lowerbound) <= self.logtol_threshold:
                    out["value near lower bound"].add(varkey)
            if self.upperbound:
                if c_senss[-1] >= self.sens_threshold:
                    out["sensitive to upper bound"].add(varkey)
                if np.log(self.upperbound/value) <= self.logtol_threshold:
                    out["value near upper bound"].add(varkey)
        if verbosity > 0 and out:
            print("")
            print("Solves with these variables bounded:")
            for key, value in sorted(out.items()):
                print("% 25s: %s" % (key, ", ".join([str(v) for v in value])))
            print("")
        return out

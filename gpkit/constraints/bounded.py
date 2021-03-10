"Implements Bounded"
from collections import defaultdict
import numpy as np
from .. import Variable
from .set import ConstraintSet
from ..small_scripts import appendsolwarning, initsolwarning


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
        if variable.units:  # non-dimensionalize the variable monomial
            variable.units = variable.hmap.units = None
        constraint = []
        if lower:
            constraint.append(lower <= variable)
        if upper:
            constraint.append(variable <= upper)
        constraints.append(constraint)
    return constraints


class Bounded(ConstraintSet):
    """Bounds contained variables, generally ensuring dual feasibility.

    Arguments
    ---------
    constraints : iterable
        constraints whose varkeys will be bounded

    eps : float (default 1e-30)
        default lower bound is eps, upper bound is 1/eps

    lower : float (default None)
        lower bound for all varkeys, replaces eps

    upper : float (default None)
        upper bound for all varkeys, replaces 1/eps
    """
    sens_threshold = 1e-7
    logtol_threshold = 3

    def __init__(self, constraints, *, eps=1e-30, lower=None, upper=None):
        if not isinstance(constraints, ConstraintSet):
            constraints = ConstraintSet(constraints)
        self.lowerbound = lower or eps
        self.upperbound = upper or 1/eps
        constrained_varkeys = constraints.constrained_varkeys()
        self.bound_varkeys = frozenset(vk for vk in constrained_varkeys
                                       if vk not in constraints.substitutions)
        bounding_constraints = varkey_bounds(self.bound_varkeys,
                                             self.lowerbound, self.upperbound)
        super().__init__({"original constraints": constraints,
                          "variable bounds": bounding_constraints})

    def process_result(self, result):
        "Add boundedness to the model's solution"
        super().process_result(result)
        if "boundedness" not in result:
            result["boundedness"] = {}
        result["boundedness"].update(self.check_boundaries(result))

    def check_boundaries(self, result):
        "Creates (and potentially prints) a dictionary of unbounded variables."
        out = defaultdict(set)
        initsolwarning(result, "Arbitrarily Bounded Variables")
        for i, varkey in enumerate(self.bound_varkeys):
            value = result["variables"][varkey]
            c_senss = [result["sensitivities"]["constraints"].get(c, 0)
                       for c in self["variable bounds"][i]]
            if self.lowerbound:
                bound = "lower bound of %.2g" % self.lowerbound
                if c_senss[0] >= self.sens_threshold:
                    out["sensitive to " + bound].add(varkey)
                if np.log(value/self.lowerbound) <= self.logtol_threshold:
                    out["value near " + bound].add(varkey)
            if self.upperbound:
                bound = "upper bound of %.2g" % self.upperbound
                if c_senss[-1] >= self.sens_threshold:
                    out["sensitive to " + bound].add(varkey)
                if np.log(self.upperbound/value) <= self.logtol_threshold:
                    out["value near " + bound].add(varkey)
        for bound, vks in out.items():
            msg = "% 34s: %s" % (bound, ", ".join([str(v) for v in vks]))
            appendsolwarning(msg, out, result,
                             "Arbitrarily Bounded Variables")
        return out

"Implements BoundedConstraintSet"
from collections import defaultdict
import numpy as np

from .. import Variable
from .set import ConstraintSet
from ..small_scripts import mag


def add_bounds(varkeys, lower, upper):
    "Returns constraints list bounding all varkeys"
    constraints = []
    for varkey in varkeys:
        units = varkey.descr.get("units", 1)
        constraints.append([upper*units >= Variable(**varkey.descr),
                            Variable(**varkey.descr) >= lower*units])
    return constraints


class BoundedConstraintSet(ConstraintSet):
    "Bounds contained variables so as to ensure dual feasibility."

    def __init__(self, constraints, substitutions=None, verbosity=1,
                 eps=1e-30, lower=None, upper=None):
        self.bound_las = None
        self.verbosity = verbosity
        self.lowerbound = lower if lower else eps
        self.upperbound = upper if upper else 1/eps
        constraintset = ConstraintSet([constraints])
        self.bounded_varkeys = tuple(vk for vk in constraintset.varkeys
                                     if "value" not in vk.descr)
        bounding_constraints = add_bounds(self.bounded_varkeys,
                                          self.lowerbound, self.upperbound)
        constraints = [bounding_constraints, constraintset]
        super(BoundedConstraintSet, self).__init__(constraints, substitutions)

    def sens_from_dual(self, las, nus):
        "Return sensitivities while capturing the relevant lambdas"
        self.bound_las = las[:2*len(self.bounded_varkeys)]
        return super(BoundedConstraintSet, self).sens_from_dual(las, nus)

    def process_result(self, result):
        "Creates (and potentially prints) a dictionary of unbounded variables."
        lam = self.bound_las
        out = defaultdict(list)
        for i, varkey in enumerate(self.bounded_varkeys):
            lam_gt, lam_lt = lam[2*i], lam[2*i+1]
            if abs(lam_gt) >= 1e-7:  # arbitrary threshold
                out["sensitive to upper bound"].append(varkey)
            if abs(lam_lt) >= 1e-7:  # arbitrary threshold
                out["sensitive to lower bound"].append(varkey)
            value = mag(result["variables"][varkey])
            distance_below = np.log(value/self.lowerbound)
            distance_above = np.log(self.upperbound/value)
            if distance_below <= 3:  # arbitrary threshold
                out["value near lower bound"].append(varkey)
            elif distance_above <= 3:  # arbitrary threshold
                out["value near upper bound"].append(varkey)
        if self.verbosity > 0:
            if out:
                print
                print "UNBOUNDED VARIABLES"
                for key, value in out.items():
                    print "% 25s: %s" % (key, value)
                print
        if not "boundedness" in result:
            result["boundedness"] = {}
        result["boundedness"].update(out)

"Implements Loose"
from .set import ConstraintSet
from ..small_scripts import appendsolwarning


class Loose(ConstraintSet):
    "ConstraintSet whose inequalities must result in an equality."
    senstol = 1e-5
    raiseerror = False

    def __init__(self, constraints, *, senstol=None):
        super().__init__(constraints)
        self.senstol = senstol or self.senstol

    def process_result(self, result):
        "Checks that all constraints are satisfied with equality"
        super().process_result(result)
        for constraint in self.flat():
            cstr = ("Constraint [ %.100s... %s %.100s... )"
                    % (constraint.left, constraint.oper, constraint.right))
            if not hasattr(constraint, "relax_sensitivity"):
                print("%s lacks a `relax_sensitivity` parameter and"
                      " so can't be checked for looseness." % cstr)
                continue
            if constraint.relax_sensitivity >= self.senstol:
                msg = ("%s is not loose: it has a sensitivity of %+.4g."
                       " (Allowable sensitivity: %.4g)" %
                       (cstr, constraint.relax_sensitivity, self.senstol))
                appendsolwarning(msg, constraint, result,
                                 "Unexpectedly Tight Constraints")
                if self.raiseerror:
                    raise RuntimeWarning(msg)

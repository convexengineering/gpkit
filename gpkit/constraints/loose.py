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
            c_senss = result["sensitivities"]["constraints"].get(constraint, 0)
            if c_senss >= self.senstol:
                cstr = ("Constraint [ %.100s... %s %.100s... )"
                        % (constraint.left, constraint.oper, constraint.right))
                msg = ("%s is not loose: it has a sensitivity of %+.4g."
                       " (Allowable sensitivity: %.4g)" %
                       (cstr, c_senss, self.senstol))
                constraint.relax_sensitivity = c_senss
                appendsolwarning(msg, constraint, result,
                                 "Unexpectedly Tight Constraints")
                if self.raiseerror:
                    raise RuntimeWarning(msg)

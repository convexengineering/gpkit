"Implements Loose"
from .set import ConstraintSet
from ..small_scripts import appendsolwarning, initsolwarning


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
        initsolwarning(result, "Unexpectedly Tight Constraints")
        if "sensitivities" not in result:
            appendsolwarning("Could not evaluate due to choice variables.",
                             (), result, "Unexpectedly Tight Constraints")
            return
        for constraint in self.flat():
            c_senss = result["sensitivities"]["constraints"].get(constraint, 0)
            if c_senss >= self.senstol:
                # pylint: disable=consider-using-f-string
                cstr = ("Constraint [ %.100s... %s %.100s... )"
                        % (constraint.left, constraint.oper, constraint.right))
                msg = ("%s is not loose: it has a sensitivity of %+.4g."
                       " (Allowable sensitivity: %.4g)" %
                       (cstr, c_senss, self.senstol))
                appendsolwarning(msg, (c_senss, constraint), result,
                                 "Unexpectedly Tight Constraints")
                if self.raiseerror:
                    raise RuntimeWarning(msg)

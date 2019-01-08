"Implements Loose"
from .set import ConstraintSet


class Loose(ConstraintSet):
    "ConstraintSet whose inequalities must result in an equality."
    senstol = 1e-5

    def __init__(self, constraints, senstol=None, raiseerror=False,
                 printwarning=False):
        super(Loose, self).__init__(constraints)
        if senstol:
            self.senstol = senstol
        self.raiseerror = raiseerror
        self.printwarning = printwarning

    def process_result(self, result):
        "Checks that all constraints are satisfied with equality"
        super(Loose, self).process_result(result)
        for constraint in self.flat(constraintsets=False):
            if not hasattr(constraint, "relax_sensitivity"):
                print ("Constraint [%.100s... %s %.100s...] does not have a"
                       " `relax_sensitivity` parameter and can't be checked"
                       " for looseness."
                       % (constraint.left, constraint.oper, constraint.right))
                continue
            if constraint.relax_sensitivity >= self.senstol:
                msg = ("Constraint [%.100s... %s %.100s...] is not loose"
                       " because it has a sensitivity of %+.4g."
                       " (Allowable sensitivity: %.4g)" %
                       (constraint.left, constraint.oper, constraint.right,
                        constraint.relax_sensitivity, self.senstol))
                if self.raiseerror:
                    raise ValueError(msg)
                else:
                    if self.printwarning:
                        print "Warning: %s\n" % msg
                    if "warnings" not in result:
                        result["warnings"] = {}
                    if "loose" not in result["warnings"]:
                        result["warnings"]["loose"] = []
                    result["warnings"]["loose"].append(msg)

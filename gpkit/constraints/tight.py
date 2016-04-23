"Implements TightConstraintSet"
from .set import ConstraintSet
from ..nomials import PosynomialInequality, SignomialInequality
from ..small_scripts import mag


class TightConstraintSet(ConstraintSet):
    "ConstraintSet whose inequalities must result in an equality."

    def __init__(self, constraints, substitutions=None, reltol=1e-6,
                 raiseerror=False):
        self.reltol = reltol
        self.raiseerror = raiseerror
        super(TightConstraintSet, self).__init__(constraints, substitutions)

    def process_result(self, result):
        "Checks that all constraints are satisfied with equality"
        super(TightConstraintSet, self).process_result(result)
        variables = result["variables"]
        for constraint in self.flat():
            if isinstance(constraint, (PosynomialInequality,
                                       SignomialInequality)):
                leftsubbed = constraint.left.sub(variables).value
                rightsubbed = constraint.right.sub(variables).value
                rel_diff = abs(1 - leftsubbed/rightsubbed)
                if rel_diff >= self.reltol:
                    msg = ("Constraint [%.30s...] is not tight because "
                           "the left hand side evaluated to %s but "
                           "the right hand side evaluated to %s "
                           "(Allowable error: %s%%, Actual error: %.2g%%)\n" %
                           (constraint, leftsubbed, rightsubbed,
                            self.reltol*100, mag(rel_diff)*100))
                    if self.raiseerror:
                        raise ValueError(msg)
                    else:
                        print "Warning: %s" % msg

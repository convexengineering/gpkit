"Implements Tight"
from .set import ConstraintSet
from ..nomials import PosynomialInequality, SignomialInequality
from ..small_scripts import mag
from .. import SignomialsEnabled


class Tight(ConstraintSet):
    "ConstraintSet whose inequalities must result in an equality."

    def __init__(self, constraints, substitutions=None, reltol=1e-6,
                 raiseerror=False):
        super(Tight, self).__init__(constraints, substitutions)
        self.reltol = reltol
        self.raiseerror = raiseerror

    def process_solution(self, sol):
        "Checks that all constraints are satisfied with equality"
        super(Tight, self).process_solution(sol)
        variables = sol["variables"]
        for constraint in self.flat(constraintsets=False):
            rel_diff = 0
            if isinstance(constraint, PosynomialInequality):
                leftsubbed = constraint.left.sub(variables).value
                rightsubbed = constraint.right.sub(variables).value
                rel_diff = abs(1 - leftsubbed/rightsubbed)
            elif isinstance(constraint, SignomialInequality):
                siglt0, = constraint.unsubbed
                posy, negy = siglt0.posy_negy()
                posy = posy.sub(variables).value
                negy = negy.sub(variables).value
                rel_diff = abs(1 - posy/negy)
                if rel_diff >= self.reltol:
                    # do another substitution for the sake of printing
                    with SignomialsEnabled():
                        leftsubbed = constraint.left.sub(variables).value
                        rightsubbed = constraint.right.sub(variables).value
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


TightConstraintSet = Tight

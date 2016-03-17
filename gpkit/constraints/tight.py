"Implements tight constraint set"
from .set import ConstraintSet
from ..nomials import PosynomialInequality, SignomialInequality

class TightConstraintSet(ConstraintSet):
    """A constraint set for which the inequality constraints must be
    satisfied with equality"""

    def process_result(self, result):
        "Checks that all constraints are satisfied with equality"
        ConstraintSet.process_result(self, result)
        variables = result["variables"]
        for constraint in self.flat():
            if isinstance(constraint, (PosynomialInequality,
                                       SignomialInequality)):
                leftsubbed = constraint.left.sub(variables)
                rightsubbed = constraint.right.sub(variables)
                if leftsubbed != rightsubbed:
                    raise ValueError("Tightness requirement not met"
                                     " for constraint %s because %s"
                                     " evaluated to %s but %s evaluated"
                                     " to %s"
                                     % (constraint,
                                        constraint.left, leftsubbed,
                                        constraint.right, rightsubbed))


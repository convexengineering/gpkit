"Implements TightConstraintSet"
from .set import ConstraintSet
from ..nomials import PosynomialInequality, SignomialInequality


class TightConstraintSet(ConstraintSet):
    "ConstraintSet whose inequalities must result in an equality."

    def __init__(self, constraints, substitutions=None, reltol=1e-6):
        self.reltol = reltol
        super(TightConstraintSet, self).__init__(constraints, substitutions)

    def process_result(self, result):
        "Checks that all constraints are satisfied with equality"
        super(TightConstraintSet, self).process_result(result)
        variables = result["variables"]
        for constraint in self.flat():
            if isinstance(constraint, (PosynomialInequality,
                                       SignomialInequality)):
                leftsubbed = constraint.left.sub(variables)
                rightsubbed = constraint.right.sub(variables)
                rel_diff = abs(1-(leftsubbed/rightsubbed).value)
                if rel_diff >= self.reltol:
                    raise ValueError("Tightness requirement not met"
                                     " for constraint %s because %s"
                                     " evaluated to %s but %s evaluated"
                                     " to %s"
                                     % (constraint,
                                        constraint.left, leftsubbed,
                                        constraint.right, rightsubbed))

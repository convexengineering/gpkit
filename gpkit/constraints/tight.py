"Implements Tight"
from .set import ConstraintSet
from ..nomials import PosynomialInequality, SignomialInequality
from ..small_scripts import mag
from ..small_scripts import appendsolwarning
from .. import SignomialsEnabled


class Tight(ConstraintSet):
    "ConstraintSet whose inequalities must result in an equality."
    reltol = 1e-3

    def __init__(self, constraints, *, reltol=None, **kwargs):
        super().__init__(constraints)
        self.reltol = reltol or self.reltol
        self.__dict__.update(kwargs)  # NOTE: for Berk's use in labelling

    def process_result(self, result):
        "Checks that all constraints are satisfied with equality"
        super().process_result(result)
        variables = result["variables"]
        for constraint in self.flat():
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
                msg = ("Constraint [%.100s... %s %.100s...] is not tight:"
                       " the left hand side evaluated to %s but"
                       " the right hand side evaluated to %s"
                       " (Allowable error: %s%%, Actual error: %.2g%%)" %
                       (constraint.left, constraint.oper, constraint.right,
                        leftsubbed, rightsubbed,
                        self.reltol*100, mag(rel_diff)*100))
                if hasattr(leftsubbed, "magnitude"):
                    rightsubbed = rightsubbed.to(leftsubbed.units).magnitude
                    leftsubbed = leftsubbed.magnitude
                constraint.tightvalues = (leftsubbed, constraint.oper,
                                          rightsubbed)
                constraint.rel_diff = rel_diff
                appendsolwarning(msg, constraint, result,
                                 "Unexpectedly Loose Constraints")

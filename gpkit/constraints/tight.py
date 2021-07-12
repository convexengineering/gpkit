"Implements Tight"
from .set import ConstraintSet
from ..small_scripts import mag
from ..small_scripts import appendsolwarning, initsolwarning
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
        initsolwarning(result, "Unexpectedly Loose Constraints")
        for constraint in self.flat():
            with SignomialsEnabled():
                leftval = constraint.left.sub(variables).value
                rightval = constraint.right.sub(variables).value
            rel_diff = mag(abs(1 - leftval/rightval))
            if rel_diff >= self.reltol:
                msg = ("Constraint [%.100s... %s %.100s...] is not tight:"
                       " the left hand side evaluated to %s but"
                       " the right hand side evaluated to %s"
                       " (Allowable error: %s%%, Actual error: %.2g%%)" %
                       (constraint.left, constraint.oper, constraint.right,
                        leftval, rightval,
                        self.reltol*100, mag(rel_diff)*100))
                if hasattr(leftval, "magnitude"):
                    rightval = rightval.to(leftval.units).magnitude
                    leftval = leftval.magnitude
                tightvalues = (leftval, constraint.oper, rightval)
                appendsolwarning(msg, (rel_diff, tightvalues, constraint),
                                 result, "Unexpectedly Loose Constraints")

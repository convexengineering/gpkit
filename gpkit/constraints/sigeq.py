"Implements SignomialEquality"
from .set import ConstraintSet
from ..nomials import SingleSignomialEquality
from ..nomials.array import array_constraint


class SignomialEquality(ConstraintSet):
    "A constraint of the general form posynomial == posynomial"

    def __init__(self, left, right):
        # TODO: really it should be easier to vectorize a constraint
        if hasattr(left, "shape"):
            cns = array_constraint("=", SingleSignomialEquality)(left, right)
        elif hasattr(right, "shape"):
            cns = array_constraint("=", SingleSignomialEquality)(right, left)
        else:
            cns = [SingleSignomialEquality(left, right)]
        ConstraintSet.__init__(self, cns)

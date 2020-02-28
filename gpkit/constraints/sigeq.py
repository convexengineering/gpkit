"Implements SignomialEquality"
from .set import ConstraintSet
from ..nomials import SingleSignomialEquality as SSE
from ..nomials.array import array_constraint


class SignomialEquality(ConstraintSet):
    "A constraint of the general form posynomial == posynomial"

    def __init__(self, left, right):
        if hasattr(left, "shape") or hasattr(right, "shape"):
            ConstraintSet.__init__(self,
                                   array_constraint("=", SSE)(left, right))
        else:
            ConstraintSet.__init__(self, [SSE(left, right)])

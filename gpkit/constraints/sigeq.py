"Implements SignomialEquality"
from .set import ConstraintSet
from ..nomials import SingleSignomialEquality
from ..nomials.array import array_constraint


class SignomialEquality(ConstraintSet):
    "A constraint of the general form posynomial == posynomial"

    def __init__(self, left, right):
        if hasattr(left, "shape"):
            cns = array_constraint("=", SingleSignomialEquality)(left, right)
        elif hasattr(right, "shape"):
            cns = array_constraint("=", SingleSignomialEquality)(right, left)
        else:
            cns = [SingleSignomialEquality(left, right)]  # pylint: disable=redefined-variable-type
        ConstraintSet.__init__(self, cns)

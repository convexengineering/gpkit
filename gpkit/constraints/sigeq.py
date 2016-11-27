"Implements SignomialEquality"
from .set import ConstraintSet
from ..nomials import SingleSignomialEquality
from ..nomials.array import array_constraint


class SignomialEquality(ConstraintSet):
    "A constraint of the general form posynomial == posynomial"

    def __init__(self, left, right):
        if hasattr(left, "shape"):
            constraints = array_constraint("=", SingleSignomialEquality)(left, right)
        else:
            constraints = [SingleSignomialEquality(left, right)]  # pylint: disable=redefined-variable-type
        ConstraintSet.__init__(self, constraints)

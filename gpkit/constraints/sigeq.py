from .set import ConstraintSet
from ..nomials import SignomialEquality
from ..nomials.array import array_constraint


sigeq_vectorized = array_constraint("=", SignomialEquality)


class SignomialEqualityConstraint(ConstraintSet):
    def __init__(self, left, right):
        if hasattr(left, "shape"):
            constraints = sigeq_vectorized(left, right)
        else:
            constraints = [SignomialEquality(left, right)]
        ConstraintSet.__init__(self, constraints)

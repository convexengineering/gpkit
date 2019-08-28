"Implements SignomialEquality"
from __future__ import unicode_literals
from .set import ConstraintSet
from ..nomials import SingleSignomialEquality
from ..nomials.array import array_constraint


class SignomialEquality(ConstraintSet):
    "A constraint of the general form posynomial == posynomial"

    def __init__(self, left, right):
        if hasattr(left, "shape") or hasattr(right, "shape"):
            cns = array_constraint("=", SingleSignomialEquality)(left, right)
        else:
            cns = [SingleSignomialEquality(left, right)]
        ConstraintSet.__init__(self, cns)

"Implements ArrayConstraint"
from __future__ import unicode_literals
from .set import ConstraintSet
from .single_equation import SingleEquationConstraint


class ArrayConstraint(SingleEquationConstraint, ConstraintSet):
    """A ConstraintSet for prettier array-constraint printing.

    ArrayConstraint gets its `sub` method from ConstrainSet,
    and so `left` and `right` are only used for printing.

    When created by NomialArray `left` and `right` are likely to be
    be either NomialArrays or Varkeys of VectorVariables.
    """
    def __init__(self, constraints, left, oper, right):
        SingleEquationConstraint.__init__(self, left, oper, right)
        ConstraintSet.__init__(self, constraints)

    def __nonzero__(self):
        "Allows the use of '=' NomialArrays as truth elements."
        if self.oper != "=":
            return NotImplemented
        return all(bool(p) for p in self.flat())

    def __bool__(self):
        "Allows the use of NomialArrays as truth elements in python3."
        return self.__nonzero__()

"Implements ArrayConstraint"
from .set import ConstraintSet
from .single_equation import SingleEquationConstraint


class ArrayConstraint(ConstraintSet, SingleEquationConstraint):
    """A ConstraintSet for prettier array-constraint printing.

    ArrayConstraint inherits its `sub` method from ConstrainSet,
    and so left and right are only used for printing.

    When created by NomialArray left and right are likely to be
    be either NomialArrays or Varkeys of VectorVariables.
    """
    def __init__(self, constraints, left, oper, right):
        super(ArrayConstraint, self).__init__(constraints)
        self.left = left
        self.oper = oper
        self.right = right

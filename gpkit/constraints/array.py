"Implements ArrayConstraint"
from .set import ConstraintSet
from .single_equation import SingleEquationConstraint


class ArrayConstraint(ConstraintSet, SingleEquationConstraint):
    """A ConstraintSet for prettier array-constraint printing.

    Because ArrayConstraint also inherits from ConstraintSet,
    its substitutions is recursive and so left and right are
    only used for string and latex printing. In practice they're
    likely to be either NomialArrays or Varkeys of VectorVariables."""
    def __init__(self, constraints, left, oper, right):
        super(ArrayConstraint, self).__init__(constraints)
        self.left = left
        self.oper = oper
        self.right = right

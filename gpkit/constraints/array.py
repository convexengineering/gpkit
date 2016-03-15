"Implements ArrayConstraint"
from .set import ConstraintSet
from .single_equation import SingleEquationConstraint


class ArrayConstraint(SingleEquationConstraint, ConstraintSet):
    """A ConstraintSet for prettier array-constraint printing.

    ArrayConstraint gets its `sub` method from ConstrainSet,
    and so left and right are only used for printing.

    When created by NomialArray left and right are likely to be
    be either NomialArrays or Varkeys of VectorVariables.
    """
    def __init__(self, constraints, left, oper, right):
        SingleEquationConstraint.__init__(self, left, oper, right)
        ConstraintSet.__init__(self, constraints)

    subinplace = ConstraintSet.subinplace

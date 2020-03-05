"Implements ArrayConstraint"
from .single_equation import SingleEquationConstraint


class ArrayConstraint(SingleEquationConstraint, list):
    """A ConstraintSet for prettier array-constraint printing.

    ArrayConstraint gets its `sub` method from ConstrainSet,
    and so `left` and `right` are only used for printing.

    When created by NomialArray `left` and `right` are likely to be
    be either NomialArrays or Varkeys of VectorVariables.
    """
    def __init__(self, constraints, left, oper, right):
        self.constraints = constraints
        list.__init__(self, constraints)
        SingleEquationConstraint.__init__(self, left, oper, right)

    def __iter__(self):
        yield from self.constraints.flat

    def lines_without(self, excluded):
        "Returns lines for indentation in hierarchical printing."
        return self.str_without(excluded).split("\n")

    def __bool__(self):
        "Allows the use of '=' NomialArrays as truth elements."
        return False if self.oper != "=" else bool(self.constraints.all())

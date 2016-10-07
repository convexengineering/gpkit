"""Models for assessing primal feasibility"""
from .model import Model
from .set import ConstraintSet
from ..nomials import Variable, VectorVariable, parse_subs, NomialArray
from ..keydict import FastKeyDict


class RelaxedAllConstraints(Model):
    """Relax constraints the same amount, as in Eqn. 10 of [Boyd2007].

    Arguments
    ---------
    constraints : iterable
        Constraints which will be relaxed (made easier).

    varname : str
        LaTeX name of relaxation variable.


    Attributes
    ----------
    relaxvar : Variable
        The variable controlling the relaxation. A solved value of 1 means no
        relaxation. Higher values indicate the amount by which all constraints
        have been made easier: e.g., a value of 1.5 means all constraints were
        50 percent easier in the final solution than in the original problem.

    [Boyd2007] : "A tutorial on geometric programming", Optim Eng 8:67-122

    """

    def __init__(self, constraints):
        if not isinstance(constraints, ConstraintSet):
            constraints = ConstraintSet(constraints)
        substitutions = dict(constraints.substitutions)
        posynomials = constraints.as_posyslt1()
        self.relaxvar = Variable("C")
        Model.__init__(self, self.relaxvar,
                       [[posy <= self.relaxvar
                         for posy in posynomials],
                        self.relaxvar >= 1],
                       substitutions)


class RelaxedConstraints(Model):
    """Relax constraints optimally, as in Eqn. 11 of [Boyd2007].

    Arguments
    ---------
    constraints : iterable
        Constraints which will be relaxed (made easier).


    Attributes
    ----------
    relaxvars : Variable
        The variables controlling the relaxation. A solved value of 1 means no
        relaxation was necessary or optimal for a particular constraint.
        Higher values indicate the amount by which that constraint has been
        made easier: e.g., a value of 1.5 means it was made 50 percent easier
        in the final solution than in the original problem.

    [Boyd2007] : "A tutorial on geometric programming", Optim Eng 8:67-122

    """

    def __init__(self, constraints):
        if not isinstance(constraints, ConstraintSet):
            constraints = ConstraintSet(constraints)
        substitutions = dict(constraints.substitutions)
        posynomials = constraints.as_posyslt1()
        N = len(posynomials)
        self.relaxvars = VectorVariable(N, "C")
        Model.__init__(self, self.relaxvars.prod(),
                       [[posynomials <= self.relaxvars],
                        self.relaxvars >= 1],
                       substitutions)


class RelaxedConstants(Model):
    """Relax constraints optimally, as in Eqn. 11 of [Boyd2007].

    Arguments
    ---------
    constraints : iterable
        Constraints which will be relaxed (made easier).

    include_only : set
        if declared, variable names must be on this list to be relaxed

    exclude : set
        if declared, variable names on this list will never be relaxed


    Attributes
    ----------
    relaxvars : Variable
        The variables controlling the relaxation. A solved value of 1 means no
        relaxation was necessary or optimal for a particular constant.
        Higher values indicate the amount by which that constant has been
        made easier: e.g., a value of 1.5 means it was made 50 percent easier
        in the final solution than in the original problem. Of course, this
        can also be determined by looking at the constant's new value directly.
    """
    def __init__(self, constraints, include_only=None, exclude=None):
        exclude = frozenset(exclude) if exclude else frozenset()
        include_only = frozenset(include_only) if include_only else frozenset()
        if not isinstance(constraints, ConstraintSet):
            constraints = ConstraintSet(constraints)
        substitutions = FastKeyDict(constraints.substitutions)
        constants, _, _ = parse_subs(constraints.varkeys,
                                     constraints.substitutions)
        self.relaxvars, relaxation_constraints = [], []
        self.origvars = []
        for key, value in constants.items():
            if include_only and key.name not in include_only:
                continue
            if key.name in exclude:
                continue
            del substitutions[key]
            original = constraints[key]
            self.origvars.append(original)
            if original.units and not hasattr(value, "units"):
                value *= original.units
            descr = dict(key.descr)
            descr.pop("value", None)
            descr["units"] = "-"
            relaxation = Variable(**descr)
            self.relaxvars.append(relaxation)
            relaxation_constraints.append([value/relaxation <= original,
                                           original <= value*relaxation,
                                           relaxation >= 1])
        Model.__init__(self, NomialArray(self.relaxvars).prod(),
                       [constraints, relaxation_constraints])
        self.substitutions = substitutions

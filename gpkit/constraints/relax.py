"""Models for assessing primal feasibility"""
from .set import ConstraintSet
from ..nomials import Variable, VectorVariable, parse_subs, NomialArray
from ..nomials import Monomial
from ..keydict import KeyDict
from ..small_classes import HashVector
from .. import NamedVariables, MODELNUM_LOOKUP


class ConstraintsRelaxedEqually(ConstraintSet):
    """Relax constraints the same amount, as in Eqn. 10 of [Boyd2007].

    Arguments
    ---------
    constraints : iterable
        Constraints which will be relaxed (made easier).


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
        with NamedVariables("Relax"):
            self.relaxvar = Variable("C")
        ConstraintSet.__init__(self,
                               [[posy <= self.relaxvar
                                 for posy in posynomials],
                                self.relaxvar >= 1],
                               substitutions)


class ConstraintsRelaxed(ConstraintSet):
    """Relax constraints, as in Eqn. 11 of [Boyd2007].

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
        with NamedVariables("Relax"):
            self.relaxvars = VectorVariable(N, "C")
        ConstraintSet.__init__(self,
                               [[posynomials <= self.relaxvars],
                                self.relaxvars >= 1],
                               substitutions)


class ConstantsRelaxed(ConstraintSet):
    """Relax constants in a constraintset.

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
        if not isinstance(constraints, ConstraintSet):
            constraints = ConstraintSet(constraints)
        exclude = frozenset(exclude) if exclude else frozenset()
        include_only = frozenset(include_only) if include_only else frozenset()
        substitutions = KeyDict(constraints.substitutions)
        constants, _, _ = parse_subs(constraints.varkeys,
                                     constraints.substitutions)
        relaxvars, relaxation_constraints = [], []
        self.origvars = []
        self.num = MODELNUM_LOOKUP["Relax"]
        MODELNUM_LOOKUP["Relax"] += 1
        for key, value in constants.items():
            if value == 0:
                continue
            if include_only and key.name not in include_only:
                continue
            if key.name in exclude:
                continue
            descr = dict(key.descr)
            descr.pop("value", None)
            descr["units"] = "-"
            descr["models"] = descr.pop("models", [])+["Relax"]
            descr["modelnums"] = descr.pop("modelnums", []) + [self.num]
            relaxation = Variable(**descr)
            relaxvars.append(relaxation)
            del substitutions[key]
            original = Variable(**key.descr)
            self.origvars.append(original)
            if original.units and not hasattr(value, "units"):
                value *= original.units
            value = Monomial(value)  # convert for use in constraint
            relaxation_constraints.append([relaxation >= 1,
                                           value/relaxation <= original,
                                           original <= value*relaxation])
        self.relaxvars = NomialArray(relaxvars)
        ConstraintSet.__init__(self, [constraints, relaxation_constraints])
        self.substitutions = substitutions

    def sens_from_dual(self, las, nus):
        """Computes constraint and variable sensitivities from dual solution

        Replaces the sensitivity of the relaxed constants (which, as a free
        variable, is 0) with the sensitivity of the relaxed constant's value.
        """
        var_senss = HashVector()
        offset = 0
        for i, constr in enumerate(self):
            n_posys = self.posymap[i]
            la = las[offset:offset+n_posys]
            nu = nus[offset:offset+n_posys]
            v_ss = constr.sens_from_dual(la, nu)
            var_senss += v_ss
            offset += n_posys

        var_senss += {var.key: las[-3*i - 2] - las[-3*i - 1]
                      for i, var in enumerate(reversed(self.origvars))}
        return var_senss

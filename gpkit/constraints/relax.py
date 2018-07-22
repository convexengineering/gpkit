"""Models for assessing primal feasibility"""
from .set import ConstraintSet
from ..nomials import Variable, VectorVariable, parse_subs, NomialArray
from ..keydict import KeyDict
from .. import NamedVariables, MODELNUM_LOOKUP
from gpkit import SignomialsEnabled

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
        relconstraints = []
        self.origconstrs = []
        with NamedVariables("Relax"):
            self.relaxvar = Variable("C")
        with SignomialsEnabled():
            for constraint in constraints.flat(constraintsets=False):
                self.origconstrs.append(constraint)
                relconstraints.append(constraint.relaxed(self.relaxvar))
        ConstraintSet.__init__(self, [relconstraints,
                                      self.relaxvar >= 1], substitutions)

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
        relconstraints = []
        self.origconstrs = []
        with NamedVariables("Relax"):
            self.relaxvars = VectorVariable(len(constraints), "C")
        with SignomialsEnabled():
            for i, constraint in enumerate(
                    constraints.flat(constraintsets=False)):
                self.origconstrs.append(constraint)
                relconstraints.append(constraint.relaxed(self.relaxvars[i]))
        ConstraintSet.__init__(self, [relconstraints,
                                      self.relaxvars >= 1], substitutions)

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
    def __init__(self, constraints, include_only=None, exclude=None):  # pylint:disable=too-many-locals
        if not isinstance(constraints, ConstraintSet):
            constraints = ConstraintSet(constraints)
        exclude = frozenset(exclude) if exclude else frozenset()
        include_only = frozenset(include_only) if include_only else frozenset()
        substitutions = KeyDict(constraints.substitutions)
        constants, _, linked = parse_subs(constraints.varkeys, substitutions)
        constrained_varkeys = constraints.constrained_varkeys()
        if linked:
            kdc = KeyDict(constants)
            combined = {k: f(kdc) for k, f in linked.items()
                        if k in constrained_varkeys}
            combined.update({k: v for k, v in constants.items()
                             if k in constrained_varkeys})
        else:
            combined = constants
        self.constants = KeyDict(combined)
        relaxvars, relaxation_constraints = [], []
        self.origvars = []
        self.num = MODELNUM_LOOKUP["Relax"]
        self._unrelaxmap = {}
        MODELNUM_LOOKUP["Relax"] += 1
        for key, value in combined.items():
            if value == 0:
                continue
            elif include_only and key.name not in include_only:
                continue
            elif key.name in exclude:
                continue
            descr = key.descr.copy()
            descr.pop("value", None)
            descr.pop("veckey", None)
            descr["models"] = descr.pop("models", [])+["Relax"]
            descr["modelnums"] = descr.pop("modelnums", []) + [self.num]
            relaxvardescr = descr.copy()
            relaxvardescr["unitrepr"] = "-"
            relaxvar = Variable(**relaxvardescr)
            relaxvars.append(relaxvar)
            del substitutions[key]
            var = Variable(**key.descr)
            self.origvars.append(var)
            unrelaxeddescr = descr.copy()
            unrelaxeddescr["name"] += "_{before}"
            unrelaxed = Variable(**unrelaxeddescr)
            self._unrelaxmap[unrelaxed.key] = key
            substitutions[unrelaxed] = value
            relaxation_constraints.append([relaxvar >= 1,
                                           unrelaxed/relaxvar <= var,
                                           var <= unrelaxed*relaxvar])
        self.relaxvars = NomialArray(relaxvars)
        ConstraintSet.__init__(self, [constraints, relaxation_constraints])
        self.substitutions = substitutions

    def process_result(self, result):
        ConstraintSet.process_result(self, result)
        csenss = result["sensitivities"]["constants"]
        for const, origvar in self._unrelaxmap.items():
            csenss[origvar] = csenss[const]
            del csenss[const]

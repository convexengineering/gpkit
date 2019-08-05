"""Models for assessing primal feasibility"""
from __future__ import unicode_literals
from .set import ConstraintSet
from ..nomials import Variable, VectorVariable, parse_subs, NomialArray
from ..keydict import KeyDict
from .. import NamedVariables, SignomialsEnabled


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
            for constraint in constraints.flat():
                self.origconstrs.append(constraint)
                relconstraints.append(constraint.relaxed(self.relaxvar))
        ConstraintSet.__init__(self, {
            "relaxed constraints": relconstraints,
            "minimum relaxation": self.relaxvar >= 1}, substitutions)


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
            for i, constraint in enumerate(constraints.flat()):
                self.origconstrs.append(constraint)
                relconstraints.append(constraint.relaxed(self.relaxvars[i]))
        ConstraintSet.__init__(self, {
            "relaxed constraints": relconstraints,
            "minimum relaxation": self.relaxvars >= 1}, substitutions)


class ConstantsRelaxed(ConstraintSet):
    """Relax constants in a constraintset.

    Arguments
    ---------
    constraints : iterable
        Constraints which will be relaxed (made easier).

    include_only : set (optional)
        variable names must be in this set to be relaxed

    exclude : set (optional)
        variable names in this set will never be relaxed


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
    # pylint:disable=too-many-locals
    def __init__(self, constraints, include_only=None, exclude=None):
        if not isinstance(constraints, ConstraintSet):
            constraints = ConstraintSet(constraints)
        exclude = frozenset(exclude) if exclude else frozenset()
        include_only = frozenset(include_only) if include_only else frozenset()
        substitutions = KeyDict(constraints.substitutions)
        constants, _, linked = parse_subs(constraints.varkeys, substitutions)
        constrained_varkeys = constraints.constrained_varkeys()
        if linked:
            kdc = KeyDict(constants)
            constants.update({k: f(kdc) for k, f in linked.items()
                              if k in constrained_varkeys})
        self.constants = constants
        relaxvars, self.origvars, relaxation_constraints = [], [], {}
        with NamedVariables("Relax") as (self.lineage, _):
            pass
        self._unrelaxmap = {}
        for key, value in constants.items():
            if value == 0:
                continue
            elif include_only and key.name not in include_only:
                continue
            elif key.name in exclude:
                continue
            key.descr.pop("gradients", None)
            descr = key.descr.copy()
            descr.pop("veckey", None)
            descr["lineage"] = descr.pop("lineage", ())+(self.lineage[-1],)
            relaxvardescr = descr.copy()
            relaxvardescr["unitrepr"] = "-"
            relaxvar = Variable(**relaxvardescr)
            relaxvars.append(relaxvar)
            del substitutions[key]
            var = Variable(**key.descr)
            self.origvars.append(var)
            unrelaxeddescr = descr.copy()
            unrelaxeddescr["lineage"] += (("OriginalValues", 0),)
            unrelaxed = Variable(**unrelaxeddescr)
            self._unrelaxmap[unrelaxed.key] = key
            substitutions[unrelaxed] = value
            relaxation_constraints[str(key)] = [relaxvar >= 1,
                                                unrelaxed/relaxvar <= var,
                                                var <= unrelaxed*relaxvar]
        self.relaxvars = NomialArray(relaxvars)
        ConstraintSet.__init__(self, {
            "original constraints": constraints,
            "relaxation constraints": relaxation_constraints})
        self.substitutions = substitutions

    def process_result(self, result):
        ConstraintSet.process_result(self, result)
        csenss = result["sensitivities"]["constants"]
        for const, origvar in self._unrelaxmap.items():
            csenss[origvar] = csenss[const]
            del csenss[const]

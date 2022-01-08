"""Models for assessing primal feasibility"""
from .set import ConstraintSet
from ..nomials import Variable, VectorVariable, parse_subs, NomialArray
from ..keydict import KeyDict
from .. import NamedVariables, SignomialsEnabled
from ..small_scripts import appendsolwarning, initsolwarning, mag


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

    def __init__(self, original_constraints):
        if not isinstance(original_constraints, ConstraintSet):
            original_constraints = ConstraintSet(original_constraints)
        self.original_constraints = original_constraints
        original_substitutions = original_constraints.substitutions

        with NamedVariables("Relax"):
            self.relaxvar = Variable("C")
        with SignomialsEnabled():
            relaxed_constraints = [c.relaxed(self.relaxvar)
                                   for c in original_constraints.flat()]

        ConstraintSet.__init__(self, {
            "minimum relaxation": self.relaxvar >= 1,
            "relaxed constraints": relaxed_constraints}, original_substitutions)

    def process_result(self, result):
        "Warns if any constraints were relaxed"
        super().process_result(result)
        self.check_relaxed(result)

    def check_relaxed(self, result):
        "Adds relaxation warnings to the result"
        initsolwarning(result, "Relaxed Constraints")
        for val, msg in get_relaxed([result["freevariables"][self.relaxvar]],
                                    ["All constraints relaxed by %i%%"]):
            appendsolwarning(msg % (0.9+(val-1)*100), self, result,
                             "Relaxed Constraints")


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

    def __init__(self, original_constraints):
        if not isinstance(original_constraints, ConstraintSet):
            original_constraints = ConstraintSet(original_constraints)
        self.original_constraints = original_constraints
        original_substitutions = original_constraints.substitutions
        with NamedVariables("Relax"):
            self.relaxvars = VectorVariable(len(original_constraints), "C")

        with SignomialsEnabled():
            relaxed_constraints = [
                c.relaxed(self.relaxvars[i])
                for i, c in enumerate(original_constraints.flat())]

        ConstraintSet.__init__(self, {
            "minimum relaxation": self.relaxvars >= 1,
            "relaxed constraints": relaxed_constraints}, original_substitutions)

    def process_result(self, result):
        "Warns if any constraints were relaxed"
        super().process_result(result)
        self.check_relaxed(result)

    def check_relaxed(self, result):
        "Adds relaxation warnings to the result"
        relaxed = get_relaxed(result["freevariables"][self.relaxvars],
                              range(len(self["relaxed constraints"])))
        initsolwarning(result, "Relaxed Constraints")
        for relaxval, i in relaxed:
            relax_percent = "%i%%" % (0.5+(relaxval-1)*100)
            oldconstraint = self.original_constraints[i]
            newconstraint = self["relaxed constraints"][i][0]
            subs = {self.relaxvars[i]: relaxval}
            relaxdleft = newconstraint.left.sub(subs)
            relaxdright = newconstraint.right.sub(subs)
            oldleftstr = str(oldconstraint.left)
            relaxedleftstr = str(relaxdleft)
            padding = len(relaxedleftstr) - len(oldleftstr)
            if padding > 0:
                oldleftstr = " " * padding + oldleftstr
            elif padding < 0:
                relaxedleftstr = " " * padding + relaxedleftstr
            msg = (" %3i: %5s relaxed, from %s %s %s\n"
                   "                       to %s %s %s"
                   % (i, relax_percent, oldleftstr,
                      oldconstraint.oper, oldconstraint.right,
                      relaxedleftstr, newconstraint.oper, relaxdright))
            appendsolwarning(msg, oldconstraint, result, "Relaxed Constraints")


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
    def __init__(self, constraints, *, include_only=None, exclude=None):
        exclude = frozenset(exclude) if exclude else frozenset()
        include_only = frozenset(include_only) if include_only else frozenset()
        with NamedVariables("Relax") as (self.lineage, _):
            pass  # gives this model the correct lineage.

        if not isinstance(constraints, ConstraintSet):
            constraints = ConstraintSet(constraints)
        substitutions = KeyDict(constraints.substitutions)
        constants, _, linked = parse_subs(constraints.vks, substitutions)
        if linked:
            kdc = KeyDict(constants)
            constrained_varkeys = constraints.constrained_varkeys()
            constants.update({k: f(kdc) for k, f in linked.items()
                              if k in constrained_varkeys})

        self._derelax_map = {}
        relaxvars, self.freedvars, relaxation_constraints = [], [], {}
        for const, val in sorted(constants.items(), key=lambda i: i[0].eqstr):
            if val == 0:
                substitutions[const] = 0
                continue
            if include_only and const.name not in include_only:
                continue
            if const.name in exclude:
                continue
            # set up the lineage
            const.descr.pop("gradients", None)  # nothing wants an old gradient
            newconstd = const.descr.copy()
            newconstd.pop("veckey", None)  # only const wants an old veckey
            # everything but const wants a new lineage, to distinguish them
            newconstd["lineage"] = (newconstd.pop("lineage", ())
                                    + (self.lineage[-1],))
            # make the relaxation variable, free to get as large as it needs
            relaxedd = newconstd.copy()
            relaxedd["unitrepr"] = "-"  # and unitless, importantly
            relaxvar = Variable(**relaxedd)
            relaxvars.append(relaxvar)
            # the newly freed const can acquire a new value
            del substitutions[const]
            freed = Variable(**const.descr)
            self.freedvars.append(freed)
            # becuase the make the newconst will take its old value
            newconstd["lineage"] += (("OriginalValues", 0),)
            newconst = Variable(**newconstd)
            substitutions[newconst] = val
            self._derelax_map[newconst.key] = const
            # add constraints so the newly freed's wiggle room
            # is proportional to the value relaxvar, and it can't antirelax
            relaxation_constraints[str(const)] = [relaxvar >= 1,
                                                  newconst/relaxvar <= freed,
                                                  freed <= newconst*relaxvar]
        ConstraintSet.__init__(self, {
            "original constraints": constraints,
            "relaxation constraints": relaxation_constraints})
        self.relaxvars = NomialArray(relaxvars)  # so they can be .prod()'d
        self.substitutions = substitutions
        self.constants = constants

    def process_result(self, result):
        "Transfers constant sensitivities back to the original constants"
        super().process_result(result)
        constant_senss = result["sensitivities"]["variables"]
        for new_constant, former_constant in self._derelax_map.items():
            constant_senss[former_constant] = constant_senss[new_constant]
            del constant_senss[new_constant]
        self.check_relaxed(result)


    def check_relaxed(self, result):
        "Adds relaxation warnings to the result"
        relaxed = get_relaxed([result["freevariables"][r]
                               for r in self.relaxvars], self.freedvars)
        initsolwarning(result, "Relaxed Constants")
        for (_, freed) in relaxed:
            msg = ("  %s: relaxed from %-.4g to %-.4g"
                   % (freed,
                      mag(self.constants[freed.key]),
                      mag(result["freevariables"][freed])))
            appendsolwarning(msg, freed, result, "Relaxed Constants")


def get_relaxed(relaxvals, mapped_list):
    "Returns 'relaxed' vals (those above an arbitrary threshold of 1.01)."
    sortrelaxed = sorted(zip(relaxvals, mapped_list), key=lambda x: -x[0])
    mostrelaxed = max(sortrelaxed[0][0], 1.01)
    for i, (val, _) in enumerate(sortrelaxed):
        if val <= 1.01 and (val-1) <= (mostrelaxed-1)/10:
            return sortrelaxed[:i]
    return sortrelaxed

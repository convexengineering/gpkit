"Implement CostedConstraintSet"
from __future__ import unicode_literals
import numpy as np
from .set import ConstraintSet
from ..small_scripts import maybe_flatten


class CostedConstraintSet(ConstraintSet):
    """A ConstraintSet with a cost

    Arguments
    ---------
    cost : gpkit.Posynomial
    constraints : Iterable
    substitutions : dict (None)
    """
    def __init__(self, cost, constraints, substitutions=None):
        self.cost = maybe_flatten(cost)
        if isinstance(self.cost, np.ndarray):  # if it's still a vector
            raise ValueError("cost must be scalar, not the vector %s" % cost)
        subs = dict(self.cost.varkeyvalues())
        if substitutions:
            subs.update(substitutions)
        ConstraintSet.__init__(self, constraints, subs)

    def __bare_init__(self, cost, constraints, substitutions, varkeys=False):
        self.cost = cost
        if isinstance(constraints, dict):
            self.idxlookup = {k: i for i, k in enumerate(constraints)}
            constraints = constraints.values()
        if not isinstance(constraints, ConstraintSet):
            constraints = ConstraintSet(constraints)
        else:
            constraints = [constraints]
        list.__init__(self, constraints)
        self.substitutions = substitutions or {}
        if varkeys:
            self.reset_varkeys()

    def constrained_varkeys(self):
        "Return all varkeys in the cost and non-ConstraintSet constraints"
        constrained_varkeys = ConstraintSet.constrained_varkeys(self)
        constrained_varkeys.update(self.cost.varkeys)
        return constrained_varkeys

    def reset_varkeys(self):
        "Resets varkeys to what is in the cost and constraints"
        ConstraintSet.reset_varkeys(self)
        self.varkeys.update(self.cost.varkeys)

    def rootconstr_str(self, excluded=()):
        "String showing cost, to be used when this is the top constraint"
        description = ["", "Cost", "----",
                       " %s" % self.cost.str_without(excluded),
                       "", "Constraints", "-----------"]
        if getattr(self, "lineage", None):
            name, num = self.lineage[-1]
            fullname = "%s" % (name if not num else name + str(num))
            description = [fullname, "="*len(fullname)] + description
        return description

    def rootconstr_latex(self, excluded=()):
        "Latex showing cost, to be used when this is the top constraint"
        return "\n".join(["\\text{minimize}",
                          "    & %s \\\\" % self.cost.latex(excluded),
                          "\\text{subject to}"])

"Implement CostedConstraintSet"
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
    lineage = None

    def __init__(self, cost, constraints, substitutions=None):
        self.cost = maybe_flatten(cost)
        if isinstance(self.cost, np.ndarray):  # if it's still a vector
            raise ValueError("Cost must be scalar, not the vector %s." % cost)
        subs = dict(self.cost.varkeyvalues())
        if substitutions:
            subs.update(substitutions)
        ConstraintSet.__init__(self, constraints, subs)
        self.varkeys.update(self.cost.varkeys)

    def __bare_init__(self, cost, constraints, substitutions):
        self.cost = cost
        self.substitutions = substitutions or {}
        if not isinstance(constraints, ConstraintSet):
            if isinstance(constraints, dict):
                self.idxlookup = {k: i for i, k in enumerate(constraints)}
                constraints = constraints.values()
            constraints = ConstraintSet(constraints)
            list.__init__(self, constraints)
        else:
            list.__init__(self, [constraints])
        self.varkeys = constraints.varkeys
        self.varkeys.update(cost.varkeys)

    def constrained_varkeys(self):
        "Return all varkeys in the cost and non-ConstraintSet constraints"
        constrained_varkeys = ConstraintSet.constrained_varkeys(self)
        constrained_varkeys.update(self.cost.varkeys)
        return constrained_varkeys

    def _rootlines(self, excluded=()):
        "String showing cost, to be used when this is the top constraint"
        description = ["", "Cost", "----",
                       " %s" % self.cost.str_without(excluded),
                       "", "Constraints", "-----------"]
        if self.lineage:
            name, num = self.lineage[-1]  # pylint: disable=unsubscriptable-object
            fullname = "%s" % (name if not num else name + str(num))
            description = [fullname, "="*len(fullname)] + description
        return description

    def _rootlatex(self, excluded=()):
        "Latex showing cost, to be used when this is the top constraint"
        return "\n".join(["\\text{minimize}",
                          "    & %s \\\\" % self.cost.latex(excluded),
                          "\\text{subject to}"])

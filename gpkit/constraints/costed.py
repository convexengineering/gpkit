"Implement CostedConstraintSet"
import numpy as np
from .set import ConstraintSet
from ..small_scripts import maybe_flatten
from ..repr_conventions import lineagestr


class CostedConstraintSet(ConstraintSet):
    """A ConstraintSet with a cost

    Arguments
    ---------
    cost : gpkit.Posynomial
    constraints : Iterable
    substitutions : dict
    """
    lineage = None

    def __init__(self, cost, constraints, substitutions=None):
        self.cost = maybe_flatten(cost)
        if isinstance(self.cost, np.ndarray):  # if it's still a vector
            raise ValueError("Cost must be scalar, not the vector %s." % cost)
        subs = {k: k.value for k in self.cost.vks if "value" in k.descr}
        if substitutions:
            subs.update(substitutions)
        ConstraintSet.__init__(self, constraints, subs, bonusvks=self.cost.vks)

    def constrained_varkeys(self):
        "Return all varkeys in the cost and non-ConstraintSet constraints"
        constrained_varkeys = ConstraintSet.constrained_varkeys(self)
        constrained_varkeys.update(self.cost.vks)
        return constrained_varkeys

    def _rootlines(self, excluded=()):
        "String showing cost, to be used when this is the top constraint"
        if self.cost.vks:
            description = ["", "Cost Function", "-------------",
                           " %s" % self.cost.str_without(excluded),
                           "", "Constraints", "-----------"]
        else:   # don't print the cost if it's a constant
            description = ["", "Constraints", "-----------"]
        if self.lineage:
            fullname = lineagestr(self)
            description = [fullname, "="*len(fullname)] + description
        return description

    def _rootlatex(self, excluded=()):
        "Latex showing cost, to be used when this is the top constraint"
        return "\n".join(["\\text{minimize}",
                          "    & %s \\\\" % self.cost.latex(excluded),
                          "\\text{subject to}"])

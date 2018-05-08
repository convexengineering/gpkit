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
    substitutions : dict
    """
    def __init__(self, cost, constraints, substitutions=None):
        self.cost = maybe_flatten(cost)
        if isinstance(self.cost, np.ndarray):  # if it's still a vector
            raise ValueError("cost must be scalar, not the vector %s" % cost)
        subs = dict(self.cost.values)
        if substitutions:
            subs.update(substitutions)
        ConstraintSet.__init__(self, constraints, subs)

    def __bare_init__(self, cost, constraints, substitutions, varkeys=False):
        self.cost = cost
        if not isinstance(constraints, ConstraintSet):
            constraints = ConstraintSet(constraints)
        else:
            constraints = [constraints]
        list.__init__(self, constraints)
        self.substitutions = substitutions or {}
        if varkeys:
            self.reset_varkeys()

    def subinplace(self, subs):
        "Substitutes in place."
        self.cost = self.cost.sub(subs)
        ConstraintSet.subinplace(self, subs)

    def constrained_varkeys(self):
        "Return all varkeys in the cost and non-ConstraintSet constraints"
        constrained_varkeys = ConstraintSet.constrained_varkeys(self)
        constrained_varkeys.update(self.cost.varkeys)
        return constrained_varkeys

    def reset_varkeys(self):
        "Resets varkeys to what is in the cost and constraints"
        ConstraintSet.reset_varkeys(self)
        self.varkeys.update(self.cost.vks)

    def rootconstr_str(self, excluded=None):
        "String showing cost, to be used when this is the top constraint"
        return "\n".join(["  # minimize",
                          "        %s" % self.cost.str_without(excluded),
                          "  # subject to"])

    def rootconstr_latex(self, excluded=None):
        "Latex showing cost, to be used when this is the top constraint"
        return "\n".join(["\\text{minimize}",
                          "    & %s \\\\" % self.cost.latex(excluded),
                          "\\text{subject to}"])

    def interact(self, ranges=None, fn_of_sol=None, **solvekwargs):
        """Easy model interaction in IPython / Jupyter

        By default, this creates a model with sliders for every constant
        which prints a new solution table whenever the sliders are changed.

        Arguments
        ---------
        fn_of_sol : function
            The function called with the solution after each solve that
            displays the result. By default prints a table.

        ranges : dictionary {str: Slider object or tuple}
            Determines which sliders get created. Tuple values may contain
            two or three floats: two correspond to (min, max), while three
            correspond to (min, step, max)

        **solvekwargs
            kwargs which get passed to the solve()/localsolve() method.
        """
        from ..interactive.widgets import modelinteract
        return modelinteract(self, ranges, fn_of_sol, **solvekwargs)

    def controlpanel(self, *args, **kwargs):
        """Easy model control in IPython / Jupyter

        Like interact(), but with the ability to control sliders and their
        ranges live. args and kwargs are passed on to interact()
        """
        from ..interactive.widgets import modelcontrolpanel
        return modelcontrolpanel(self, *args, **kwargs)

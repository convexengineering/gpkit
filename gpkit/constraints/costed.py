"Implement CostedConstraintSet"
from .set import ConstraintSet


class CostedConstraintSet(ConstraintSet):
    """
    A ConstraintSet with a cost

    Arguments
    ---------
    cost: gpkit.Posynomial
    constraints: Iterable
    substitutions: dict
    """
    def __init__(self, cost, constraints, substitutions=None):
        if hasattr(cost, "prod"):
            # it's a vector!
            if not cost.shape:
                cost, = cost.flatten()
            else:
                raise ValueError("cost must be a scalar, not the vector %s"
                                 % cost)
        self.cost = cost
        subs = dict(self.cost.values)
        if substitutions:
            subs.update(substitutions)
        ConstraintSet.__init__(self, constraints, subs)

    def subinplace(self, subs):
        "Substitutes in place."
        self.cost = self.cost.sub(subs)
        ConstraintSet.subinplace(self, subs)

    def reset_varkeys(self, init_dict=None):
        "Resets varkeys to what is in the cost and constraints"
        ConstraintSet.reset_varkeys(self, self.cost.varlocs)
        if init_dict is not None:
            self.varkeys.update(init_dict)

    def rootconstr_str(self, excluded=None):
        "The appearance of a ConstraintSet in addition to its contents"
        return "\n".join(["  # minimize",
                          "        %s" % self.cost.str_without(excluded),
                          "  # subject to"])

    def rootconstr_latex(self, excluded=None):
        "The appearance of a ConstraintSet in addition to its contents"
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
        from ..interactive import modelinteract
        return modelinteract(self, ranges, fn_of_sol, **solvekwargs)

    def controlpanel(self, *args, **kwargs):
        """Easy model control in IPython / Jupyter

        Like interact(), but with the ability to control sliders and their
        ranges live. args and kwargs are passed on to interact()
        """
        from ..interactive import modelcontrolpanel
        return modelcontrolpanel(self, *args, **kwargs)

"Implements Model"
from .base import ConstraintBase
from .set import ConstraintSet
from ..geometric_program import GeometricProgram
from .signomial_program import SignomialProgram
from .link import LinkConstraint
from .prog_factories import _progify_fctry, _solve_fctry


class Model(ConstraintBase):
    "A ConstraintSet for convenient solving and setup"
    def __init__(self, cost=None, constraints=[], substitutions=None,
                 *args, **kwargs):
        if hasattr(self, "setup"):
            args = [arg for arg in [cost, constraints] if arg] + list(args)
            ConstraintBase.__init__(self, substitutions, *args, **kwargs)
        else:
            ## Support the ConstraintSet interface
            if not constraints:  # assume args were [constraints]
                constraints, cost = cost, None
            elif isinstance(constraints, dict):
                # assume args were [constraints, substitutions]
                constraints, substitutions, cost = cost, constraints, None
            if cost:
                self.cost = cost
            ConstraintSet.__init__(self, constraints, substitutions)
        if "name" in kwargs:
            self._add_models_tovars(kwargs["name"])

    def link(self, other, include_only=None, exclude=None):
        "Connects this model with a set of constraints"
        lc = LinkConstraint([self, other], include_only, exclude)
        cost = self.cost.sub(lc.linked)
        return Model(cost, [lc], lc.substitutions)

    def __and__(self, other):
        return self.link(other)

    def __or__(self, other):
        return Model(self.cost, [self, other])

    def zero_lower_unbounded_variables(self):
        "Recursively substitutes 0 for variables that lack a lower bound"
        zeros = True
        while zeros:
            bounds = self.gp(verbosity=0).missingbounds
            zeros = {var: 0 for var, bound in bounds.items()
                     if bound == "lower"}
            self.substitutions.update(zeros)

    gp = _progify_fctry(GeometricProgram)
    sp = _progify_fctry(SignomialProgram)
    solve = _solve_fctry(_progify_fctry(GeometricProgram, "solve"))
    localsolve = _solve_fctry(_progify_fctry(SignomialProgram, "localsolve"))
    #
    # def __repr__(self):
    #     return "gpkit.%s(%s)" % (self.__class__.__name__, str(self))
    #
    # def __str__(self):
    #     """String representation of a Model.
    #     Contains all of its parameters."""
    #     return "\n".join(["# minimize",
    #                       "    %s," % self.cost,
    #                       "[   # subject to"] +
    #                      ["    %s," % constr
    #                       for constr in self.constraints] +
    #                      ['],',
    #                       "    substitutions=%s" % self.allsubs])
    #
    # def latex(self, show_subs=True):
    #     """LaTeX representation of a Model.
    #     Contains all of its parameters."""
    #     # TODO: print sweeps and linkedsweeps
    #     latex_list = ["\\begin{array}{ll}",
    #                   "\\text{}",
    #                   "\\text{minimize}",
    #                   "    & %s \\\\" % self.cost.latex(),
    #                   "\\text{subject to}"]
    #     latex_list += ["    & %s \\\\" % constr.latex()
    #                    for constr in self.constraints]
    #     if show_subs:
    #         sub_latex = ["    & %s \gets %s%s \\\\" % (var.latex(),
    #                                                     latex_num(val),
    #                                                     var.unitstr)
    #                      for var, val in self.allsubs.items()]
    #         latex_list += ["\\text{substituting}"] + sorted(sub_latex)
    #     latex_list += ["\\end{array}"]
    #     return "\n".join(latex_list)

    def _repr_latex_(self):
        return "$$"+self.latex()+"$$"

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
        from .interactive.widgets import modelinteract
        return modelinteract(self, ranges, fn_of_sol, **solvekwargs)

    def controlpanel(self, *args, **kwargs):
        """Easy model control in IPython / Jupyter

        Like interact(), but with the ability to control sliders and their ranges
        live. args and kwargs are passed on to interact()
        """
        from .interactive.widgets import modelcontrolpanel
        return modelcontrolpanel(self, *args, **kwargs)

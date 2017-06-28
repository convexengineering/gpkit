"Implements Model"
import numpy as np
from .costed import CostedConstraintSet
from ..nomials import Monomial
from .prog_factories import _progify_fctry, _solve_fctry
from .geometric_program import GeometricProgram
from .signomial_program import SignomialProgram
from ..small_scripts import mag
from ..tools.autosweep import autosweep_1d
from ..exceptions import InvalidGPConstraint
from .. import NamedVariables


class Model(CostedConstraintSet):
    """Symbolic representation of an optimization problem.

    The Model class is used both directly to create models with constants and
    sweeps, and indirectly inherited to create custom model classes.

    Arguments
    ---------
    cost : Posynomial (optional)
        Defaults to `Monomial(1)`.

    constraints : ConstraintSet or list of constraints (optional)
        Defaults to an empty list.

    substitutions : dict (optional)
        This dictionary will be substituted into the problem before solving,
        and also allows the declaration of sweeps and linked sweeps.

    name : str (optional)
        Allows "naming" a model in a way similar to inherited instances,
        and overrides the inherited name if there is one.

    Attributes with side effects
    ----------------------------
    `program` is set during a solve
    `solution` is set at the end of a solve
    """

    # name and num identify a model uniquely
    name = None
    num = None
    # naming holds the name and num evironment in which a model was created
    # this includes its own name and num, and those of models containing it
    naming = None
    program = None
    solution = None

    def __init__(self, cost=None, constraints=None, *args, **kwargs):
        setup_vars = None
        substitutions = kwargs.pop("substitutions", None)  # reserved keyword
        if hasattr(self, "setup"):
            self.cost = None
            with NamedVariables(self.__class__.__name__):
                start_args = [cost, constraints]
                args = tuple(a for a in start_args if a is not None) + args
                constraints = self.setup(*args, **kwargs)  # pylint: disable=no-member
                from .. import NAMEDVARS, MODELS, MODELNUMS
                setup_vars = NAMEDVARS[tuple(MODELS), tuple(MODELNUMS)]
                self.name, self.num = MODELS[:-1], MODELNUMS[:-1]
                self.naming = (tuple(MODELS), tuple(MODELNUMS))
            cost = self.cost
        else:
            if args and not substitutions:
                # backwards compatibility: substitutions as third arg
                substitutions, = args

        cost = cost or Monomial(1)
        constraints = constraints or []
        CostedConstraintSet.__init__(self, cost, constraints, substitutions)
        if setup_vars:
            # add all the vars created in .setup to the Model's varkeys
            # even if they aren't used in any constraints
            self.unique_varkeys = frozenset(v.key for v in setup_vars)
            # TODO: is unique_varkeys really the right way to do this?
            self.reset_varkeys()

    gp = _progify_fctry(GeometricProgram)
    sp = _progify_fctry(SignomialProgram)
    solve = _solve_fctry(_progify_fctry(GeometricProgram, "solve"))
    localsolve = _solve_fctry(_progify_fctry(SignomialProgram, "localsolve"))

    def zero_lower_unbounded_variables(self):
        "Recursively substitutes 0 for variables that lack a lower bound"
        zeros = True
        while zeros:
            # pylint: disable=no-member
            bounds = self.gp(verbosity=0).missingbounds
            zeros = {var: 0 for var, bound in bounds.items()
                     if bound == "lower"}
            self.substitutions.update(zeros)

    def subconstr_str(self, excluded=None):
        "The collapsed appearance of a ConstraintBase"
        if self.name:
            return "%s_%s" % (self.name, self.num)

    def subconstr_latex(self, excluded=None):
        "The collapsed appearance of a ConstraintBase"
        if self.name:
            return "%s_{%s}" % (self.name, self.num)

    def sweep(self, sweeps, **solveargs):
        "Sweeps {var: values} pairs in sweeps. Returns swept solutions."
        sols = []
        for sweepvar, sweepvals in sweeps.items():
            original_val = self.substitutions.get(sweepvar, None)
            self.substitutions.update({sweepvar: ('sweep', sweepvals)})
            try:
                sols.append(self.solve(**solveargs))
            except InvalidGPConstraint:
                sols.append(self.localsolve(**solveargs))
            if original_val:
                self.substitutions[sweepvar] = original_val
            else:
                del self.substitutions[sweepvar]
        if len(sols) == 1:
            return sols[0]
        return sols

    def autosweep(self, sweeps, tol=0.01, samplepoints=100, **solveargs):
        """Autosweeps {var: (start, end)} pairs in sweeps to tol.

        Returns swept and sampled solutions.
        The original simplex tree can be accessed at sol.bst
        """
        sols = []
        for sweepvar, sweepvals in sweeps.items():
            sweepvar = self[sweepvar].key
            start, end = sweepvals
            bst = autosweep_1d(self, tol, sweepvar, [start, end], **solveargs)
            sols.append(bst.sample_at(np.linspace(start, end, samplepoints)))
        if len(sols) == 1:
            return sols[0]
        return sols

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def debug(self, solver=None, verbosity=1, **solveargs):
        "Attempts to diagnose infeasible models."
        from .relax import ConstantsRelaxed, ConstraintsRelaxed
        from .bounded import Bounded

        sol = None
        relaxedconsts = False

        solveargs["solver"] = solver
        solveargs["verbosity"] = verbosity

        print "> Trying to solve with bounded variables and relaxed constants"

        if self.substitutions:
            constsrelaxed = ConstantsRelaxed(Bounded(self))
            feas = Model(constsrelaxed.relaxvars.prod()**30 * self.cost,
                         constsrelaxed)
            # NOTE: It hasn't yet been seen but might be possible that
            #       the self.cost component above could cause infeasibility
        else:
            feas = Model(self.cost, Bounded(self))

        try:
            try:
                sol = feas.solve(**solveargs)
            except InvalidGPConstraint:
                sol = feas.localsolve(**solveargs)

            if self.substitutions:
                for orig in (o for o, r in zip(constsrelaxed.origvars,
                                               constsrelaxed.relaxvars)
                             if sol(r) >= 1.01):
                    if not relaxedconsts:
                        if sol["boundedness"]:
                            print("and these constants relaxed:")
                        else:
                            print("\nSolves with these constants relaxed:")
                        relaxedconsts = True
                    print("  %s: relaxed from %-.4g to %-.4g"
                          % (orig, mag(self.substitutions[orig]),
                             mag(sol(orig))))
            print "> ...success!"
        except (ValueError, RuntimeWarning):
            print("> ...does not solve with bounded variables"
                  " and relaxed constants.")
        print "\n> Trying to solve with relaxed constraints"

        try:
            constrsrelaxed = ConstraintsRelaxed(self)
            feas = Model(constrsrelaxed.relaxvars.prod()**30 * self.cost,
                         constrsrelaxed)
            try:
                sol_constraints = feas.solve(**solveargs)
            except InvalidGPConstraint:
                sol_constraints = feas.localsolve(**solveargs)

            relaxvals = sol_constraints(constrsrelaxed.relaxvars)
            if any(rv >= 1.01 for rv in relaxvals):
                if sol_constraints["boundedness"]:
                    print("and these constraints relaxed:")
                else:
                    print("\nSolves with relaxed constraints:")
                    if not relaxedconsts:
                        # then this is the only solution we have to return
                        sol = sol_constraints
            iterator = enumerate(zip(relaxvals, feas[0][0][0]))
            for i, (relaxval, constraint) in iterator:
                if relaxval >= 1.01:
                    relax_percent = "%i%%" % (0.5+(relaxval-1)*100)
                    print("  %i: %4s relaxed  Canonical form: %s <= %.2f)"
                          % (i, relax_percent, constraint.right, relaxval))
            print "> ...success!"
        except (ValueError, RuntimeWarning):
            print("> ...does not solve with relaxed constraints.")

        return sol

"Implements Model"
from .costed import CostedConstraintSet
from ..nomials import Monomial
from .prog_factories import _progify_fctry, _solve_fctry
from ..geometric_program import GeometricProgram
from .signomial_program import SignomialProgram
from .linked import LinkedConstraintSet
from ..small_scripts import mag
from .. import end_variable_naming, begin_variable_naming, NamedVariables


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

    def __new__(cls, *args, **kwargs):
        # Implemented for backwards compatibility with v0.4
        obj = super(Model, cls).__new__(cls, *args, **kwargs)
        if cls.__name__ != "Model" and not hasattr(cls, "setup"):
            obj.name = cls.__name__
            obj.num, obj.naming = begin_variable_naming(obj.name)
        return obj

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
            if self.__class__.__name__ != "Model":
                from .. import NAMEDVARS, MODELS, MODELNUMS
                setup_vars = NAMEDVARS[tuple(MODELS), tuple(MODELNUMS)]
                end_variable_naming()
                if setup_vars:
                    print("Declaring a named Model's variables in __init__ is"
                          " not recommended. For details see gpkit.rtfd.org")
                    # backwards compatibility: don't add unused vars
                    setup_vars = None

        cost = cost if cost else Monomial(1)
        constraints = constraints if constraints else []
        CostedConstraintSet.__init__(self, cost, constraints, substitutions)
        if setup_vars:
            # add all the vars created in .setup to the Model's varkeys
            # even if they aren't used in any constraints
            self.unused_variables = setup_vars
            self.reset_varkeys()

    gp = _progify_fctry(GeometricProgram)
    sp = _progify_fctry(SignomialProgram)
    solve = _solve_fctry(_progify_fctry(GeometricProgram, "solve"))
    localsolve = _solve_fctry(_progify_fctry(SignomialProgram, "localsolve"))

    def link(self, other, include_only=None, exclude=None):
        "Connects this model with a set of constraints"
        lc = LinkedConstraintSet([self, other], include_only, exclude)
        cost = self.cost.sub(lc.linked)
        return Model(cost, lc, lc.substitutions)

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

    # pylint: disable=too-many-locals
    def debug(self, verbosity=1, **solveargs):
        "Attempts to diagnose infeasible models."
        from .relax import ConstantsRelaxed, ConstraintsRelaxed
        from .bounded import Bounded

        sol = None
        relaxedconsts = False

        print "Debugging..."
        print "_____________________"

        if self.substitutions:
            constsrelaxed = ConstantsRelaxed(Bounded(self))
            feas = Model(constsrelaxed.relaxvars.prod()**30 * self.cost,
                         constsrelaxed)
            # NOTE: It hasn't yet been seen but might be possible that
            #       the self.cost component above could cause infeasibility
        else:
            feas = Model(self.cost, Bounded(self))

        try:
            sol = feas.solve(verbosity=verbosity, **solveargs)

            if self.substitutions:
                for orig in (o for o, r in zip(constsrelaxed.origvars,
                                               constsrelaxed.relaxvars)
                             if sol(r) >= 1.01):
                    if not relaxedconsts:
                        if sol["boundedness"]:
                            print "and these constants relaxed:"
                        else:
                            print
                            print "Solves with these constants relaxed:"
                        relaxedconsts = True
                    print ("  %s: relaxed from %-.4g to %-.4g"
                           % (orig, mag(self.substitutions[orig]),
                              mag(sol(orig))))
        except (ValueError, RuntimeWarning):
            print
            print ("Model does not solve with bounded variables"
                   " and relaxed constants.")
        print "_____________________"

        try:
            constrsrelaxed = ConstraintsRelaxed(self)
            feas = Model(constrsrelaxed.relaxvars.prod()**30 * self.cost,
                         constrsrelaxed)
            sol_constraints = feas.solve(verbosity=verbosity, **solveargs)

            relaxvals = sol_constraints(constrsrelaxed.relaxvars)
            if any(rv >= 1.01 for rv in relaxvals):
                if sol_constraints["boundedness"]:
                    print "and these constraints relaxed:"
                else:
                    print
                    print "Solves with relaxed constraints:"
                    if not relaxedconsts:
                        # then this is the only solution we have to return
                        sol = sol_constraints
            iterator = enumerate(zip(relaxvals, feas[0][0][0]))
            for i, (relaxval, constraint) in iterator:
                if relaxval >= 1.01:
                    relax_percent = "%i%%" % (0.5+(relaxval-1)*100)
                    print ("  %i: %4s relaxed  Canonical form: %s <= %.2f)"
                           % (i, relax_percent, constraint.right, relaxval))

        except (ValueError, RuntimeWarning):
            print
            print ("Model does not solve with relaxed constraints.")

        print "_____________________"
        return sol

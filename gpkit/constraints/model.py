"Implements Model"
import numpy as np
from .costed import CostedConstraintSet
from ..nomials import Monomial
from .prog_factories import progify, solvify
from .gp import GeometricProgram
from .sgp import SequentialGeometricProgram
from ..tools.autosweep import autosweep_1d
from ..exceptions import InvalidGPConstraint
from .. import NamedVariables
from ..tools.docstring import expected_unbounded
from .set import add_meq_bounds
from ..exceptions import Infeasible


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

    Attributes with side effects
    ----------------------------
    `program` is set during a solve
    `solution` is set at the end of a solve
    """
    program = None
    solution = None

    def __init__(self, cost=None, constraints=None, *args, **kwargs):
        setup_vars = None
        substitutions = kwargs.pop("substitutions", None)  # reserved keyword
        if hasattr(self, "setup"):
            self.cost = None
            # lineage holds the (name, num) environment a model was created in,
            # including its own (name, num), and those of models above it
            with NamedVariables(self.__class__.__name__) as (self.lineage,
                                                             setup_vars):
                args = tuple(arg for arg in [cost, constraints]
                             if arg is not None) + args
                cs = self.setup(*args, **kwargs)  # pylint: disable=no-member
                if (isinstance(cs, tuple) and len(cs) == 2
                        and isinstance(cs[1], dict)):
                    constraints, substitutions = cs
                else:
                    constraints = cs
            cost = self.cost
        elif args and not substitutions:
            # backwards compatibility: substitutions as third argument
            substitutions, = args

        cost = cost or Monomial(1)
        constraints = constraints or []
        if setup_vars:
            # add all the vars created in .setup to the Model's varkeys
            # even if they aren't used in any constraints
            self.unique_varkeys = frozenset(v.key for v in setup_vars)
        CostedConstraintSet.__init__(self, cost, constraints, substitutions)
        docstr = self.__class__.__doc__
        if self.lineage and docstr and "SKIP VERIFICATION" not in docstr:
            if "Unbounded" in docstr or "Bounded by" in docstr:
                self.verify_docstring()

    gp = progify(GeometricProgram)
    solve = solvify(progify(GeometricProgram, "solve"))

    sp = progify(SequentialGeometricProgram)
    localsolve = solvify(progify(SequentialGeometricProgram, "localsolve"))

    def verify_docstring(self):  # pylint:disable=too-many-locals,too-many-branches,too-many-statements
        "Verifies docstring bounds are sufficient but not excessive."
        err = "while verifying %s:\n" % self.__class__.__name__
        bounded, meq_bounded = self.bounded.copy(), self.meq_bounded.copy()
        doc = self.__class__.__doc__
        exp_unbounds = expected_unbounded(self, doc)
        unexp_bounds = bounded.intersection(exp_unbounds)
        if unexp_bounds:  # anything bounded that shouldn't be? err!
            for direction in ["lower", "upper"]:
                badvks = [v for v, d in unexp_bounds if d == direction]
                if not badvks:
                    continue
                badvks = ", ".join(str(v) for v in badvks)
                badvks += (" were" if len(badvks) > 1 else " was")
                err += ("    %s %s-bounded; expected %s-unbounded"
                        "\n" % (badvks, direction, direction))
            raise ValueError(err)
        bounded.update(exp_unbounds)  # if not, treat expected as bounded
        add_meq_bounds(bounded, meq_bounded)  # and add more meqs
        self.missingbounds = {}  # now let's figure out what's missing
        for bound in meq_bounded:  # first add the un-dealt-with meq bounds
            for condition in list(meq_bounded[bound]):
                meq_bounded[bound].remove(condition)
                newcond = condition - bounded
                if newcond and not any(c.issubset(newcond)
                                       for c in meq_bounded[bound]):
                    meq_bounded[bound].add(newcond)
            bsets = " or ".join(str(list(c)) for c in meq_bounded[bound])
            self.missingbounds[bound] = (", but would gain it from any of"
                                         " these sets of bounds: " + bsets)
        # then add everything that's not in bounded
        if len(bounded)+len(self.missingbounds) != 2*len(self.varkeys):
            for key in self.varkeys:
                for bound in ("upper", "lower"):
                    if (key, bound) not in bounded:
                        if (key, bound) not in self.missingbounds:
                            self.missingbounds[(key, bound)] = ""
        if self.missingbounds:  # anything unbounded? err!
            boundstrs = "\n".join("  %s has no %s bound%s" % (v, b, x)
                                  for (v, b), x
                                  in self.missingbounds.items())
            docstring = ("To fix this add the following to %s's"
                         " docstring (you may not need it all):"
                         " \n" % self.__class__.__name__)
            for direction in ["upper", "lower"]:
                mb = [k for (k, b) in self.missingbounds if b == direction]
                if mb:
                    docstring += """
%s Unbounded
---------------
%s
""" % (direction.title(), ", ".join(set(k.name for k in mb)))
            raise ValueError(err + boundstrs + "\n\n" + docstring)

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
        return sols if len(sols) > 1 else sols[0]

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
        return sols if len(sols) > 1 else sols[0]

    def debug(self, solver=None, verbosity=1, **solveargs):
        "Attempts to diagnose infeasible models."
        from .relax import ConstantsRelaxed, ConstraintsRelaxed
        from .bounded import Bounded

        sol = None
        solveargs["solver"] = solver
        solveargs["verbosity"] = verbosity - 1
        solveargs["process_result"] = False

        bounded = Bounded(self)
        tants = ConstantsRelaxed(bounded)
        if tants.relaxvars.size:
            feas = Model(tants.relaxvars.prod()**30 * self.cost, tants)
        else:
            feas = Model(self.cost, bounded)

        try:
            try:
                sol = feas.solve(**solveargs)
            except InvalidGPConstraint:
                sol = feas.sp(use_pccp=False).localsolve(**solveargs)
            # limited results processing
            bounded.check_boundaries(sol)
            tants.check_relaxed(sol)
        except Infeasible:
            if verbosity:
                print("<DEBUG> Model is not feasible with relaxed constants"
                      " and bounded variables.")
            traints = ConstraintsRelaxed(self)
            feas = Model(traints.relaxvars.prod()**30 * self.cost, traints)
            try:
                try:
                    sol = feas.solve(**solveargs)
                except InvalidGPConstraint:
                    sol = feas.sp(use_pccp=False).localsolve(**solveargs)
                # limited results processing
                traints.check_relaxed(sol)
            except Infeasible:
                print("<DEBUG> Model is not feasible with bounded constraints.")
        if sol and verbosity:
            warnings = sol.table(tables=["warnings"]).split("\n")[3:-2]
            if warnings:
                print("<DEBUG> Model is feasible with these modifications:")
                print("\n" + "\n".join(warnings) + "\n")
            else:
                print("<DEBUG> Model seems feasible without modification,"
                      " or only needs relaxations of less than 1%."
                      " Check the returned solution for details.")
        return sol

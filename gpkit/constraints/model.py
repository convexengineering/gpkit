"Implements Model"
import numpy as np
from .costed import CostedConstraintSet
from ..nomials import Monomial
from .prog_factories import progify, solvify
from .gp import GeometricProgram
from .sgp import SequentialGeometricProgram
from ..small_scripts import mag
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

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def debug(self, solver=None, verbosity=1, **solveargs):
        "Attempts to diagnose infeasible models."
        from .relax import ConstantsRelaxed, ConstraintsRelaxed
        from .bounded import Bounded

        sol = None
        solveargs["solver"] = solver
        solveargs["verbosity"] = verbosity - 1
        solveargs["process_result"] = False

        if verbosity:
            print("< DEBUGGING >")
            print("> Trying with bounded variables and relaxed constants:")

        bounded = Bounded(self)
        if self.substitutions:
            tants = ConstantsRelaxed(bounded)
            feas = Model(tants.relaxvars.prod()**30 * self.cost, tants)
        else:
            feas = Model(self.cost, bounded)

        try:
            try:
                sol = feas.solve(**solveargs)
            except InvalidGPConstraint:
                sol = feas.sp(use_pccp=False).localsolve(**solveargs)
            sol["boundedness"] = bounded.check_boundaries(sol,
                                                          verbosity=verbosity)
            if self.substitutions:
                relaxed = get_relaxed([sol(r) for r in tants.relaxvars],
                                      tants.freedvars,
                                      min_return=0 if sol["boundedness"] else 1)
                if verbosity and relaxed:
                    if sol["boundedness"]:
                        print("and these constants relaxed:")
                    else:
                        print("\nSolves with these constants relaxed:")
                    for (_, freed) in relaxed:
                        print("  %s: relaxed from %-.4g to %-.4g"
                              % (freed, mag(tants.constants[freed.key]),
                                 mag(sol(freed))))
                    print("")
            if verbosity > 0:
                print(">> Success!")
        except Infeasible:
            if verbosity > 0:
                print(">> Failure.")
                print("> Trying with relaxed constraints:")

            try:
                traints = ConstraintsRelaxed(self)
                feas = Model(traints.relaxvars.prod()**30 * self.cost, traints)
                try:
                    sol = feas.solve(**solveargs)
                except InvalidGPConstraint:
                    sol = feas.sp(use_pccp=False).localsolve(**solveargs)
                relaxed_constraints = feas[0]["relaxed constraints"]
                relaxed = get_relaxed(sol(traints.relaxvars),
                                      range(len(relaxed_constraints)))
                if verbosity > 0 and relaxed:
                    print("\nSolves with these constraints relaxed:")
                    for relaxval, i in relaxed:
                        relax_percent = "%i%%" % (0.5+(relaxval-1)*100)
                        oldconstraint = traints.original_constraints[i]
                        newconstraint = relaxed_constraints[i][0]
                        subs = {traints.relaxvars[i]: relaxval}
                        relaxdleft = newconstraint.left.sub(subs)
                        relaxdright = newconstraint.right.sub(subs)
                        print(" %3i: %5s relaxed, from %s %s %s \n"
                              "                     to %s %s %s "
                              % (i, relax_percent, oldconstraint.left,
                                 oldconstraint.oper, oldconstraint.right,
                                 relaxdleft, newconstraint.oper, relaxdright))
                if verbosity > 0:
                    print("\n>> Success!\n")
            except (ValueError, RuntimeWarning):
                if verbosity > 0:
                    print(">> Failure\n")
        return sol


def get_relaxed(relaxvals, mapped_list, min_return=1):
    "Determines which relaxvars are considered 'relaxed'"
    sortrelaxed = sorted(zip(relaxvals, mapped_list), key=lambda x: -x[0])
    # arbitrarily, 1.01 is the threshold below which we don't show slack
    mostrelaxed = max(sortrelaxed[0][0], 1.01)
    for i, (val, _) in enumerate(sortrelaxed):
        if i >= min_return and val <= 1.01 and (val-1) <= (mostrelaxed-1)/10:
            return sortrelaxed[:i]
    return sortrelaxed

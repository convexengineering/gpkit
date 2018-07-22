"Implements Model"
import numpy as np
from .costed import CostedConstraintSet
from ..nomials import Monomial
from .prog_factories import _progify_fctry, _solve_fctry
from .gp import GeometricProgram
from .sgp import SequentialGeometricProgram
from ..small_scripts import mag
from ..tools.autosweep import autosweep_1d
from ..exceptions import InvalidGPConstraint
from .. import NamedVariables
from ..tools.docstring import expected_unbounded
from .set import add_meq_bounds


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
                cs = self.setup(*args, **kwargs)  # pylint: disable=no-member
                if (isinstance(cs, tuple) and len(cs) == 2
                        and isinstance(cs[1], dict)):
                    constraints, substitutions = cs  # TODO: remove
                else:
                    constraints = cs
                from .. import NAMEDVARS, MODELS, MODELNUMS
                setup_vars = NAMEDVARS[tuple(MODELS), tuple(MODELNUMS)]
                self.name, self.num = MODELS[-1], MODELNUMS[-1]
                self.naming = (tuple(MODELS), tuple(MODELNUMS))
            cost = self.cost  # TODO: remove
        elif args and not substitutions:
            # backwards compatibility: substitutions as third arg
            substitutions, = args

        cost = cost or Monomial(1)
        constraints = constraints or []
        if setup_vars:
            # add all the vars created in .setup to the Model's varkeys
            # even if they aren't used in any constraints
            self.unique_varkeys = frozenset(v.key for v in setup_vars)
        CostedConstraintSet.__init__(self, cost, constraints, substitutions)
        if hasattr(self, "setup") and self.__class__.__doc__:
            if (("Unbounded" in self.__class__.__doc__ or
                 "Bounded by" in self.__class__.__doc__) and
                    "SKIP VERIFICATION" not in self.__class__.__doc__):
                self.verify_docstring()

    gp = _progify_fctry(GeometricProgram)
    sp = _progify_fctry(SequentialGeometricProgram)
    solve = _solve_fctry(_progify_fctry(GeometricProgram, "solve"))
    localsolve = _solve_fctry(_progify_fctry(SequentialGeometricProgram,
                                             "localsolve"))

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

    def as_gpconstr(self, x0):
        "Returns approximating constraint, keeping name and num"
        cs = CostedConstraintSet.as_gpconstr(self, x0)
        cs.name, cs.num = self.name, self.num
        return cs

    def subconstr_str(self, excluded=None):
        "The collapsed appearance of a ConstraintBase"
        return "%s_%s" % (self.name, self.num) if self.name else None

    def subconstr_latex(self, excluded=None):
        "The collapsed appearance of a ConstraintBase"
        return "%s_{%s}" % (self.name, self.num) if self.name else None

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
        """Attempts to diagnose infeasible models.

        If a model debugs but errors in a process_result call, debug again
        with `process_results=False`
        """
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
            constsrelaxed = ConstantsRelaxed(bounded)
            feas = Model(constsrelaxed.relaxvars.prod()**30 * self.cost,
                         constsrelaxed)
            # NOTE: It hasn't yet been seen but might be possible that
            #       the self.cost component above could cause infeasibility
        else:
            feas = Model(self.cost, bounded)

        try:
            try:
                sol = feas.solve(**solveargs)
            except InvalidGPConstraint:
                sol = feas.localsolve(**solveargs)
            sol["boundedness"] = bounded.check_boundaries(sol)
            if self.substitutions:
                relaxed = get_relaxed([sol(r) for r in constsrelaxed.relaxvars],
                                      constsrelaxed.origvars,
                                      min_return=0 if sol["boundedness"] else 1)
                if verbosity and relaxed:
                    if sol["boundedness"]:
                        print("and these constants relaxed:")
                    else:
                        print("\nSolves with these constants relaxed:")
                    for (_, orig) in relaxed:
                        print("  %s: relaxed from %-.4g to %-.4g"
                              % (orig, mag(constsrelaxed.constants[orig.key]),
                                 mag(sol(orig))))
                    print
            if verbosity:
                print(">> Success!")
        except (ValueError, RuntimeWarning):
            if verbosity:
                print(">> Failure.")
                print("> Trying with relaxed constraints:")

            try:
                constrsrelaxed = ConstraintsRelaxed(self)
                feas = Model(constrsrelaxed.relaxvars.prod()**30 * self.cost,
                             constrsrelaxed)
                try:
                    sol = feas.solve(**solveargs)
                except InvalidGPConstraint:
                    sol = feas.localsolve(**solveargs)
                relaxed = get_relaxed(sol(constrsrelaxed.relaxvars),
                                      range(len(feas[0][0])))
                if verbosity and relaxed:
                    print("\nSolves with these constraints relaxed:")
                    for relaxval, i in relaxed:
                        constraint = feas[0][0][i][0]
                        # substitutions of the final relax value
                        conleft = constraint.left.sub(
                            {constrsrelaxed.relaxvars[i]: relaxval})
                        conright = constraint.right.sub(
                            {constrsrelaxed.relaxvars[i]: relaxval})
                        origconstraint = constrsrelaxed.origconstrs[i]
                        relax_percent = "%i%%" % (0.5+(relaxval-1)*100)
                        print(" %3i: %5s relaxed, from %s %s %s \n"
                              "                     to %s %s %s "
                              % (i, relax_percent, origconstraint.left,
                                 origconstraint.oper, origconstraint.right,
                                 conleft, constraint.oper, conright))
                if verbosity:
                    print("\n>> Success!")
            except (ValueError, RuntimeWarning):
                if verbosity:
                    print(">> Failure")
        if verbosity:
            print
        return sol


def get_relaxed(relaxvals, mapped_list, min_return=1):
    "Determines which relaxvars are considered 'relaxed'"
    sortrelaxed = sorted(zip(relaxvals, mapped_list), key=lambda x: x[0],
                         reverse=True)
    # arbitrarily 1.01 is the min that counts as "relaxed"
    mostrelaxed = max(sortrelaxed[0][0], 1.01)
    for i, (val, _) in enumerate(sortrelaxed):
        if i >= min_return and val <= 1.01 and (val-1) <= (mostrelaxed-1)/10:
            return sortrelaxed[:i]
    return sortrelaxed

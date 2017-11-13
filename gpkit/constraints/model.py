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
                    constraints, substitutions = cs
                else:
                    constraints = cs
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
        if setup_vars:
            # add all the vars created in .setup to the Model's varkeys
            # even if they aren't used in any constraints
            self.unique_varkeys = frozenset(v.key for v in setup_vars)
        CostedConstraintSet.__init__(self, cost, constraints, substitutions)
        self.missingbounds = {}
        if hasattr(self, "setup"):
            bounded = set(self.bounded)
            self.bounded.difference_update(self.bounded)
            for key in self.varkeys:
                if key in self.substitutions:
                    for bound in ("upper", "lower"):
                        bounded.add((key, bound))
            still_alive = True
            while still_alive:
                still_alive = False  # if no changes are made, the loop exits
                for bound, conditions in self.meq_bounded.items():
                    if bound in bounded:
                        del self.meq_bounded[bound]
                        continue
                    self.meq_bounded[bound] = set(conditions)
                    for condition in conditions:
                        if condition.issubset(bounded):
                            del self.meq_bounded[bound]
                            bounded.add(bound)
                            still_alive = True
                            break
            for bound in self.meq_bounded:
                boundstr = (", but would gain it from any of these"
                            " sets of bounds: ")
                for condition in list(self.meq_bounded[bound]):
                    self.meq_bounded[bound].remove(condition)
                    newcond = set(condition)
                    newcond.difference_update(bounded)
                    if newcond and not any(c.issubset(newcond)
                                           for c in self.meq_bounded[bound]):
                        self.meq_bounded[bound].add(frozenset(newcond))
                boundstr += " or ".join(str(list(condition))
                                        for condition in self.meq_bounded[bound])
                self.missingbounds[bound] = boundstr
            if len(bounded)+len(self.missingbounds) != 2*len(self.varkeys):
                for key in self.varkeys:
                    for bound in ("upper", "lower"):
                        if (key, bound) not in bounded:
                            if (key, bound) not in self.missingbounds:
                                self.missingbounds[(key, bound)] = ""
            err = False
            errmessage = ""
            expected_but_not_seen = set()
            seen_but_not_expected = set(self.missingbounds)
            for direction in ["upper", "lower"]:
                flag = direction[0].upper()+direction[1:]+" Unbounded\n"
                count = self.__class__.__doc__.count(flag)
                if count == 0:
                    continue
                elif count > 1:
                    raise ValueError("multiple instances of %s" % flag)
                idx = self.__class__.__doc__.index(flag) + len(flag)
                idx2 = self.__class__.__doc__[idx:].index("\n")
                idx3 = self.__class__.__doc__[idx:][idx2+1:].index("\n")
                varstrs = self.__class__.__doc__[idx:][idx2+1:][:idx3].strip()
                if varstrs:
                    for var in varstrs.split(", "):
                        bound = (getattr(self, var).key, direction)
                        # TODO: catch err if var not found
                        if bound in self.missingbounds:
                            seen_but_not_expected.remove(bound)
                        else:
                            expected_but_not_seen.add(bound)
            print "sbne", seen_but_not_expected
            print "ebns", expected_but_not_seen
            if expected_but_not_seen or seen_but_not_expected:
                boundstrs = "\n".join("  %s has no %s bound%s" % (v, b, x)
                                      for (v, b), x in self.missingbounds.items())
                docstring = """
The docstring for %s should include:

    Upper Unbounded
    ---------------
    %s

    Lower Unbounded
    ---------------
    %s
""" % (self.__class__.__name__,
       ", ".join(k.str_without("models") for (k, b) in self.missingbounds if b == "upper"),
       ", ".join(k.str_without("models") for (k, b) in self.missingbounds if b == "lower"))
                raise ValueError("named Model is not fully bounded:\n"
                                 + boundstrs + "\n\n" + docstring)

    gp = _progify_fctry(GeometricProgram)
    sp = _progify_fctry(SequentialGeometricProgram)
    solve = _solve_fctry(_progify_fctry(GeometricProgram, "solve"))
    localsolve = _solve_fctry(_progify_fctry(SequentialGeometricProgram,
                                             "localsolve"))

    def zero_lower_unbounded_variables(self):
        "Recursively substitutes 0 for variables that lack a lower bound"
        zeros = True
        while zeros:
            # pylint: disable=no-member
            gp = self.gp(allow_missingbounds=True)
            zeros = {var: 0 for (var, bound) in gp.missingbounds
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
                if relaxed:
                    if sol["boundedness"]:
                        print("and these constants relaxed:")
                    else:
                        print("\nSolves with these constants relaxed:")
                    for (_, orig) in relaxed:
                        print("  %s: relaxed from %-.4g to %-.4g"
                              % (orig, mag(constsrelaxed.constants[orig.key]),
                                 mag(sol(orig))))
                    print
            print(">> Success!")
        except (ValueError, RuntimeWarning):
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
                                      range(len(feas[0][0][0])))
                if relaxed:
                    print("\nSolves with these constraints relaxed:")
                    for relaxval, i in relaxed:
                        constraint = feas[0][0][0][i]
                        relax_percent = "%i%%" % (0.5+(relaxval-1)*100)
                        print(" %3i: %5s relaxed, from %s <= 1\n"
                              "                       to %s <= %.4g"
                              % (i, relax_percent, constraint.right,
                                 constraint.right, relaxval))
                print("\n>> Success!")
            except (ValueError, RuntimeWarning):
                print(">> Failure")
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

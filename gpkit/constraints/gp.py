"""Implement the GeometricProgram class"""
import sys
from time import time
from collections import defaultdict
import numpy as np
from ..nomials import NomialData
from ..small_classes import CootMatrix, SolverLog, Numbers, FixedScalar
from ..keydict import KeyDict
from ..small_scripts import mag
from ..solution_array import SolutionArray
from .costed import CostedConstraintSet
from ..exceptions import (InvalidPosynomial, Infeasible, UnknownInfeasible,
                          PrimalInfeasible, DualInfeasible, UnboundedGP)


DEFAULT_SOLVER_KWARGS = {"cvxopt": {"kktsolver": "ldl"}}
SOLUTION_TOL = {"cvxopt": 1e-3, "mosek_cli": 1e-4, "mosek_conif": 1e-3}


def _get_solver(solver, kwargs):
    """Get the solverfn and solvername associated with solver"""
    if solver is None:
        from .. import settings
        try:
            solver = settings["default_solver"]
        except KeyError:
            raise ValueError("No default solver was set during build, so"
                             " solvers must be manually specified.")
    if solver == "cvxopt":
        from ..solvers.cvxopt import optimize
    elif solver == "mosek_cli":
        from ..solvers.mosek_cli import optimize_generator
        optimize = optimize_generator(**kwargs)
    elif solver == "mosek_conif":
        from ..solvers.mosek_conif import optimize
    elif hasattr(solver, "__call__"):
        solver, optimize = solver.__name__, solver
    else:
        raise ValueError("Unknown solver '%s'." % solver)
    return solver, optimize


class GeometricProgram(CostedConstraintSet, NomialData):
    # pylint: disable=too-many-instance-attributes
    """Standard mathematical representation of a GP.

    Attributes with side effects
    ----------------------------
    `solver_out` and `solve_log` are set during a solve
    `result` is set at the end of a solve if solution status is optimal

    Examples
    --------
    >>> gp = gpkit.geometric_program.GeometricProgram(
                        # minimize
                        x,
                        [   # subject to
                            1/x  # <= 1, implicitly
                        ], {})
    >>> gp.solve()
    """
    def __init__(self, cost, constraints, substitutions,
                 *, allow_missingbounds=False):
        # pylint:disable=super-init-not-called
        # initialize attributes modified by internal methods
        self._result = None
        self.v_ss = None
        self.nu_by_posy = None
        self.solve_log = None
        self.solver_out = None
        self.__bare_init__(cost, constraints, substitutions)
        for key, sub in self.substitutions.items():
            if isinstance(sub, FixedScalar):
                sub = sub.value
                if hasattr(sub, "units"):
                    sub = sub.to(key.units or "dimensionless").magnitude
                self.substitutions[key] = sub
            if not isinstance(sub, (Numbers, np.ndarray)):
                raise ValueError("substitution {%s: %s} has invalid value type"
                                 " %s." % (key, sub, type(sub)))
        try:
            self.posynomials = [cost.sub(self.substitutions)]
        except InvalidPosynomial:
            raise InvalidPosynomial("cost must be a Posynomial")
        self.posynomials.extend(self.as_posyslt1(self.substitutions))
        self.hmaps = [p.hmap for p in self.posynomials]
        ## Generate various maps into the posy- and monomials
        # k [j]: number of monomials (rows of A) present in each constraint
        self.k = [len(hm) for hm in self.hmaps]
        p_idxs = []  # p_idxs [i]: posynomial index of each monomial
        self.m_idxs = []  # m_idxs [i]: monomial indices of each posynomial
        for i, p_len in enumerate(self.k):
            self.m_idxs.append(list(range(len(p_idxs), len(p_idxs) + p_len)))
            p_idxs += [i]*p_len
        self.p_idxs = np.array(p_idxs)
        # m_idxs: first exp-index of each monomial equality
        self.meq_idxs = {sum(self.k[:i]) for i, p in enumerate(self.posynomials)
                         if getattr(p, "from_meq", False)}
        self.gen()  # A [i, v]: sparse matrix of powers in each monomial
        self.missingbounds = self.check_bounds(allow_missingbounds)

    def check_bounds(self, allow_missingbounds=True):
        "Checks if any variables are unbounded, through equality constraints."
        missingbounds = {}
        for var, locs in self.varlocs.items():
            upperbound, lowerbound = False, False
            for i in locs:
                if i not in self.meq_idxs:
                    if self.exps[i][var] > 0:  # pylint:disable=simplifiable-if-statement
                        upperbound = True
                    else:
                        lowerbound = True
                if upperbound and lowerbound:
                    break
            if not upperbound:
                missingbounds[(var, "upper")] = ""
            if not lowerbound:
                missingbounds[(var, "lower")] = ""

        conditional_missingbounds = gen_mono_eq_bounds(self.exps, self.meq_idxs)
        check_conditional_bounds(missingbounds, conditional_missingbounds)
        if not missingbounds:
            return {}
        if allow_missingbounds:
            return missingbounds

        raise UnboundedGP("    \n".join("%s has no %s bound%s" % (v, b, x)
                                        for (v, b), x in missingbounds.items()))

    def gen(self):
        "Generates nomial and solve data (A, p_idxs) from posynomials"
        self._hashvalue = self._varlocs = self._varkeys = None
        self._exps, self._cs = [], []
        for hmap in self.hmaps:
            self._exps.extend(hmap.keys())
            self._cs.extend(hmap.values())
        self.vks = self.varlocs

        row, col, data = [], [], []
        for j, var in enumerate(self.varlocs):
            row.extend(self.varlocs[var])
            col.extend([j]*len(self.varlocs[var]))
            data.extend(self.exps[i][var] for i in self.varlocs[var])
        for i, exp in enumerate(self.exps):
            if not exp:  # space the matrix out for trailing constant terms
                row.append(i)
                col.append(0)
                data.append(0)
        self.A = CootMatrix(row, col, data)

    # pylint: disable=too-many-statements, too-many-locals
    def solve(self, solver=None, *, verbosity=1,
              process_result=True, gen_result=True, **kwargs):
        """Solves a GeometricProgram and returns the solution.

        Arguments
        ---------
        solver : str or function (optional)
            By default uses a solver found during installation.
            If "mosek_conif", "mosek_cli", or "cvxopt", uses that solver.
            If a function, passes that function cs, A, p_idxs, and k.
        verbosity : int (default 1)
            If greater than 0, prints solver name and solve time.
        **kwargs :
            Passed to solver constructor and solver function.


        Returns
        -------
        result : SolutionArray
        """
        solvername, solverfn = _get_solver(solver, kwargs)
        solver_kwargs = DEFAULT_SOLVER_KWARGS.get(solvername, {})
        solver_kwargs.update(kwargs)
        solver_out = {}

        if verbosity > 0:
            print("Using solver '%s'" % solvername)
            print("Solving for %i variables." % len(self.varlocs))
        try:
            starttime = time()
            infeasibility, original_stdout = None, sys.stdout
            sys.stdout = SolverLog(original_stdout, verbosity=verbosity-1)
            solver_out = solverfn(c=self.cs, A=self.A, p_idxs=self.p_idxs,
                                  k=self.k, **solver_kwargs)
        except Infeasible as e:
            infeasibility = e
        except Exception as e:
            raise UnknownInfeasible("Something unexpected went wrong.") from e
        finally:
            self.solve_log = "\n".join(sys.stdout)
            sys.stdout = original_stdout
            self.solver_out = solver_out
            solver_out["solver"] = solvername
            solver_out["soltime"] = time() - starttime
            if verbosity > 0:
                print("Solving took %.3g seconds." % solver_out["soltime"])

        if infeasibility:
            if isinstance(infeasibility, PrimalInfeasible):
                msg = ("The model had no feasible points; "
                       "you may wish to relax some constraints or constants.")
            elif isinstance(infeasibility, DualInfeasible):
                msg = ("The model ran to an infinitely low cost;"
                       " bounding the right variables would prevent this.")
            elif isinstance(infeasibility, UnknownInfeasible):
                msg = "The solver failed for an unknown reason."
            msg += (" Running `.debug()` may pinpoint the trouble. You can"
                    " also try another solver, or increase the verbosity.")
            raise infeasibility.__class__(msg) from infeasibility

        if gen_result:  # NOTE: SIDE EFFECTS
            self._result = self.generate_result(solver_out,
                                                verbosity=verbosity-1,
                                                process_result=process_result)
            return self.result

        solver_out["generate_result"] = \
            lambda: self.generate_result(solver_out, dual_check=False)

        return solver_out

    @property
    def result(self):
        "Creates and caches a result from the raw solver_out"
        if not self._result:
            self._result = self.generate_result(self.solver_out)
        return self._result

    def generate_result(self, solver_out, *, verbosity=0,
                        process_result=True, dual_check=True):
        "Generates a full SolutionArray and checks it."
        if verbosity > 0:
            soltime = solver_out["soltime"]
            tic = time()

        # result packing #
        result = self._compile_result(solver_out)  # NOTE: SIDE EFFECTS
        if verbosity > 0:
            print("Result packing took %.2g%% of solve time." %
                  ((time() - tic) / soltime * 100))
            tic = time()

        # solution checking #
        try:
            tol = SOLUTION_TOL.get(solver_out["solver"], 1e-5)
            self.check_solution(result["cost"], solver_out['primal'],
                                solver_out["nu"], solver_out["la"], tol)
        except Infeasible as chkerror:
            chkwarn = str(chkerror)
            if dual_check or ("Dual" not in chkwarn and "nu" not in chkwarn):
                print("Solution check warning: %s" % chkwarn)
        if verbosity > 0:
            print("Solution checking took %.2g%% of solve time." %
                  ((time() - tic) / soltime * 100))
            tic = time()

        # result processing #
        if process_result:
            self.process_result(result)
        if verbosity > 0:
            print("Processing results took %.2g%% of solve time." %
                  ((time() - tic) / soltime * 100))

        return result

    def _generate_nula(self, solver_out):
        if "nu" in solver_out:
            # solver gave us monomial sensitivities, generate posynomial ones
            nu = np.ravel(solver_out["nu"])
            self.nu_by_posy = [nu[mi] for mi in self.m_idxs]
            la = np.array([sum(nup) for nup in self.nu_by_posy])
        elif "la" in solver_out:
            # solver gave us posynomial sensitivities, generate monomial ones
            la = np.ravel(solver_out["la"])
            if len(la) == len(self.hmaps) - 1:
                # assume the solver dropped the cost's sensitivity (always 1.0)
                la = np.hstack(([1.0], la))
            Ax = np.ravel(self.A.dot(solver_out['primal']))
            z = Ax + np.log(self.cs)
            m_iss = [self.p_idxs == i for i in range(len(la))]
            self.nu_by_posy = [la[p_i]*np.exp(z[m_is])/sum(np.exp(z[m_is]))
                               for p_i, m_is in enumerate(m_iss)]
            nu = np.hstack(self.nu_by_posy)
        else:
            raise RuntimeWarning("The dual solution was not returned.")
        solver_out["nu"], solver_out["la"] = nu, la

    def _compile_result(self, solver_out):
        """Creates a result dict (as returned by solve() from solver output

        This internal method is called from within the solve() method, unless
        solver_out["status"] is not "optimal", in which case a RuntimeWarning
        is raised prior to this method being called. In that case, users
        may use this method to attempt to create a results dict from the
        output of the failed solve.

        Arguments
        ---------
        solver_out: dict
            dict in format returned by solverfn within GeometricProgram.solve

        Returns
        -------
        result: dict
            dict in format returned by GeometricProgram.solve()
        """
        self._generate_nula(solver_out)
        primal = solver_out["primal"]
        nu, la = solver_out["nu"], solver_out["la"]
        # confirm lengths before calling zip
        if not self.varlocs and len(primal) == 1 and primal[0] == 0:
            primal = []  # an empty result, as returned by MOSEK
        assert len(self.varlocs) == len(primal)
        result = {"freevariables": KeyDict(zip(self.varlocs, np.exp(primal)))}
        # get cost #
        if "objective" in solver_out:
            result["cost"] = float(solver_out["objective"])
        else:
            # use self.posynomials[0] because the cost may have had constants
            freev = result["freevariables"]
            cost = self.posynomials[0].sub(freev)
            if cost.varkeys:
                raise ValueError("cost contains unsolved variables %s"
                                 % cost.varkeys.keys())
            result["cost"] = mag(cost.c)
        # get variables #
        result["constants"] = KeyDict(self.substitutions)
        result["variables"] = KeyDict(result["freevariables"])
        result["variables"].update(result["constants"])
        # get sensitivities #
        result["sensitivities"] = {"nu": nu, "la": la}
        self.v_ss = self.sens_from_dual(la[1:].tolist(), self.nu_by_posy[1:],
                                        result)
        # add cost's sensitivity in (nu could be self.nu_by_posy[0])
        cost_senss = {var: sum([self.cost.exps[i][var]*nu[i] for i in locs])
                      for (var, locs) in self.cost.varlocs.items()}
        var_senss = self.v_ss.copy()
        for key, value in cost_senss.items():
            var_senss[key] = value + var_senss.get(key, 0)
        # carry linked sensitivities over to their constants
        for v in list(v for v in var_senss if v.gradients):
            dlogcost_dlogv = var_senss.pop(v)
            val = result["constants"][v]
            for c, dv_dc in v.gradients.items():
                if val != 0:
                    dlogv_dlogc = dv_dc * result["constants"][c]/val
                # make nans / infs explicitly to avoid warnings
                elif dlogcost_dlogv == 0:
                    dlogv_dlogc = np.nan
                else:
                    dlogv_dlogc = np.inf * dv_dc*result["constants"][c]
                accum = var_senss.get(c, 0)
                var_senss[c] = dlogcost_dlogv*dlogv_dlogc + accum
                if v in cost_senss:
                    if c in self.cost.varkeys:
                        dlogcost_dlogv = cost_senss.pop(v)
                        accum = cost_senss.get(c, 0)
                        cost_senss[c] = dlogcost_dlogv*dlogv_dlogc + accum

        result["sensitivities"]["cost"] = cost_senss
        result["sensitivities"]["variables"] = KeyDict(var_senss)
        result["sensitivities"]["constants"] = KeyDict(
            {k: v for k, v in var_senss.items() if k in result["constants"]})
        result["soltime"] = solver_out["soltime"]
        return SolutionArray(result)

    def check_solution(self, cost, primal, nu, la, tol, abstol=1e-20):
        """Run checks to mathematically confirm solution solves this GP

        Arguments
        ---------
        cost:   float
            cost returned by solver
        primal: list
            primal solution returned by solver
        nu:     numpy.ndarray
            monomial lagrange multiplier
        la:     numpy.ndarray
            posynomial lagrange multiplier

        Raises
        ------
        Infeasible if any problems are found
        """
        def _almost_equal(num1, num2):
            "local almost equal test"
            return (num1 == num2 or abs((num1 - num2) / (num1 + num2)) < tol
                    or abs(num1 - num2) < abstol)
        A = self.A.tocsr()

        # check primal sol #
        primal_exp_vals = self.cs * np.exp(A.dot(primal))   # c*e^Ax
        if not _almost_equal(primal_exp_vals[self.m_idxs[0]].sum(), cost):
            raise Infeasible("Primal solution computed cost did not match"
                             " solver-returned cost: %s vs %s." %
                             (primal_exp_vals[self.m_idxs[0]].sum(), cost))
        for mi in self.m_idxs[1:]:
            if primal_exp_vals[mi].sum() > 1 + tol:
                raise Infeasible("Primal solution violates constraint: %s is "
                                 "greater than 1." % primal_exp_vals[mi].sum())

        # check dual sol #
        # note: follows dual formulation in section 3.1 of
        # http://web.mit.edu/~whoburg/www/papers/hoburg_phd_thesis.pdf
        if not _almost_equal(self.nu_by_posy[0].sum(), 1.):
            raise Infeasible("Dual variables associated with objective sum"
                             " to %s, not 1." % self.nu_by_posy[0].sum())
        if any(nu < 0):
            minnu = min(nu)
            if minnu > -tol/1000.:  # HACK, see issue 528
                print("Allowing negative dual variables up to %s." % minnu)
            else:
                raise Infeasible("Dual solution has negative entries as"
                                 " large as %s." % minnu)
        if any(np.abs(A.T.dot(nu)) > tol):
            raise Infeasible("Sum of nu^T * A did not vanish.")
        b = np.log(self.cs)
        dual_cost = sum(
            self.nu_by_posy[i].dot(
                b[mi] - np.log(self.nu_by_posy[i]/la[i]))
            for i, mi in enumerate(self.m_idxs) if la[i])
        if not _almost_equal(np.exp(dual_cost), cost):
            raise Infeasible("Dual cost %s does not match primal cost %s"
                             % (np.exp(dual_cost), cost))


def gen_mono_eq_bounds(exps, meq_idxs):  # pylint: disable=too-many-locals
    "Generate conditional monomial equality bounds"
    meq_bounds = defaultdict(set)
    for i in meq_idxs:
        if i % 2:  # skip the second index of a meq
            continue
        p_upper, p_lower, n_upper, n_lower = set(), set(), set(), set()
        for key, x in exps[i].items():
            if x > 0:
                p_upper.add((key, "upper"))
                p_lower.add((key, "lower"))
            else:
                n_upper.add((key, "upper"))
                n_lower.add((key, "lower"))
        # (consider x*y/z == 1)
        # for a var (e.g. x) to be upper bounded by this monomial equality,
        #   - vars of the same sign/side (y) must be lower bounded
        #   - AND vars of the opposite sign/side (z) must be upper bounded
        p_ub = frozenset(n_upper).union(p_lower)
        p_lb = frozenset(n_lower).union(p_upper)
        n_ub = frozenset(p_upper).union(n_lower)
        n_lb = frozenset(p_lower).union(n_upper)
        for keys, ub, lb in ((p_upper, p_ub, p_lb), (n_upper, n_ub, n_lb)):
            for key, _ in keys:
                meq_bounds[(key, "upper")].add(ub.difference([(key, "lower")]))
                meq_bounds[(key, "lower")].add(lb.difference([(key, "upper")]))
    return meq_bounds


def check_conditional_bounds(missingbounds, meq_bounds):
    "Bounds variables with monomial equalities"
    still_alive = True
    while still_alive:
        still_alive = False  # if no changes are made, the loop exits
        for bound in list(meq_bounds):
            if bound not in missingbounds:
                del meq_bounds[bound]
                continue
            conditions = meq_bounds[bound]
            for condition in conditions:
                if not any(bound in missingbounds for bound in condition):
                    del meq_bounds[bound]
                    del missingbounds[bound]
                    still_alive = True
                    break
    for (var, bound) in meq_bounds:
        boundstr = (", but would gain it from any of these sets of bounds: ")
        for condition in list(meq_bounds[(var, bound)]):
            meq_bounds[(var, bound)].remove(condition)
            newcond = condition.intersection(missingbounds)
            if newcond and not any(c.issubset(newcond)
                                   for c in meq_bounds[(var, bound)]):
                meq_bounds[(var, bound)].add(newcond)
        boundstr += " or ".join(str(list(condition))
                                for condition in meq_bounds[(var, bound)])
        missingbounds[(var, bound)] = boundstr

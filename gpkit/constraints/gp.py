"""Implement the GeometricProgram class"""
import sys
from time import time
from collections import defaultdict
import numpy as np
from ..nomials import NomialData
from ..small_classes import CootMatrix, SolverLog, Numbers
from ..keydict import KeyDict, KeySet
from ..small_scripts import mag
from ..solution_array import SolutionArray
from .costed import CostedConstraintSet


DEFAULT_SOLVER_KWARGS = {"cvxopt": {"kktsolver": "ldl"}}


class GeometricProgram(CostedConstraintSet, NomialData):
    # pylint: disable=too-many-instance-attributes
    """Standard mathematical representation of a GP.

    Arguments
    ---------
    cost : Constraint
        Posynomial to minimize when solving
    constraints : list of Posynomials
        Constraints to maintain when solving (implicitly Posynomials <= 1)
        GeometricProgram does not accept equality constraints (e.g. x == 1);
         instead use two inequality constraints (e.g. x <= 1, 1/x <= 1)
    verbosity : int (optional)
        If verbosity is greater than zero, warns about missing bounds
        on creation.

    Attributes with side effects
    ----------------------------
    `solver_out` and `solver_log` are set during a solve
    `result` is set at the end of a solve if solution status is optimal

    Examples
    --------
    >>> gp = gpkit.geometric_program.GeometricProgram(
                        # minimize
                        x,
                        [   # subject to
                            1/x  # <= 1, implicitly
                        ])
    >>> gp.solve()
    """
    def __init__(self, cost, constraints, substitutions,
                 allow_missingbounds=False):
        # pylint:disable=super-init-not-called
        # initialize attributes modified by internal methods
        self.result = None
        self.v_ss = None
        self.nu_by_posy = None
        self.solver_log = None
        self.solver_out = None
        # GPs get varkeys from NomialData._reset, in .gen()
        self.__bare_init__(cost, constraints, substitutions, varkeys=False)
        for key, sub in self.substitutions.items():
            if hasattr(sub, "exp") and not sub.exp:
                sub = sub.value
                if hasattr(sub, "units"):
                    sub = sub.to(key.units or "dimensionless").magnitude
                self.substitutions[key] = sub
            # only allow Numbers and ndarrays
            if not isinstance(sub, (Numbers, np.ndarray)):
                raise ValueError("substitution {%s: %s} with value type %s is"
                                 " not allowed in .substitutions; such"
                                 " substitutions must be done by using"
                                 " .subinplace()." % (key, sub, type(sub)))
        self.posynomials = [cost.sub(self.substitutions)]
        self.posynomials.extend(self.as_posyslt1(self.substitutions))
        self.hmaps = [p.hmap for p in self.posynomials]
        ## Generate various maps into the posy- and monomials
        # k [j]: number of monomials (columns of F) present in each constraint
        self.k = [len(hm) for hm in self.hmaps]
        p_idxs = []  # p_idxs [i]: posynomial index of each monomial
        self.m_idxs = []  # m_idxs [i]: monomial indices of each posynomial
        for i, p_len in enumerate(self.k):
            self.m_idxs.append(list(range(len(p_idxs), len(p_idxs) + p_len)))
            p_idxs += [i]*p_len
        self.p_idxs = np.array(p_idxs)
        # m_idxs: first exp-index of each monomial equality
        self.meq_idxs = [sum(self.k[:i]) for i, p in enumerate(self.posynomials)
                         if getattr(p, "from_meq", False)]
        self.gen()  # A [i, v]: sparse matrix of powers in each monomial
        if any(c <= 0 for c in self._cs):
            raise ValueError("GeometricPrograms cannot contain Signomials.")
        if self.missingbounds and not allow_missingbounds:
            boundstrs = "\n".join("  %s has no %s bound%s" % (v, b, x)
                                  for (v, b), x in self.missingbounds.items())
            raise ValueError("Geometric Program is not fully bounded:\n"
                             + boundstrs)

    def gen(self):
        "Generates nomial and solve data (A, p_idxs) from posynomials"
        self._reset()  # method from NomialData
        self._exps, self._cs = [], []
        for hmap in self.hmaps:
            self._exps.extend(hmap.keys())
            self._cs.extend(hmap.values())
        self.A, self.missingbounds = genA(self.exps, self.varlocs,
                                          self.meq_idxs)

    @property
    def varkeys(self):
        "The GP's varkeys, created when necessary."
        if self._varkeys is None:
            self._varkeys = KeySet(self.varlocs)
        return self._varkeys

    # pylint: disable=too-many-statements, too-many-locals
    def solve(self, solver=None, verbosity=1, warn_on_check=False,
              process_result=True, **kwargs):
        """Solves a GeometricProgram and returns the solution.

        Arguments
        ---------
        solver : str or function (optional)
            By default uses one of the solvers found during installation.
            If set to "mosek", "mosek_cli", or "cvxopt", uses that solver.
            If set to a function, passes that function cs, A, p_idxs, and k.
        verbosity : int (optional)
            If greater than 0, prints solver name and solve time.
        **kwargs :
            Passed to solver constructor and solver function.


        Returns
        -------
        result : dict
            A dictionary containing the translated solver result; keys below.

            cost : float
                The value of the objective at the solution.
            variables : dict
                The value of each variable at the solution.
            sensitivities : dict
                monomials : array of floats
                    Each monomial's dual variable value at the solution.
                posynomials : array of floats
                    Each posynomials's dual variable value at the solution.
        """
        def _get_solver(solver):
            """Get the solverfn and solvername associated with solver"""
            if solver is None:
                from .. import settings
                solver = settings.get("default_solver", None)
                if not solver:
                    raise ValueError(
                        "No solver was given; perhaps gpkit was not properly"
                        " installed, or found no solvers during the"
                        " installation process.")

            if solver == "cvxopt":
                from .._cvxopt import cvxoptimize
                solverfn = cvxoptimize
            elif solver == "mosek_cli":
                from .._mosek import cli_expopt
                solverfn = cli_expopt.imize_fn(**kwargs)
            elif solver == "mosek":
                from .._mosek import expopt
                solverfn = expopt.imize
            elif hasattr(solver, "__call__"):
                solverfn = solver
                solver = solver.__name__
            else:
                raise ValueError("Unknown solver '%s'." % solver)
            return solverfn, solver

        solverfn, solvername = _get_solver(solver)

        starttime = time()
        if verbosity > 0:
            print("Using solver '%s'" % solvername)
            print("Solving for %i variables." % len(self.varlocs))

        solver_kwargs = DEFAULT_SOLVER_KWARGS.get(solvername, {})
        solver_kwargs.update(kwargs)

        # NOTE: SIDE EFFECTS AS WE LOG SOLVER'S STDOUT AND OUTPUT
        original_stdout = sys.stdout
        self.solver_log = SolverLog(verbosity-1, original_stdout)
        try:
            sys.stdout = self.solver_log   # CAPTURED
            solver_out = solverfn(c=self.cs, A=self.A, p_idxs=self.p_idxs,
                                  k=self.k, **solver_kwargs)
            self.solver_out = solver_out
        finally:
            sys.stdout = original_stdout
        # STDOUT HAS BEEN RETURNED. ENDING SIDE EFFECTS.
        self.solver_log = "\n".join(self.solver_log)

        soltime = time() - starttime
        if verbosity > 0:
            print("Solving took %.3g seconds." % (soltime,))
            tic = time()

        # allow mosek's NEAR_DUAL_FEAS solution status, because our check
        # will catch anything that's not actually near enough.
        # TODO: implement this in the mosek / mosek_cli interfaces, not here.
        solver_status = str(solver_out.get("status", None))
        if solver_status.lower() not in ["optimal", "near_dual_feas"]:
            raise RuntimeWarning(
                "final status of solver '%s' was '%s', not 'optimal'.\n\n"
                "The solver's result is stored in model.program.solver_out. "
                "A result dict can be generated via "
                "program._compile_result(program.solver_out)." %
                (solvername, solver_status))

        if solver_status.lower() == "near_dual_feas":
            print(RuntimeWarning(
                "final status of solver '%s' was '%s', not 'optimal'.\n\n"
                % (solvername, solver_status)))

        self.result = self._compile_result(solver_out)  # NOTE: SIDE EFFECTS
        if verbosity > 1:
            print("result packing took %.2g%% of solve time" %
                  ((time() - tic) / soltime * 100))
            tic = time()
        try:
            self.check_solution(self.result["cost"], solver_out['primal'],
                                nu=solver_out["nu"], la=solver_out["la"])
        except RuntimeWarning as e:
            if warn_on_check:
                print("Solution check warning: %s" % e)
            else:
                raise e
        if verbosity > 1:
            print("solution checking took %.2g%% of solve time" %
                  ((time() - tic) / soltime * 100))

        if process_result:
            self.process_result(self.result)
        self.result["soltime"] = soltime
        return self.result

    def _generate_nula(self, solver_out):
        solver_out["primal"] = np.ravel(solver_out['primal'])

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

        # get sensitivities #
        result["constants"] = KeyDict(self.substitutions)
        result["variables"] = KeyDict(result["freevariables"])
        result["variables"].update(result["constants"])
        result["sensitivities"] = {"nu": nu, "la": la}
        self.v_ss = self.sens_from_dual(la[1:].tolist(), self.nu_by_posy[1:],
                                        result)
        # add cost's sensitivity in (nu could be self.nu_by_posy[0])
        cost_senss = {var: sum([self.cost.exps[i][var]*nu[i] for i in locs])
                      for (var, locs) in self.cost.varlocs.items()}
        # not using HashVector addition because we want to preseve zeros
        var_senss = self.v_ss.copy()
        for key, value in cost_senss.items():
            var_senss[key] = value + var_senss.get(key, 0)
        # carry linked sensitivities over to their constants
        for v in var_senss.keys():
            if v.gradients:
                dlogcost_dlogv = var_senss.pop(v)
                val = result["constants"][v]
                for c, dv_dc in v.gradients.items():
                    dlogv_dlogc = dv_dc * result["constants"][c]/val
                    accum = var_senss.get(c, 0)
                    var_senss[c] = dlogcost_dlogv*dlogv_dlogc + accum
                    if v in cost_senss:
                        if c in self.cost.varkeys:
                            dlogcost_dlogv = cost_senss.pop(v)
                            accum = cost_senss.get(c, 0)
                            cost_senss[c] = dlogcost_dlogv*dlogv_dlogc + accum

        result["sensitivities"]["cost"] = cost_senss
        result["sensitivities"]["variables"] = KeyDict(var_senss)
        const_senss = {k: v for k, v in var_senss.items()
                       if k in result["constants"]}
        result["sensitivities"]["constants"] = KeyDict(const_senss)
        return SolutionArray(result)

    # TODO: set tol by solver? or otherwise return it to 1e-5 for mosek
    def check_solution(self, cost, primal, nu, la, tol=1e-3, abstol=1e-20):
        """Run a series of checks to mathematically confirm sol solves this GP

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
        RuntimeWarning, if any problems are found
        """
        def _almost_equal(num1, num2):
            "local almost equal test"
            return (num1 == num2 or abs((num1 - num2) / (num1 + num2)) < tol
                    or abs(num1 - num2) < abstol)
        A = self.A.tocsr()
        # check primal sol
        primal_exp_vals = self.cs * np.exp(A.dot(primal))   # c*e^Ax
        if not _almost_equal(primal_exp_vals[self.m_idxs[0]].sum(), cost):
            raise RuntimeWarning("Primal solution computed cost did not match"
                                 " solver-returned cost: %s vs %s" %
                                 (primal_exp_vals[self.m_idxs[0]].sum(), cost))
        for mi in self.m_idxs[1:]:
            if primal_exp_vals[mi].sum() > 1 + tol:
                raise RuntimeWarning("Primal solution violates constraint:"
                                     " %s is greater than 1." %
                                     primal_exp_vals[mi].sum())
        # check dual sol
        # note: follows dual formulation in section 3.1 of
        # http://web.mit.edu/~whoburg/www/papers/hoburg_phd_thesis.pdf
        if not _almost_equal(self.nu_by_posy[0].sum(), 1.):
            raise RuntimeWarning("Dual variables associated with objective sum"
                                 " to %s, not 1" % self.nu_by_posy[0].sum())
        if any(nu < 0):
            if all(nu > -tol/1000.):  # HACK, see issue 528
                print("Allowing negative dual variable(s) as small as"
                      " %s." % min(nu))
            else:
                raise RuntimeWarning("Dual solution has negative entries as"
                                     " small as %s." % min(nu))
        ATnu = A.T.dot(nu)
        if any(np.abs(ATnu) > tol):
            raise RuntimeWarning("sum of nu^T * A did not vanish")
        b = np.log(self.cs)
        dual_cost = sum(self.nu_by_posy[i].dot(b[mi] -
                                               np.log(self.nu_by_posy[i]/la[i])
                                               if la[i] else 0)
                        for i, mi in enumerate(self.m_idxs))
        if not _almost_equal(np.exp(dual_cost), cost):
            raise RuntimeWarning("Dual cost %s does not match primal"
                                 " cost %s" % (np.exp(dual_cost), cost))


def genA(exps, varlocs, meq_idxs):  # pylint: disable=invalid-name
    """Generates A matrix from exps and varidxs

    Arguments
    ---------
        exps : list of Hashvectors
            Exponents for each monomial in a GP
        varidxs : dict
            Locations of each variable in exps

    Returns
    -------
        A : sparse Cootmatrix
            Exponents of the various free variables for each monomial: rows
            of A are monomials, columns of A are variables.
        missingbounds : dict
            Keys: variables that lack bounds. Values: which bounds are missed.
    """

    meq_bounds = defaultdict(set)
    for i in meq_idxs:
        exp = exps[i]
        for v1 in exp:
            s_exp = np.sign(exp[v1])
            ubs = frozenset((v2, "upper" if np.sign(e) != s_exp else "lower")
                            for v2, e in exp.items() if v1 != v2)
            lbs = frozenset((v2, "lower" if np.sign(e) != s_exp else "upper")
                            for v2, e in exp.items() if v1 != v2)
            meq_bounds[(v1, "upper")].add(ubs)
            meq_bounds[(v1, "lower")].add(lbs)

    missingbounds = {}
    A = CootMatrix([], [], [])
    for j, var in enumerate(varlocs):
        varbounds = None
        for i in varlocs[var]:
            exp = exps[i][var]
            A.append(i, j, exp)
            if i not in meq_idxs:
                if varbounds == "both":
                    pass
                elif varbounds is None:
                    varbounds = np.sign(exp)
                elif np.sign(exp) != varbounds:
                    varbounds = "both"
        if varbounds != "both":
            if varbounds == 1 or varbounds is None:
                missingbounds[(var, "lower")] = ""
            if varbounds == -1 or varbounds is None:
                missingbounds[(var, "upper")] = ""

    check_mono_eq_bounds(missingbounds, meq_bounds)

    # space the matrix out for trailing constant terms
    for i, exp in enumerate(exps):
        if not exp:
            A.append(i, 0, 0)

    return A, missingbounds


def check_mono_eq_bounds(missingbounds, meq_bounds):
    "Bounds variables with monomial equalities"
    still_alive = True
    while still_alive:
        still_alive = False  # if no changes are made, the loop exits
        for bound, conditions in meq_bounds.items():
            if bound not in missingbounds:
                del meq_bounds[bound]
                continue
            for condition in conditions:
                if not any(bound in missingbounds for bound in condition):
                    del meq_bounds[bound]
                    del missingbounds[bound]
                    still_alive = True
                    break
    for (var, bound) in meq_bounds:
        boundstr = (", but would gain it from any of these"
                    " sets of bounds: ")
        for condition in list(meq_bounds[(var, bound)]):
            meq_bounds[(var, bound)].remove(condition)
            newcond = condition.intersection(missingbounds)
            if newcond and not any(c.issubset(newcond)
                                   for c in meq_bounds[(var, bound)]):
                meq_bounds[(var, bound)].add(newcond)
        boundstr += " or ".join(str(list(condition))
                                for condition in meq_bounds[(var, bound)])
        missingbounds[(var, bound)] = boundstr

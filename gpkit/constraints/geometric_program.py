"""Implement the GeometricProgram class"""
import sys
from time import time
import numpy as np
from ..nomials import NomialData
from ..small_classes import CootMatrix, SolverLog
from ..keydict import FastKeyDict
from ..small_scripts import mag
from ..solution_array import SolutionArray
from .costed import CostedConstraintSet
from .set import ConstraintSet


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
    def __init__(self, cost, constraints, substitutions=None, verbosity=1):
        # pylint:disable=super-init-not-called
        # initialize attributes modified by internal methods
        self.result = None
        self.solver_log = None
        self.solver_out = None

        # barebones ConstraintSet init
        self.cost = cost
        if not isinstance(constraints, ConstraintSet):
            constraints = ConstraintSet(constraints)
        list.__init__(self, [constraints])  # pylint:disable=non-parent-init-called
        self.substitutions = substitutions if substitutions else {}

        # sideways NomialData init to create self.exps, self.cs, etc
        self.posynomials = [cost.sub(self.substitutions)]
        self.posynomials.extend(self.as_posyslt1(self.substitutions))
        NomialData.init_from_nomials(self, self.posynomials)
        if self.any_nonpositive_cs:
            raise ValueError("GeometricPrograms cannot contain Signomials.")

        ## Generate various maps into the posy- and monomials
        # k [j]: number of monomials (columns of F) present in each constraint
        self.k = [len(p.cs) for p in self.posynomials]
        # p_idxs [i]: posynomial index of each monomial
        p_idxs = []
        # m_idxs [i]: monomial indices of each posynomial
        self.m_idxs = []
        for i, p_len in enumerate(self.k):
            self.m_idxs.append(list(range(len(p_idxs), len(p_idxs) + p_len)))
            p_idxs += [i]*p_len
        self.p_idxs = np.array(p_idxs)
        # A [i, v]: sparse matrix of variable's powers in each monomial
        self.A, self.missingbounds = genA(self.exps, self.varlocs)
        if verbosity > 0:
            for var, bound in sorted(self.missingbounds.items()):
                print("%s has no %s bound" % (var, bound))

    # pylint: disable=too-many-statements
    def solve(self, solver=None, verbosity=1, *args, **kwargs):
        """Solves a GeometricProgram and returns the solution.

        Arguments
        ---------
        solver : str or function (optional)
            By default uses one of the solvers found during installation.
            If set to "mosek", "mosek_cli", or "cvxopt", uses that solver.
            If set to a function, passes that function cs, A, p_idxs, and k.
        verbosity : int (optional)
            If greater than 0, prints solver name and solve time.
        *args, **kwargs :
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
                solverfn = cli_expopt.imize_fn(*args, **kwargs)
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

        if verbosity > 0:
            print("Using solver '%s'" % solvername)
            print("Solving for %i variables." % len(self.varlocs))
            tic = time()

        default_kwargs = DEFAULT_SOLVER_KWARGS.get(solvername, {})
        for k in default_kwargs:
            kwargs.setdefault(k, default_kwargs[k])

        # NOTE: SIDE EFFECTS AS WE LOG SOLVER'S STDOUT AND OUTPUT
        original_stdout = sys.stdout
        self.solver_log = SolverLog(verbosity-1, original_stdout)
        try:
            sys.stdout = self.solver_log   # CAPTURED
            solver_out = solverfn(c=self.cs, A=self.A, p_idxs=self.p_idxs,
                                  k=self.k, *args, **kwargs)
            self.solver_out = solver_out
        finally:
            sys.stdout = original_stdout
         # STDOUT HAS BEEN RETURNED. ENDING SIDE EFFECTS.

        if verbosity > 0:
            soltime = time() - tic
            print("Solving took %.3g seconds." % (soltime,))
            tic = time()

        if solver_out.get("status", "").lower() != "optimal":
            raise RuntimeWarning(
                "final status of solver '%s' was '%s', not 'optimal'.\n\n"
                "The solver's result is stored in model.program.solver_out. "
                "A result dict can be generated via "
                "program._compile_result(program.solver_out)." %
                (solvername, solver_out.get("status", None)))

        self._generate_nula(solver_out)
        self.result = self._compile_result(solver_out)  # NOTE: SIDE EFFECTS
        if verbosity > 1:
            print("result packing took %.2g%% of solve time" %
                  ((time() - tic) / soltime * 100))
            tic = time()

        self.check_solution(self.result["cost"], solver_out['primal'],
                            nu=solver_out["nu"], la=solver_out["la"])
        if verbosity > 1:
            print("solution checking took %.2g%% of solve time" %
                  ((time() - tic) / soltime * 100))

        self.process_result(self.result)
        return self.result

    def _generate_nula(self, solver_out):
        solver_out["primal"] = np.ravel(solver_out['primal'])

        if "nu" in solver_out:
            # solver gave us monomial sensitivities, generate posynomial ones
            nu = np.ravel(solver_out["nu"])
            la = np.array([sum(nu[self.p_idxs == i])
                           for i in range(len(self.posynomials))])
        elif "la" in solver_out:
            # solver gave us posynomial sensitivities, generate monomial ones
            la = np.ravel(solver_out["la"])
            if len(la) == len(self.posynomials) - 1:
                # assume the solver dropped the cost's sensitivity (always 1.0)
                la = np.hstack(([1.0], la))
            Ax = np.ravel(self.A.dot(solver_out['primal']))
            z = Ax + np.log(self.cs)
            m_iss = [self.p_idxs == i for i in range(len(la))]
            nu = np.hstack([la[p_i]*np.exp(z[m_is])/sum(np.exp(z[m_is]))
                            for p_i, m_is in enumerate(m_iss)])
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
        primal = solver_out["primal"]
        nu, la = solver_out["nu"], solver_out["la"]
        # confirm lengths before calling zip
        assert len(self.varlocs) == len(primal)
        result = {"freevariables": FastKeyDict(zip(self.varlocs,
                                                   np.exp(primal)))}

        ## Get cost
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

        ## Get sensitivities
        result["sensitivities"] = {"nu": nu, "la": la}
        var_senss = self.sens_from_dual(la[1:].tolist(),
                                        [[nu[i] for i in m_idx]
                                         for m_idx in self.m_idxs[1:]])
        # add cost's sensitivity in
        var_senss += {var: sum([self.cost.exps[i][var]*nu[i] for i in locs])
                      for (var, locs) in self.cost.varlocs.items()
                      if (var in self.cost.varlocs
                          and var not in self.posynomials[0].varlocs)}

        result["sensitivities"]["constants"] = FastKeyDict(var_senss)
        result["constants"] = FastKeyDict(self.substitutions)
        result["variables"] = FastKeyDict(result["freevariables"])
        result["variables"].update(result["constants"])
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
        nu0 = nu[self.m_idxs[0]]
        if not _almost_equal(nu0.sum(), 1.):
            raise RuntimeWarning("Dual variables associated with objective"
                                 " sum to %s, not 1" % nu0.sum())
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
        dual_cost = sum(nu[mi].dot(b[mi]) -
                        (nu[mi].dot(np.log(nu[mi]/la[i])) if la[i] else 0)
                        for i, mi in enumerate(self.m_idxs))
        if not _almost_equal(np.exp(dual_cost), cost):
            raise RuntimeWarning("Dual cost %s does not match primal"
                                 " cost %s" % (np.exp(dual_cost), cost))


def genA(exps, varlocs):
    # pylint: disable=invalid-name
    """Generates A matrix from exps and varlocs

    Arguments
    ---------
        exps : list of Hashvectors
            Exponents for each monomial in a GP
        varlocs : dict
            Locations of each variable in exps

    Returns
    -------
        A : sparse Cootmatrix
            Exponents of the various free variables for each monomial: rows
            of A are monomials, columns of A are variables.
        missingbounds : dict
            Keys: variables that lack bounds. Values: which bounds are missed.
    """

    missingbounds = {}
    A = CootMatrix([], [], [])
    for j, var in enumerate(varlocs):
        varsign = "both" if "value" in var.descr else None
        for i in varlocs[var]:
            exp = exps[i][var]
            A.append(i, j, exp)
            if varsign is "both":
                pass
            elif varsign is None:
                varsign = np.sign(exp)
            elif np.sign(exp) != varsign:
                varsign = "both"

        if varsign != "both":
            if varsign == 1:
                bound = "lower"
            elif varsign == -1:
                bound = "upper"
            else:
                # just being safe
                raise RuntimeWarning("Unexpected varsign %s" % varsign)
            missingbounds[var] = bound

    # add constant terms
    for i, exp in enumerate(exps):
        if not exp:
            A.append(i, 0, 0)

    return A, missingbounds

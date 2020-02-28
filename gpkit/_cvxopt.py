"Implements the GPkit interface to CVXOPT"
import numpy as np
from cvxopt import spmatrix, matrix, log
from cvxopt.solvers import gp
from gpkit.exceptions import UnknownInfeasible, DualInfeasible


def cvxoptimize(c, A, k, *args, **kwargs):
    """Interface to the CVXOPT solver

        Definitions
        -----------
        "[a,b] array of floats" indicates array-like data with shape [a,b]
        n is the number of monomials in the gp
        m is the number of variables in the gp
        p is the number of posynomial constraints in the gp

        Arguments
        ---------
        c : floats array of shape n
            Coefficients of each monomial
        A : floats array of shape (n, m)
            Exponents of the various free variables for each monomial.
        k : ints array of shape p+1
            k[0] is the number of monomials (rows of A) present in the objective
            k[1:] is the number of monomials present in each constraint

        Returns
        -------
        dict
            Contains the following keys
                "success": bool
                "objective_sol" float
                    Optimal value of the objective
                "primal_sol": floats array of size m
                    Optimal value of free variables. Note: not in logspace.
                "dual_sol": floats array of size p
                    Optimal value of the dual variables, in logspace.
    """
    g = log(matrix(c))
    F = spmatrix(A.data, A.row, A.col, tc="d")
    try:
        solution = gp(k, F, g, *args, **kwargs)
    except ValueError as e:
        raise DualInfeasible() from e
    if solution["status"] == "unknown":
        raise UnknownInfeasible()
    if solution["status"] != "optimal":
        raise UnknownInfeasible("solution status: " + solution["status"])

    return dict(status=solution["status"],
                objective=np.exp(solution["primal objective"]),
                primal=np.ravel(solution["x"]),
                la=solution["znl"])

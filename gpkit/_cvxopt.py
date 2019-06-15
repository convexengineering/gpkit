"Implements the GPkit interface to CVXOPT"
import numpy as np
from cvxopt import spmatrix, matrix, log
from cvxopt.solvers import gp


def cvxoptimize(c, A, k, *args, **kwargs):
    """Interface to the CVXOPT solver

        Definitions
        -----------
        "[a,b] array of floats" indicates array-like data with shape [a,b]
        n is the number of monomials in the gp
        m is the number of variables in the gp
        p is the number of posynomials in the gp

        Arguments
        ---------
        c : floats array of shape n
            Coefficients of each monomial
        A : floats array of shape (m,n)
            Exponents of the various free variables for each monomial.
        k : ints array of shape n
            number of monomials (columns of F) present in each constraint

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
    F = spmatrix(A.data, A.row, A.col, tc='d')
    solution = gp(k, F, g, *args, **kwargs)
    return dict(status=solution['status'],
                primal=np.ravel(solution['x']),
                la=solution['znl'])

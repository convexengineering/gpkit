"Implements the GPkit interface to CVXOPT"
import numpy as np
from cvxopt import spmatrix, matrix
from cvxopt.solvers import gp
from gpkit.exceptions import UnknownInfeasible, DualInfeasible


# pylint: disable=too-many-locals
def optimize(*, c, A, k, meq_idxs, use_leqs=True, **kwargs):
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
    log_c = np.log(np.array(c))
    lse_mons, lin_mons, leq_mons = [], [], []
    lse_posys, lin_posys, leq_posys = [], [], []
    for i, n_monomials in enumerate(k):
        start = sum(k[:i])
        mons = range(start, start+k[i])
        if use_leqs and start in meq_idxs.all:
            if start in meq_idxs.first_half:
                leq_posys.append(i)
                leq_mons.extend(mons)
        elif i != 0 and n_monomials == 1:
            lin_mons.extend(mons)
            lin_posys.append(i)
        else:
            lse_mons.extend(mons)
            lse_posys.append(i)
    A = A.tocsr()
    maxcol = A.shape[1]-1
    if leq_mons:
        A_leq = A[leq_mons, :].tocoo()
        log_c_leq = log_c[leq_mons]
        kwargs["A"] = spmatrix([float(r) for r in A_leq.data]+[0],
                               [int(r) for r in A_leq.row]+[0],
                               [int(r) for r in A_leq.col]+[maxcol], tc="d")
        kwargs["b"] = matrix(-log_c_leq)
    if lin_mons:
        A_lin = A[lin_mons, :].tocoo()
        log_c_lin = log_c[lin_mons]
        kwargs["G"] = spmatrix([float(r) for r in A_lin.data]+[0],
                               [int(r) for r in A_lin.row]+[0],
                               [int(r) for r in A_lin.col]+[maxcol], tc="d")
        kwargs["h"] = matrix(-log_c_lin)
    k_lse = [k[i] for i in lse_posys]
    A_lse = A[lse_mons, :].tocoo()
    log_c_lse = log_c[lse_mons]
    F = spmatrix([float(r) for r in A_lse.data]+[0],
                 [int(r) for r in A_lse.row]+[0],
                 [int(r) for r in A_lse.col]+[maxcol], tc="d")
    g = matrix(log_c_lse)
    try:
        solution = gp(k_lse, F, g, **kwargs)
    except ValueError as e:
        raise DualInfeasible() from e
    if solution["status"] != "optimal":
        raise UnknownInfeasible("solution status " + repr(solution["status"]))
    la = np.zeros(len(k))
    la[lin_posys] = list(solution["zl"])
    la[lse_posys] = [1.] + list(solution["znl"])
    for leq_posy, yi in zip(leq_posys, solution["y"]):
        if yi >= 0:
            la[leq_posy] = yi
        else:  # flip it around to the other "inequality"
            la[leq_posy+1] = -yi
    return dict(status=solution["status"],
                objective=np.exp(solution["primal objective"]),
                primal=np.ravel(solution["x"]),
                la=la)

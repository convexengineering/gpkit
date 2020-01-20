"Implements the GPkit interface to MOSEK (version >= 9) python-based Optimizer API"

import numpy as np


def mskoptimize(c, A, k, p_idxs, *args, **kwargs):
    """
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
    A : gpkit.small_classes.CootMatrix, of shape (n, m)
        Exponents of the various free variables for each monomial.
    k : ints array of shape p+1
        k[0] is the number of monomials (rows of A) present in the objective
        k[1:] is the number of monomials (rows of A) present in each constraint
    p_idxs : list of bool arrays, each of shape m
        p_idxs[i] selects rows of A and entries of c of the i-th posynomial
        fi(x) = c[p_idxs[i]] @ exp(A[p_idxs[i],:] @ x). The 0-th posynomial
        gives the objective function, and the remaining posynomials should
        be constrained to be <= 1.

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
    import mosek
    #
    #   Prepare some initial problem data
    #
    #       Say that the optimization variable is y = [x;t], where
    #       x is of length m, and t is of length 3*n. Entry
    #       t[3*ell] is the epigraph variable for the ell-th monomial
    #       t[3*ell + 1] should be == 1.
    #       t[3*ell + 2] should be == A[ell, :] @ x.
    #
    c = np.array(c)
    n, m = A.shape
    p = len(p_idxs) - 1
    n_msk = m + 3*n
    #
    #   Create MOSEK task. add variables and conic constraints.
    #
    env = mosek.Env()
    task = env.Task(0, 0)
    task.appendvars(n_msk)
    task.putvarboundlist(np.arange(n_msk, dtype=int),
                         [mosek.boundkey.fr] * n_msk,
                         np.zeros(n_msk),
                         np.zeros(n_msk))
    for i in range(n):
        idx = m + 3*i
        task.appendcone(mosek.conetype.pexp, 0.0, np.arange(idx, idx + 3))
    #
    #   Calls to MOSEK's "putaijlist"
    #
    task.appendcons(2*n + p)
    # Linear equations: t[3*ell + 1] == 1.
    rows = [i for i in range(n)]
    cols = (m + 3*np.arange(n) + 1).tolist()
    vals = [1.0] * n
    task.putaijlist(rows, cols, vals)
    # Linear equations: A[ell,:] @ x - t[3*ell + 2] == 0
    cur_con_idx = n
    rows = [cur_con_idx + r for r in A.rows]
    task.putaijlist(rows, A.cols, A.vals)
    rows = [cur_con_idx + i for i in range(n)]
    cols = (m + 3*np.arange(n) + 2).tolist()
    vals = [-1.0] * n
    task.putaijlist(rows, cols, vals)
    # Linear inequalities: c[sels] @ t[3 * sels] <= 1, for sels = np.nonzero(p_idxs[i])
    cur_con_idx = 2*n
    rows, cols, vals = [], [], []
    for i in range(p):
        sels = np.nonzero(p_idxs[i + 1])[0]
        rows.extend([cur_con_idx] * sels.size)
        cols.extend(m + 3 * sels)
        vals.extend(c[sels])
        cur_con_idx += 1
    task.putaijlist(rows, cols, vals)
    #
    #   Build the right-hand-sides of the [in]equality constraints
    #
    type_constraint = [mosek.boundkey.fx] * (2*n) + [mosek.boundkey.up] * p
    h = np.concatenate([np.ones(n), np.zeros(n), np.ones(p)])
    task.putconboundlist(np.arange(h.size, dtype=int), type_constraint, h, h)
    #
    #   Set the objective function
    #
    sels = np.nonzero(p_idxs[0])[0]
    cols = (m + 3 * sels).tolist()
    task.putclist(cols, c[sels].tolist())
    task.putobjsense(mosek.objsense.minimize)
    #
    #   Set solver parameters, and call .solve().
    #
    task.optimize()
    #
    #   Recover the solution
    #
    msk_solsta = task.getsolsta(mosek.soltype.itr)
    if msk_solsta == mosek.solsta.optimal:
        str_status = 'optimal'
        x = [0.] * m
        task.getxxslice(mosek.soltype.itr, 0, len(x), x)
        solution = {'status': str_status, 'primal': np.array(x)}
        return solution
    elif msk_solsta == mosek.solsta.prim_infeas_cer:
        str_status = 'infeasible'
        solution = {'status': str_status, 'primal': None}
        return solution
    elif msk_solsta == mosek.solsta.dual_infeas_cer:
        str_status = 'unbounded'
        solution = {'status': str_status, 'primal': None}
        return solution
    else:
        raise RuntimeError('Unexpected solver status.')

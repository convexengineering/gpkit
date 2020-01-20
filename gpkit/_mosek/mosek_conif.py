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
        sel = p_idxs == i selects rows of A and entries of c of the i-th posynomial
        fi(x) = c[sel] @ exp(A[sel,:] @ x). The 0-th posynomial gives the objective
        function, and the remaining posynomials should be constrained to be <= 1.

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
    #       Say that the optimization variable is y = [x;t;z], where
    #       x is of length m, t is of length 3*n, and z is of length p+1.
    #
    #       Triples
    #           t[3*ell]
    #           t[3*ell + 1] should be == 1.
    #           t[3*ell + 2] should be == A[ell, :] @ x + np.log(c[ell]) - z[posynom idx corresponding to ell].
    #       should belong to the exponential cone.
    #
    #       For each i from {0,...,p}, the "t" should also satisfy
    #           sum(t[3*ell] for ell correponsing to posynomial i) <= 1.
    #
    #       The vector "z" should satisfy
    #           z[1:] <= 0
    #
    #
    c = np.array(c)
    n, m = A.shape
    p = len(k) - 1
    n_msk = m + 3*n + p + 1
    #
    #   Create MOSEK task. add variables and conic constraints.
    #
    env = mosek.Env()
    task = env.Task(0, 0)
    task.appendvars(n_msk)
    bound_types = [mosek.boundkey.fr] * (m + 3*n + 1) + [mosek.boundkey.up] * p
    task.putvarboundlist(np.arange(n_msk, dtype=int),
                         bound_types, np.zeros(n_msk), np.zeros(n_msk))
    for i in range(n):
        idx = m + 3*i
        task.appendcone(mosek.conetype.pexp, 0.0, np.arange(idx, idx + 3))
    #
    #   Calls to MOSEK's "putaijlist"
    #
    task.appendcons(2*n + p + 1)
    # Linear equations: t[3*ell + 1] == 1.
    rows = [i for i in range(n)]
    cols = (m + 3*np.arange(n) + 1).tolist()
    vals = [1.0] * n
    task.putaijlist(rows, cols, vals)
    # Linear equations: A[ell,:] @ x  - t[3*ell + 2] - z[posynom idx corresponding to ell] == -np.log(c[ell])
    cur_con_idx = n
    rows = [cur_con_idx + r for r in A.row]
    task.putaijlist(rows, A.col, A.data)
    rows = [cur_con_idx + i for i in range(n)]
    cols = (m + 3*np.arange(n) + 2).tolist()
    vals = [-1.0] * n
    task.putaijlist(rows, cols, vals)
    rows = [cur_con_idx + i for i in range(n)]
    cols = [m + 3*n + p_idxs[i] for i in range(n)]
    vals = [-1.0] * n
    task.putaijlist(rows, cols, vals)
    # Linear inequalities: 1 @ t[3 * sels] <= 1, for sels = np.nonzero(p_idxs[i] == i)
    cur_con_idx = 2*n
    rows, cols, vals = [], [], []
    for i in range(p+1):
        sels = np.nonzero(p_idxs == i)[0]
        rows.extend([cur_con_idx] * sels.size)
        cols.extend(m + 3 * sels)
        vals.extend([1] * sels.size)
        cur_con_idx += 1
    task.putaijlist(rows, cols, vals)
    #
    #   Build the right-hand-sides of the [in]equality constraints
    #
    type_constraint = [mosek.boundkey.fx] * (2*n) + [mosek.boundkey.up] * (p + 1)
    h = np.concatenate([np.ones(n), -np.log(c), np.ones(p + 1)])
    task.putconboundlist(np.arange(h.size, dtype=int), type_constraint, h, h)
    #
    #   Set the objective function
    #
    task.putclist([int(m + 3*n)], [1])
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
        # recover primal variables
        x = [0.] * m
        task.getxxslice(mosek.soltype.itr, 0, m, x)
        x = np.array(x)
        # recover dual variables for log-sum-exp epigraph constraints
        # (skip epigraph of the objective function).
        z_duals = [0.] * p
        task.getsuxslice(mosek.soltype.itr, m + 3*n + 1, n_msk, z_duals)
        z_duals = np.array(z_duals)
        z_duals[z_duals < 0] = 0
        # wrap things up in a dictionary
        solution = {'status': str_status, 'primal': x, 'la': z_duals}
    elif msk_solsta == mosek.solsta.prim_infeas_cer:
        str_status = 'infeasible'
        solution = {'status': str_status, 'primal': None}
    elif msk_solsta == mosek.solsta.dual_infeas_cer:
        str_status = 'unbounded'
        solution = {'status': str_status, 'primal': None}
    else:
        raise RuntimeError('Unexpected solver status.')
    task.__exit__(None, None, None)
    env.__exit__(None, None, None)
    return solution

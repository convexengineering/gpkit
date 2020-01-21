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
    # separate exponential from linear constraints.
    # the objective is always kept in exponential form.
    exp_posys = [0] + [i+1 for i, val in enumerate(k[1:]) if val > 1]
    lin_posys = [i for i in range(len(k)) if i not in exp_posys]
    if len(lin_posys) > 0:
        A = A.tocsr()
        lin_idxs = []
        for i in lin_posys:
            temp = np.nonzero(p_idxs == i)
            temp = temp[0]
            lin_idxs.append(temp)
        lin_idxs = np.concatenate(lin_idxs)
        nonlin_idxs = np.ones(A.shape[0], dtype=bool)
        nonlin_idxs[lin_idxs] = False
        A_exp = A[nonlin_idxs, :].tocoo()
        c_exp = c[nonlin_idxs]
        A_lin = A[lin_idxs, :].tocoo()
        c_lin = c[lin_idxs]
    else:
        c_lin = np.array([])
        A_exp = A
        c_exp = c

    m = A.shape[1]
    k_exp = [k[i] for i in exp_posys]
    n_exp = sum(k_exp)
    p_exp = len(k_exp)
    n_msk = m + 3*n_exp + p_exp
    exp_p_idx = []
    for i, ki in enumerate(k_exp):
        exp_p_idx.extend([i] * ki)
    exp_p_idx = np.array(exp_p_idx)
    #
    #   Create MOSEK task. add variables and conic constraints.
    #
    env = mosek.Env()
    task = env.Task(0, 0)
    task.appendvars(n_msk)
    bound_types = [mosek.boundkey.fr] * (m + 3*n_exp + 1) + [mosek.boundkey.up] * (p_exp - 1)
    task.putvarboundlist(np.arange(n_msk, dtype=int),
                         bound_types, np.zeros(n_msk), np.zeros(n_msk))
    for i in range(n_exp):
        idx = m + 3*i
        task.appendcone(mosek.conetype.pexp, 0.0, np.arange(idx, idx + 3))
    #
    #   Affine constraints related to the exponential cone
    #
    task.appendcons(2*n_exp + p_exp)
    # 1st n_exp: Linear equations: t[3*ell + 1] == 1
    rows = [i for i in range(n_exp)]
    cols = (m + 3*np.arange(n_exp) + 1).tolist()
    vals = [1.0] * n_exp
    task.putaijlist(rows, cols, vals)
    # 2nd n_exp: Linear equations:
    #       A[ell,:] @ x  - t[3*ell + 2] - z[posynom idx corresponding to ell] == -np.log(c[ell])
    cur_con_idx = n_exp
    rows = [cur_con_idx + r for r in A_exp.row]
    task.putaijlist(rows, A_exp.col, A_exp.data)
    rows = [cur_con_idx + i for i in range(n_exp)]
    cols = (m + 3*np.arange(n_exp) + 2).tolist()
    vals = [-1.0] * n_exp
    task.putaijlist(rows, cols, vals)
    rows = [cur_con_idx + i for i in range(n_exp)]
    cols = [m + 3*n_exp + exp_p_idx[i] for i in range(n_exp)]
    vals = [-1.0] * n_exp
    task.putaijlist(rows, cols, vals)
    # last p_exp: Linear inequalities:
    #       1 @ t[3 * sels] <= 1, for sels = np.nonzero(exp_p_idxs[i] == i)
    cur_con_idx = 2*n_exp
    rows, cols, vals = [], [], []
    for i in range(p_exp):
        sels = np.nonzero(exp_p_idx == i)[0]
        rows.extend([cur_con_idx] * sels.size)
        cols.extend(m + 3 * sels)
        vals.extend([1] * sels.size)
        cur_con_idx += 1
    task.putaijlist(rows, cols, vals)
    # Build the right-hand-sides of the [in]equality constraints
    type_constraint = [mosek.boundkey.fx] * (2*n_exp) + [mosek.boundkey.up] * p_exp
    h = np.concatenate([np.ones(n_exp), -np.log(c_exp), np.ones(p_exp)])
    task.putconboundlist(np.arange(h.size, dtype=int), type_constraint, h, h)
    #
    #   Affine constraints, not needing the exponential cone
    #
    cur_con_idx = 2*n_exp + p_exp
    if c_lin.size > 0:
        task.appendcons(c_lin.size)
        rows = [cur_con_idx + r for r in A_lin.row]
        task.putaijlist(rows, A_lin.col, A_lin.data)
        type_constraint = [mosek.boundkey.up] * c_lin.size
        con_indices = np.arange(cur_con_idx, cur_con_idx + c_lin.size, dtype=int)
        h = -np.log(c_lin)
        task.putconboundlist(con_indices, type_constraint, h, h)
        cur_con_idx += c_lin.size
    #
    #   Set the objective function
    #
    task.putclist([int(m + 3*n_exp)], [1])
    task.putobjsense(mosek.objsense.minimize)
    #
    #   Set solver parameters, and call .solve().
    #
    verbose = False
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    if verbose:
        import sys

        def streamprinter(text):
            sys.stdout.write(text)
            sys.stdout.flush()

        print('\n')
        env.set_Stream(mosek.streamtype.log, streamprinter)
        task.set_Stream(mosek.streamtype.log, streamprinter)
        task.putintparam(mosek.iparam.infeas_report_auto, mosek.onoffkey.on)
        task.putintparam(mosek.iparam.log_presolve, 0)

    task.optimize()

    if verbose:
        task.solutionsummary(mosek.streamtype.msg)
    #
    #   Recover the solution
    #
    msk_solsta = task.getsolsta(mosek.soltype.itr)
    if msk_solsta == mosek.solsta.optimal:
        # recover primal variables
        x = [0.] * m
        task.getxxslice(mosek.soltype.itr, 0, m, x)
        x = np.array(x)
        # recover dual variables for log-sum-exp epigraph constraints
        # (skip epigraph of the objective function).
        z_duals = [0.] * (p_exp - 1)
        task.getsuxslice(mosek.soltype.itr, m + 3*n_exp + 1, n_msk, z_duals)
        z_duals = np.array(z_duals)
        z_duals[z_duals < 0] = 0
        # recover dual variables for the remaining user-provided constraints
        if c_lin.size > 0:
            aff_duals = [0.] * c_lin.size
            task.getsucslice(mosek.soltype.itr, 2*n_exp + p_exp, cur_con_idx, aff_duals)
            aff_duals = np.array(aff_duals)
            aff_duals[aff_duals < 0] = 0
            # merge z_duals with aff_duals
            merged_duals = np.zeros(len(k))
            merged_duals[exp_posys[1:]] = z_duals
            merged_duals[lin_posys] = aff_duals
            merged_duals = merged_duals[1:]
        else:
            merged_duals = z_duals
        # wrap things up in a dictionary
        solution = {'status': 'optimal', 'primal': x, 'la': merged_duals}
    elif msk_solsta == mosek.solsta.prim_infeas_cer:
        solution = {'status': 'infeasible', 'primal': None}
    elif msk_solsta == mosek.solsta.dual_infeas_cer:
        solution = {'status': 'unbounded', 'primal': None}
    else:
        solution = {'status': 'unknown', 'primal': None}
    task.__exit__(None, None, None)
    env.__exit__(None, None, None)
    return solution

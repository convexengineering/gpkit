"""Implements the GPkit interface to MOSEK (version >= 9)
   through python-based Optimizer API"""

import numpy as np
import mosek

def mskoptimize(c, A, k, p_idxs, *args, **kwargs):
    # pylint: disable=unused-argument
    # pylint: disable-msg=too-many-locals
    # pylint: disable-msg=too-many-statements
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
    p_idxs : ints array of shape n.
        sel = p_idxs == i selects rows of A
                          and entries of c of the i-th posynomial
        fi(x) = c[sel] @ exp(A[sel,:] @ x).
                The 0-th posynomial gives the objective function, and the
                remaining posynomials should be constrained to be <= 1.

    Returns
    -------
    dict
        Contains the following keys
            "status": string
                "optimal", "infeasible", "unbounded", or "unknown".
            "primal" np.ndarray or None
                The values of the ``m`` primal variables.
            "la": np.ndarray or None
                The dual variables to the ``p`` posynomial constraints, when
                those constraints are represented in log-sum-exp ("LSE") form.
    """
    #
    #   Initial transformations of problem data.
    #
    #       separate monomial constraints (call them "lin"),
    #       from those which require
    #       an LSE representation (call those "lse").
    #
    #       NOTE: the epigraph of the objective function always gets an "lse"
    #       representation, even if the objective is a monomial.
    #
    log_c = np.log(np.array(c))
    lse_posys = [0] + [i+1 for i, val in enumerate(k[1:]) if val > 1]
    lin_posys = [i for i in range(len(k)) if i not in lse_posys]
    if len(lin_posys) > 0:
        A = A.tocsr()
        lin_idxs = np.concatenate([np.nonzero(p_idxs == i)[0]
                                   for i in lin_posys])
        lse_idxs = np.ones(A.shape[0], dtype=bool)
        lse_idxs[lin_idxs] = False
        A_lse = A[lse_idxs, :].tocoo()
        log_c_lse = log_c[lse_idxs]
        A_lin = A[lin_idxs, :].tocoo()
        log_c_lin = log_c[lin_idxs]
    else:
        log_c_lin = np.array([])  # A_lin won't be referenced later,
        A_lse = A                 # so no need to define it.
        log_c_lse = log_c
    k_lse = [k[i] for i in lse_posys]
    n_lse = sum(k_lse)
    p_lse = len(k_lse)
    lse_p_idx = []
    for i, ki in enumerate(k_lse):
        lse_p_idx.extend([i] * ki)
    lse_p_idx = np.array(lse_p_idx)
    #
    #   Create MOSEK task. Add variables, and conic constraints.
    #
    #       Say that MOSEK's optimization variable is a block vector,
    #       [x;t;z], where ...
    #           x is the user-defined primal variable (length m),
    #           t is an aux variable for exponential cones (length 3 * n_lse),
    #           z is an epigraph variable for LSE terms (length p_lse).
    #
    #       The variable z[0] is special,
    #       because it's the epigraph of the objective function
    #       in LSE form. The sign of this variable is not constrained.
    #
    #       The variables z[1:] are epigraph terms for "log",
    #       in constraints that naturally write as
    #       LSE(Ai @ x + log_ci) <= 0. These variables need to be <= 0.
    #
    #       The main constraints on (x, t, z) are described
    #       in next comment block.
    #
    env = mosek.Env()
    task = env.Task(0, 0)
    m = A.shape[1]
    msk_nvars = m + 3 * n_lse + p_lse
    task.appendvars(msk_nvars)
    bound_types = [mosek.boundkey.fr] * (m + 3*n_lse + 1) + \
                  [mosek.boundkey.up] * (p_lse - 1)
    task.putvarboundlist(np.arange(msk_nvars, dtype=int),
                         bound_types, np.zeros(msk_nvars), np.zeros(msk_nvars))
    for i in range(n_lse):
        idx = m + 3*i
        task.appendcone(mosek.conetype.pexp, 0.0, np.arange(idx, idx + 3))
    #
    #   Affine constraints related to the exponential cone
    #
    #       For each i in {0, ..., n_lse - 1}, we need
    #           t[3*i + 1] == 1, and
    #           t[3*i + 2] == A_lse[i, :] @ x + log_c_lse[i] - z[lse_p_idx[i]].
    #       This contributes 2 * n_lse constraints.
    #
    #       For each j from {0, ..., p_lse - 1}, the "t" should also satisfy
    #           sum(t[3*i] for i where i == lse_p_idx[j]) <= 1.
    #       This contributes another p_lse constraints.
    #
    #       The above constraints imply that for ``sel = lse_p_idx == i``,
    #       we have LSE(A_lse[sel, :] @ x + log_c_lse[sel]) <= z[i].
    #
    #       We specify the necessary constraints to MOSEK in three phases.
    #       Over the course of these three phases,
    #       we make a total of five calls to "putaijlist"
    #       and a single call to "putconboundlist".
    #
    task.appendcons(2*n_lse + p_lse)
    # 1st n_lse: Linear equations: t[3*i + 1] == 1
    rows = list(range(n_lse))
    cols = (m + 3*np.arange(n_lse) + 1).tolist()
    vals = [1.0] * n_lse
    task.putaijlist(rows, cols, vals)
    cur_con_idx = n_lse
    # 2nd n_lse: Linear equations between (x,t,z).
    rows = [cur_con_idx + r for r in A_lse.row]
    task.putaijlist(rows, A_lse.col, A_lse.data)  # coefficients on "x"
    rows = [cur_con_idx + i for i in range(n_lse)]
    cols = (m + 3*np.arange(n_lse) + 2).tolist()
    vals = [-1.0] * n_lse
    task.putaijlist(rows, cols, vals)  # coefficients on "t"
    rows = [cur_con_idx + i for i in range(n_lse)]
    cols = [m + 3*n_lse + lse_p_idx[i] for i in range(n_lse)]
    vals = [-1.0] * n_lse
    task.putaijlist(rows, cols, vals)  # coefficients on "z".
    cur_con_idx = 2 * n_lse
    # last p_lse: Linear inequalities on certain sums of "t".
    rows, cols, vals = [], [], []
    for i in range(p_lse):
        sels = np.nonzero(lse_p_idx == i)[0]
        rows.extend([cur_con_idx] * sels.size)
        cols.extend(m + 3 * sels)
        vals.extend([1] * sels.size)
        cur_con_idx += 1
    task.putaijlist(rows, cols, vals)
    cur_con_idx = 2 * n_lse + p_lse
    # Build the right-hand-sides of the [in]equality constraints
    type_constraint = [mosek.boundkey.fx] * (2*n_lse) + \
                      [mosek.boundkey.up] * p_lse
    h = np.concatenate([np.ones(n_lse), -log_c_lse, np.ones(p_lse)])
    task.putconboundlist(np.arange(h.size, dtype=int), type_constraint, h, h)
    #
    #   Affine constraints, not needing the exponential cone
    #
    #       Require A_lin @ x <= -log_c_lin.
    #
    if log_c_lin.size > 0:
        task.appendcons(log_c_lin.size)
        rows = [cur_con_idx + r for r in A_lin.row]
        task.putaijlist(rows, A_lin.col, A_lin.data)
        type_constraint = [mosek.boundkey.up] * log_c_lin.size
        con_indices = np.arange(cur_con_idx, cur_con_idx + log_c_lin.size,
                                dtype=int)
        h = -log_c_lin
        task.putconboundlist(con_indices, type_constraint, h, h)
        cur_con_idx += log_c_lin.size
    #
    #   Set the objective function
    #
    task.putclist([int(m + 3*n_lse)], [1])
    task.putobjsense(mosek.objsense.minimize)
    #
    #   Set solver parameters, and call .solve().
    #
    verbose = False
    if kwargs.get('verbose'):
        verbose = kwargs['verbose']
    if verbose:
        # pylint: disable=import-outside-toplevel
        import sys
        # pylint: enable=import-outside-toplevel

        def streamprinter(text):
            """ Stream printer for output from mosek. """
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
        z_duals = [0.] * (p_lse - 1)
        task.getsuxslice(mosek.soltype.itr, m + 3*n_lse + 1,
                         msk_nvars, z_duals)
        z_duals = np.array(z_duals)
        z_duals[z_duals < 0] = 0
        # recover dual variables for the remaining user-provided constraints
        if log_c_lin.size > 0:
            aff_duals = [0.] * log_c_lin.size
            task.getsucslice(mosek.soltype.itr, 2*n_lse + p_lse, cur_con_idx,
                             aff_duals)
            aff_duals = np.array(aff_duals)
            aff_duals[aff_duals < 0] = 0
            # merge z_duals with aff_duals
            merged_duals = np.zeros(len(k))
            merged_duals[lse_posys[1:]] = z_duals
            merged_duals[lin_posys] = aff_duals
            merged_duals = merged_duals[1:]
        else:
            merged_duals = z_duals
        # wrap things up in a dictionary
        solution = {'status': 'optimal', 'primal': x, 'la': merged_duals}
    elif msk_solsta == mosek.solsta.prim_infeas_cer:
        solution = {'status': 'infeasible', 'primal': None, 'la': None}
    elif msk_solsta == mosek.solsta.dual_infeas_cer:
        solution = {'status': 'unbounded', 'primal': None, 'la': None}
    else:
        solution = {'status': 'unknown', 'primal': None, 'la': None}
    task.__exit__(None, None, None)
    env.__exit__(None, None, None)
    return solution

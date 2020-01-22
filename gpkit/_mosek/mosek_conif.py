"Implements the GPkit interface to MOSEK (v >= 9) python-based Optimizer API"
from __future__ import print_function
import sys
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
    p_idxs : ints array of shape n.
        sel = p_idxs == i selects rows of A and entries of c of the i-th
        posynomial fi(x) = c[sel] @ exp(A[sel,:] @ x). The 0-th posynomial gives
        the objective function, and the remaining posynomials should be
        constrained to be <= 1.

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
    import mosek
    if not hasattr(mosek.conetype, 'pexp'):
        msg = """
        mosek_conif requires MOSEK version >= 9. The imported version of MOSEK
        does not have attribute ``mosek.conetype.pexp``, which was introduced
        in MOSEK 9. 
        """
        raise RuntimeError(msg)
    #
    #   Initial transformations of problem data.
    #
    #       separate monomial constraints (call them "lin"), from those which
    #       require an LSE representation (call those "lse").
    #
    #       NOTE: the epigraph of the objective function always gets an "lse"
    #       representation, even if the objective is a monomial.
    #
    log_c = np.log(np.array(c))
    lse_posys = [0] + [i+1 for i, val in enumerate(k[1:]) if val > 1]
    lin_posys = [i for i in range(len(k)) if i not in lse_posys]
    if lin_posys:
        A = A.tocsr()
        lin_idxs = np.concatenate(
            [np.nonzero(p_idxs == i)[0] for i in lin_posys])
        lse_idxs = np.ones(A.shape[0], dtype=bool)
        lse_idxs[lin_idxs] = False
        A_lse = A[lse_idxs, :].tocoo()
        log_c_lse = log_c[lse_idxs]
        A_lin = A[lin_idxs, :].tocoo()
        log_c_lin = log_c[lin_idxs]
    else:
        log_c_lin = np.array([])
        # A_lin won't be referenced later, so no need to define it.
        A_lse = A
        log_c_lse = log_c
    k_lse = [k[i] for i in lse_posys]
    n_lse = sum(k_lse)
    p_lse = len(k_lse)
    lse_p_idx = []
    for i, ki in enumerate(k_lse):
        lse_p_idx.extend([i] * ki)
    lse_p_idx = np.array(lse_p_idx)
    #
    #   Create MOSEK task. Add variables, bound constraints, and conic
    #   constraints.
    #
    #       Say that MOSEK's optimization variable is a block vector, [x;t;z],
    #       where ...
    #           x is the user-defined primal variable (length m),
    #           t is an auxiliary variable for exponential cones (length
    #           3 * n_lse), and
    #           z is an epigraph variable for LSE terms (length p_lse).
    #
    #       The variable z[0] is special, because it's the epigraph of the
    #       objective function in LSE form. The sign of this variable is
    #       not constrained.
    #
    #       The variables z[1:] are epigraph terms for "log", in constraints
    #       that naturally write as LSE(Ai @ x + log_ci) <= 0. These variables
    #       need to be <= 0.
    #
    #       The main constraints on (x, t, z) are described in next
    #       comment block.
    #
    env = mosek.Env()
    task = env.Task(0, 0)
    m = A.shape[1]
    msk_nvars = m + 3 * n_lse + p_lse
    task.appendvars(msk_nvars)
    # "x" is free
    task.putvarboundlist(np.arange(m), [mosek.boundkey.fr] * m,
                         np.zeros(m), np.zeros(m))
    # t[3 * i + i] == 1, other components are free.
    bound_types = [mosek.boundkey.fr, mosek.boundkey.fx, mosek.boundkey.fr]
    task.putvarboundlist(np.arange(m, m + 3*n_lse), bound_types * n_lse,
                         np.ones(3*n_lse), np.ones(3*n_lse))
    # z[0] is free; z[1:] <= 0.
    bound_types = [mosek.boundkey.fr] + [mosek.boundkey.up] * (p_lse - 1)
    task.putvarboundlist(np.arange(m + 3*n_lse, msk_nvars), bound_types,
                         np.zeros(p_lse), np.zeros(p_lse))
    # t[3*i], t[3*i + 1], t[3*i + 2] belongs to the exponential cone
    task.appendconesseq([mosek.conetype.pexp] * n_lse, [0.0] * n_lse,
                        [3] * n_lse, m)
    #
    #   Exponential cone affine constraints (other than t[3*i + 1] == 1).
    #
    #       For each i in {0, ..., n_lse - 1}, we need
    #           t[3*i + 2] == A_lse[i, :] @ x + log_c_lse[i] - z[lse_p_idx[i]].
    #
    #       For each j from {0, ..., p_lse - 1}, the "t" should also satisfy
    #           sum(t[3*i] for i where i == lse_p_idx[j]) <= 1.
    #
    #       When combined with bound constraints ``t[3*i + 1] == 1``, the
    #       above constraints imply
    #           LSE(A_lse[sel, :] @ x + log_c_lse[sel]) <= z[i]
    #       for ``sel = lse_p_idx == i``.
    #
    task.appendcons(n_lse + p_lse)
    # Linear equations between (x,t,z).
    #   start with coefficients on "x"
    rows = [r for r in A_lse.row]
    cols = [c for c in A_lse.col]
    vals = [v for v in A_lse.data]
    #   add coefficients on "t"
    rows += list(range(n_lse))
    cols += (m + 3*np.arange(n_lse) + 2).tolist()
    vals += [-1.0] * n_lse
    #   add coefficients on "z"
    rows += list(range(n_lse))
    cols += [m + 3*n_lse + lse_p_idx[i] for i in range(n_lse)]
    vals += [-1.0] * n_lse
    task.putaijlist(rows, cols, vals)
    cur_con_idx = n_lse
    # Linear inequalities on certain sums of "t".
    rows, cols, vals = [], [], []
    for i in range(p_lse):
        sels = np.nonzero(lse_p_idx == i)[0]
        rows.extend([cur_con_idx] * sels.size)
        cols.extend(m + 3 * sels)
        vals.extend([1] * sels.size)
        cur_con_idx += 1
    task.putaijlist(rows, cols, vals)
    # Build the right-hand-sides of the [in]equality constraints
    type_constraint = [mosek.boundkey.fx] * n_lse + [mosek.boundkey.up] * p_lse
    h = np.concatenate([-log_c_lse, np.ones(p_lse)])
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
        con_indices = np.arange(cur_con_idx, cur_con_idx + log_c_lin.size)
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
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    if verbose:

        def streamprinter(text):
            """A helper, for logging MOSEK output to sys.stdout."""
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
        task.getsuxslice(mosek.soltype.itr, m + 3*n_lse + 1, msk_nvars, z_duals)
        z_duals = np.array(z_duals)
        z_duals[z_duals < 0] = 0
        # recover dual variables for the remaining user-provided constraints
        if log_c_lin.size > 0:
            aff_duals = [0.] * log_c_lin.size
            task.getsucslice(mosek.soltype.itr, n_lse + p_lse, cur_con_idx,
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

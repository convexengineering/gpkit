# -*- coding: utf-8 -*-
"""Module for using the MOSEK 9 Fusion interface

    Example
    -------
    ``result = _mosek.expopt9.imize(cs, A, p_idxs)``

    Raises
    ------
    ImportError
        If the local MOSEK library could not be loaded

"""

import os
import sys

try:
    from mosek.fusion import *  # pylint: disable=import-error
except Exception as e:
    raise ImportError("Could not load MOSEK library: "+repr(e))


from numpy import log, exp, array
import numpy as np

# Models log(sum(exp(Ax+b))) <= 0.
# Each row of [A b] describes one of the exp-terms
def logsumexp(M, A, x, b):
    k = int(A.shape[0])
    u = M.variable(k)
    a = M.constraint(Expr.sum(u), Domain.equalsTo(1.0))
    b = M.constraint(Expr.hstack(u,
                                 Expr.constTerm(k, 1.0),
                                 Expr.add(Expr.mul(A, x), b)),
                                          Domain.inPExpCone())
    return a, b


def imize(c, A, k, *args, **kwargs):
    with Model('gpkitmodel') as M:  # save M somewhere
        A = np.array(A.todense(), "double")
        vars = M.variable(A.shape[1])
        M.objective('Objective', ObjectiveSense.Minimize,
                    Expr.dot(A[0, :], vars))

        constraints = [0.0]*(len(k)-1)
        acc = 1  # fix for posynomial objectives
        for i in range(1, len(k)):
            n = k[i]
            if n > 1:
                constraints[i-1] = logsumexp(M, A[acc:acc+n, :], vars, np.log(c[acc:acc+n]))
            else:
                constraints[i-1] = M.constraint(Expr.dot(A[acc, :], vars),
                                                Domain.lessThan(-np.log(c[acc])))
            acc += n

        M.setLogHandler(sys.stdout)
        M.acceptedSolutionStatus(AccSolutionStatus.Anything)
        M.solve()
        duals = []
        for con in constraints:
            if isinstance(con, tuple):
                duali = [-c.dual() for c in con]
                duals.append(duali[0][0])
            else:
                duals.append(-con.dual()[0])

        return dict(status=str(M.getPrimalSolutionStatus())[15:],
                    objective=np.exp(M.primalObjValue()),
                    primal=vars.level(),
                    la=duals)

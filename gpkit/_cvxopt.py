"Implements the GPkit interface to CVXOPT"
from cvxopt import spmatrix, matrix, log
from cvxopt.solvers import gp
from cvxopt.info import version as cvxopt_version


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
    gpsolver = gp if cvxopt_version >= "1.1.8" else gp118
    # above use of gp118 is a local hack for windows
    # until Python (x,y) updates their cvxopt version

    g = log(matrix(c))
    F = spmatrix(A.data, A.row, A.col, tc='d')
    solution = gpsolver(k, F, g, *args, **kwargs)
    return dict(status=solution['status'],
                primal=solution['x'],
                la=solution['znl'])


def gp118(K, F, g, G=None, h=None, A=None, b=None, kktsolver=None, **kwargs):

    # pylint: disable=trailing-whitespace
    # pylint: disable=bad-whitespace
    # pylint: disable=unidiomatic-typecheck
    # pylint: disable=no-name-in-module
    # pylint: disable=invalid-name
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    # pylint: disable=bad-continuation
    # pylint: disable=unused-variable
    # pylint: disable=multiple-statements
    # pylint: disable=no-member
    # pylint: disable=bad-indentation
    # pylint: disable=missing-docstring
    
    """
    This is GPkit's local copy of the gp method from cvxopt 1.1.8
    This is a patch for users with cvxopt < 1.1.8, for which gp
        did not accept kktsolver or **kwargs.

    -------------------------------------------------------------

    Solves a geometric program

        minimize    log sum exp (F0*x+g0)
        subject to  log sum exp (Fi*x+gi) <= 0,  i=1,...,m
                    G*x <= h      
                    A*x = b

    Input arguments.

        K is a list of positive integers [K0, K1, K2, ..., Km].

        F is a sum(K)xn dense or sparse 'd' matrix with block rows F0, 
        F1, ..., Fm.  Each Fi is Kixn.

        g is a sum(K)x1 dense or sparse 'd' matrix with blocks g0, g1, 
        g2, ..., gm.  Each gi is Kix1.

        G is an mxn dense or sparse 'd' matrix.

        h is an mx1 dense 'd' matrix.

        A is a pxn dense or sparse 'd' matrix.

        b is a px1 dense 'd' matrix.

        The default values for G, h, A and b are empty matrices with 
        zero rows.


    Output arguments.

        Returns a dictionary with keys 'status', 'x', 'snl', 'sl',
        'znl', 'zl', 'y', 'primal objective', 'dual objective', 'gap',
        'relative gap', 'primal infeasibility', 'dual infeasibility',
        'primal slack', 'dual slack'.

        The 'status' field has values 'optimal' or 'unknown'.
        If status is 'optimal', x, snl, sl, y, znl, zl  are approximate 
        solutions of the primal and dual optimality conditions

            f(x)[1:] + snl = 0,  G*x + sl = h,  A*x = b 
            Df(x)'*[1; znl] + G'*zl + A'*y + c = 0 
            snl >= 0,  znl >= 0,  sl >= 0,  zl >= 0
            snl'*znl + sl'* zl = 0,

        where fk(x) = log sum exp (Fk*x + gk). 

        If status is 'unknown', x, snl, sl, y, znl, zl are the last
        iterates before termination.  They satisfy snl > 0, znl > 0, 
        sl > 0, zl > 0, but are not necessarily feasible.

        The values of the other fields are the values returned by cpl()
        applied to the epigraph form problem

            minimize   t 
            subjec to  f0(x) <= t
                       fk(x) <= 0, k = 1, ..., mnl
                       G*x <= h
                       A*x = b.

        Termination with status 'unknown' indicates that the algorithm 
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, snl, sl, znl, zl are close to optimal.


    Control parameters.

       The following control parameters can be modified by adding an
       entry to the dictionary options.

       options['show_progress'] True/False (default: True)
       options['maxiters'] positive integer (default: 100)
       options['refinement'] nonnegative integer (default: 1)
       options['abstol'] scalar (default: 1e-7)
       options['reltol'] scalar (default: 1e-6)
       options['feastol'] scalar (default: 1e-7).
    """

    options = kwargs.get('options', {}) # GPkit: changed globals to {}

    import math 
    from cvxopt import base, blas, misc
    from cvxopt.cvxprog import cp # added for GPkit


    if type(K) is not list or [ k for k in K if type(k) is not int 
        or k <= 0 ]:
        raise TypeError("'K' must be a list of positive integers")
    mnl = len(K)-1
    l = sum(K)

    if type(F) not in (matrix, spmatrix) or F.typecode != 'd' or \
        F.size[0] != l:
        raise TypeError("'F' must be a dense or sparse 'd' matrix "\
            "with %d rows" %l)
    if type(g) is not matrix or g.typecode != 'd' or g.size != (l,1): 
        raise TypeError("'g' must be a dene 'd' matrix of "\
            "size (%d,1)" %l)
    n = F.size[1]

    if G is None: G = spmatrix([], [], [], (0,n))
    if h is None: h = matrix(0.0, (0,1))
    if type(G) not in (matrix, spmatrix) or G.typecode != 'd' or \
        G.size[1] != n:
        raise TypeError("'G' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    ml = G.size[0]
    if type(h) is not matrix or h.typecode != 'd' or h.size != (ml,1):
        raise TypeError("'h' must be a dense 'd' matrix of "\
            "size (%d,1)" %ml)
    dims = {'l': ml, 's': [], 'q': []}

    if A is None: A = spmatrix([], [], [], (0,n))
    if b is None: b = matrix(0.0, (0,1))
    if type(A) not in (matrix, spmatrix) or A.typecode != 'd' or \
        A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    p = A.size[0]
    if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1): 
        raise TypeError("'b' must be a dense 'd' matrix of "\
            "size (%d,1)" %p)

    y = matrix(0.0, (l,1))
    u = matrix(0.0, (max(K),1))
    Fsc = matrix(0.0, (max(K),n))

    cs1 = [ sum(K[:i]) for i in range(mnl+1) ] 
    cs2 = [ cs1[i] + K[i] for i in range(mnl+1) ]
    ind = list(zip(range(mnl+1), cs1, cs2))

    def Fgp(x = None, z = None):

        if x is None: return mnl, matrix(0.0, (n,1))
	
        f = matrix(0.0, (mnl+1,1))
        Df = matrix(0.0, (mnl+1,n))

        # y = F*x+g
        blas.copy(g, y)
        base.gemv(F, x, y, beta=1.0)

        if z is not None: H = matrix(0.0, (n,n))

        for i, start, stop in ind:

            # yi := exp(yi) = exp(Fi*x+gi) 
            ymax = max(y[start:stop])
            y[start:stop] = base.exp(y[start:stop] - ymax)

            # fi = log sum yi = log sum exp(Fi*x+gi)
            ysum = blas.asum(y, n=stop-start, offset=start)
            f[i] = ymax + math.log(ysum)

            # yi := yi / sum(yi) = exp(Fi*x+gi) / sum(exp(Fi*x+gi))
            blas.scal(1.0/ysum, y, n=stop-start, offset=start)

            # gradfi := Fi' * yi 
            #        = Fi' * exp(Fi*x+gi) / sum(exp(Fi*x+gi))
            base.gemv(F, y, Df, trans='T', m=stop-start, incy=mnl+1,
                offsetA=start, offsetx=start, offsety=i)

            if z is not None:

                # Hi = Fi' * (diag(yi) - yi*yi') * Fi 
                #    = Fisc' * Fisc
                # where 
                # Fisc = diag(yi)^1/2 * (I - 1*yi') * Fi
                #      = diag(yi)^1/2 * (Fi - 1*gradfi')

                Fsc[:K[i], :] = F[start:stop, :] 
                for k in range(start,stop):
                   blas.axpy(Df, Fsc, n=n, alpha=-1.0, incx=mnl+1,
                       incy=Fsc.size[0], offsetx=i, offsety=k-start)
                   blas.scal(math.sqrt(y[k]), Fsc, inc=Fsc.size[0],
                       offset=k-start)

                # H += z[i]*Hi = z[i] * Fisc' * Fisc
                blas.syrk(Fsc, H, trans='T', k=stop-start, alpha=z[i],
                    beta=1.0)

        if z is None: return f, Df
        else: return f, Df, H

    return cp(Fgp, G, h, dims, A, b, kktsolver = kktsolver)

def cvxoptimize_fn(options):
    from cvxopt import solvers, spmatrix, matrix, log, exp
    solvers.options.update({'show_progress': False})
    solvers.options.update(options)
    gpsolver = solvers.gp

    def cvxoptimize(c, A, p_idxs, k):
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
            A: floats array of shape (m,n)
                Exponents of the various free variables for each monomial.
            p_idxs: ints array of shape n
                Posynomial index of each monomial (not used)

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
        solution = gpsolver(k, F, g)
        return dict(status=solution['status'],
                    primal=solution['x'],
                    la=solution['znl'])

    return cvxoptimize

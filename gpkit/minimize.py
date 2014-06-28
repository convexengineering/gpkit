import numpy as np
from scipy.sparse import dok_matrix
from scipy import float32
from itertools import chain

def chooser(cost, constraints, solver, options):
    posynomials = [cost]+constraints

    freevars = frozenset().union(*[p.vars for p in posynomials])
    monomials = list(chain(*[p.monomials for p in posynomials]))

    # c: constant coefficients of the various monomials in F
    c = np.array([m.c for m in monomials])
    assert all(c>0), "Negative coefficients are not allowed:\n%s"%c

    # k: number of monomials (columns of F) present in each constraint
    k = [len(p.monomials) for p in posynomials]

    # A: exponents of the various free variables for each monomial
    #    rows of A are variables, columns are monomials
    A = dok_matrix((len(freevars),len(monomials)), dtype=float32)
    for i, v in enumerate(freevars):
        for j, m in enumerate(monomials):
            if m.exps[v]: A[i,j] = m.exps[v]
    A = A.tocoo()

    # map: posynomial index (0 for cost function)
    i  = 0
    map_ = []
    for l in k:
        map_ += [i]*l
        i += 1

    if solver == 'cvxopt':
        solution = cvxopt_solver(c, A, k, options)
    elif solver == "mosek_cli":
        filename = options.get('filename', 'gpkit_mosek')
        solution = mosek_cli(c, A, map_, filename)
    else:
        raise Exception("That solver is not implemented!")

    return dict(zip(freevars, solution))


def cvxopt_solver(c, A, k, options):
    from cvxopt import sparse, matrix, log, exp

    solvers.options.update(options)

    k = k
    g = log(matrix(c))
    F = sparse(A.data, A.row, A.column)

    solution = solvers.gp(k, F, g)
    # TODO: catch errors, delays, etc.
    return exp(solution['x'])


from subprocess import check_output

def mosek_cli(c, A, map_, filename):
    with open(filename, 'w') as f:
        numcon = 1+map_[-1]
        numter, numvar = map(int, A.shape)
        for n in [numcon, numter, numvar]:
            f.write("%d\n" % n)

        f.write("\n*c\n")
        f.writelines(["%.20e\n" % x for x in c])

        f.write("\n*map_\n")
        f.writelines(["%d\n" % x for x in map_])

        t_j_Atj = np.array([A.col, A.row, A.data]).T.tolist()

        f.write("\n*t  j  A_tj\n")
        f.writelines(["%d %d %.20e\n" % tuple(x)
                    for x in t_j_Atj])

    check_output("mskexpopt "+filename, shell=True)

    with open(filename+".sol") as f:
        assert f.readline() == "PROBLEM STATUS      : PRIMAL_AND_DUAL_FEASIBLE\n"
        assert f.readline() == "SOLUTION STATUS     : OPTIMAL\n"
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        vals = []
        for line in f:
            if line == "\n": break
            else:
                idx, val = line.split()
                vals.append(np.exp(float(val)))
        return vals

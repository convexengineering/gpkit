from scipy import float32
from itertools import chain
from collections import namedtuple

coot_matrix = namedtuple('coot', ['row', 'col', 'data'])
class coot_matrix(coot_matrix):
    shape = (None, None)
    dt = float32

    def append(self, i, j, x):
        assert (i >= 0 and j >= 0), "Positive indices only in a coot matrix"
        self.row.append(i)
        self.col.append(j)
        self.data.append(self.dt(x))

    def update_shape(self):
        self.shape = (max(self.row)+1, max(self.col)+1)

def chooser(cost, constraints, solver, options, debug=False):
    # print a nice summary somtimes
    if debug:
        print "\t minimize"
        print "\t\t", cost
        print "\t subject to"
        for p in constraints:
            print "\t\t", p, "<= 1"

    posynomials = [cost]+constraints
    freevars = frozenset().union(*[p.vars for p in posynomials])
    monomials = list(chain(*[p.monomials for p in posynomials]))

    # c: constant coefficients of the various monomials in F
    c = [m.c for m in monomials]
    assert all([x>0 for x in c]), "Negative coefficients not allowed:\n%s"%c

    # k: number of monomials (columns of F) present in each constraint
    k = [len(p.monomials) for p in posynomials]

    # A: exponents of the various free variables for each monomial
    #    rows of A are variables, columns are monomials
    A = coot_matrix([], [], [])
    for i, v in enumerate(freevars):
        for j, m in enumerate(monomials):
            if v in m.vars:
                A.append(i, j, m.exps[v])
    A.update_shape()

    if debug:
        from scipy.sparse import coo_matrix
        print coo_matrix((A.data, (A.row, A.col))).todense()

    # p_idx: posynomial index (0 for cost function)
    #        mosek can't even keep straight what they call this,
    #        so I'm giving it this more descriptive name
    i  = 0
    p_idx = []
    for l in k:
        p_idx += [i]*l
        i += 1

    if solver == 'cvxopt':
        solution = cvxoptimize(c, A, k, options)

    elif solver == "mosek_cli":
        import _mosek.cli_expopt
        filename = options.get('filename', 'gpkit_mosek')
        solution = _mosek.cli_expopt.imize(c, A, p_idx, filename)

    elif solver == "mosek":
        import _mosek.expopt
        solution = _mosek.expopt.imize(c, A, p_idx)

    elif solver == "attached":
        solution = options['solver'](c, A, p_idx)
    else:
        raise Exception("That solver is not implemented!")

    return dict(zip(freevars, solution))



def cvxoptimize(c, A, k, options):
    from cvxopt import sparse, matrix, log, exp
    solvers.options.update(options)
    k = k
    g = log(matrix(c))
    F = sparse(A.data, A.row, A.column)
    solution = solvers.gp(k, F, g)
    # TODO: catch errors, delays, etc.
    return exp(solution['x'])
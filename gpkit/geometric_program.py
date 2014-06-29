from itertools import chain
from collections import namedtuple
from collections import defaultdict

coot_matrix = namedtuple('coot', ['row', 'col', 'data'])
class coot_matrix(coot_matrix):
    shape = (None, None)

    def append(self, i, j, x):
        assert (i >= 0 and j >= 0), "Positive indices only in a coot matrix"
        self.row.append(i)
        self.col.append(j)
        self.data.append(x)

    def update_shape(self):
        self.shape = (max(self.row)+1, max(self.col)+1)

    def todense(self):
        from scipy.sparse import coo_matrix
        return coo_matrix((self.data, (self.row, self.col))).todense()


class GP(object):

    def __repr__(self):
        return "\n".join([
               "gpkit.GeometricProgram:",
               "  minimize",
               "     %s" % self.cost,
               "  subject to"] +
              ["     %s <= 1" % p 
               for p in self.constraints] +
              ["  via the %s solver" % self.solver]
                )

    def __init__(self, cost, constraints,
                 constants={}, solver='mosek', options={}):
        self.cost = cost
        self.constraints = constraints
        self.constants = constants
        self.new_constants = True
        self.solver = solver
        self.options = options

        self.posynomials = [cost]+constraints
        self.posynomials_update()

    def posynomials_update(self):
        posynomials = self.posynomials
        self.variables = set().union(*[p.vars for p in posynomials])
        self.monomials = list(chain(*[p.monomials for p in posynomials]))

        # k: number of monomials (columns of F) present in each constraint
        self.k = [len(p.monomials) for p in posynomials]

        # p_idx: posynomial index (0 for cost function)
        #        mosek can't even keep straight what they call this,
        #        so I'm giving it this more descriptive name
        i  = 0
        self.p_idx = []
        for l in self.k:
            self.p_idx += [i]*l
            i += 1

        self.var_locations = defaultdict(list)
        for i, m in enumerate(self.monomials):
            for var in m.vars:
                self.var_locations[var].append(i)

        self.monomials_update()

    def monomials_update(self):
        if self.new_constants:
            self.subbed_monomials = [m.sub(self.constants)
                                     for m in self.monomials]
            self.freevars = self.variables.difference(self.constants)
            self.new_constants = False

        monomials = self.subbed_monomials

        # c: constant coefficients of the various monomials in F
        self.c = [m.c for m in monomials]

        # A: exponents of the various free variables for each monomial
        #    rows of A are variables, columns are monomials
        self.A = coot_matrix([], [], [])
        for i, v in enumerate(self.freevars):
            for j, m in enumerate(monomials):
                if v in m.vars:
                    self.A.append(i, j, m.exps[v])
        self.A.update_shape()

    def replace_constants(self, constants):
        self.new_constants = True
        self.constants = constants
        self.monomials_update()
        
    def add_constraints(self, posynomials):
        if not isinstance(posynomials, list):
            posynomials = [posynomials]

        self.posynomials += posynomials
        self.posynomials_update()

    def rm_constraints(self, posynomials):
        if not isinstance(posynomials, list):
            posynomials = [posynomials]

        for p in posynomial:
            self.posynomials.remove(p)
        self.posynomials_update()


    def solve(self):
        c, A, k, p_idx = self.c, self.A, self.k, self.p_idx
        solver = self.solver
        options = self.options

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


        self.solution = dict(zip(self.freevars, solution))
        return self.solution



def cvxoptimize(c, A, k, options):
    from cvxopt import sparse, matrix, log, exp
    solvers.options.update(options)
    k = k
    g = log(matrix(c))
    F = sparse(A.data, A.row, A.column)
    solution = solvers.gp(k, F, g)
    # TODO: catch errors, delays, etc.
    return exp(solution['x'])
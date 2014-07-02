from itertools import chain
from collections import namedtuple
from collections import defaultdict
from collections import Iterable

try:
    import numpy as np
except ImportError:
    print "Could not import numpy: will not be able to sweep variables"

try:
    import scipy
    from scipy.sparse import coo_matrix
except ImportError:
    print "Could not import scipy: will not be able to use splines"

coot_matrix = namedtuple('coot', ['row', 'col', 'data'])


class coot_matrix(coot_matrix):
    "A very simple sparse matrix representation."
    shape = (None, None)

    def append(self, i, j, x):
        assert (i >= 0 and j >= 0), "Only positive indices allowed"
        self.row.append(i)
        self.col.append(j)
        self.data.append(x)

    def update_shape(self):
        self.shape = (max(self.row)+1, max(self.col)+1)

    def tocoo(self):
        "Converts to a Scipy sparse coo_matrix"
        return coo_matrix((self.data, (self.row, self.col)))

    def todense(self): return self.tocoo().todense()
    def tocsr(self): return self.tocoo().tocsr()
    def tocsc(self): return self.tocoo().tocsc()
    def todok(self): return self.tocoo().todok()
    def todia(self): return self.tocoo().todia()


class GP(object):

    def __repr__(self):
        return "\n".join(["gpkit.GeometricProgram:",
                          "  minimize",
                          "     %s" % self.cost,
                          "  subject to"] +
                         ["     %s <= 1" % p
                          for p in self.constraints] +
                         ["  via the %s solver" % self.solver]
                         )

    def __init__(self, cost, constraints,
                 constants={}, sweep={},
                 solver='mosek', options={}):
        self.cost = cost
        self.constraints = constraints
        self.sweep = sweep
        self.solver = solver
        self.options = options
        self.constants = {}
        self.constants_update(constants)
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
        i = 0
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

    def constants_update(self, newconstants):
        for var, constant in newconstants.iteritems():
            if isinstance(constant, Iterable):
                pass  # catch vector-valued constants
        self.constants.update(newconstants)
        self.new_constants = True
        if hasattr(self, 'monomials'):
            self.monomials_update()

    def add_constraints(self, posynomials):
        if not isinstance(posynomials, list):
            posynomials = [posynomials]

        self.posynomials += posynomials
        self.posynomials_update()

    def rm_constraints(self, posynomials):
        if not isinstance(posynomials, list):
            posynomials = [posynomials]

        for p in posynomials:
            self.posynomials.remove(p)
        self.posynomials_update()

    def solve(self):
        self.solution = {}
        if self.sweep:
            solution = self._solve_sweep()
        else:
            result = self.__run_solver()
            self.check_result(result)
            solution = dict(zip(self.freevars,
                                result['primal_sol']))
        self.solution = solution
        return self.solution

    def _solve_sweep(self):
        self.constants_update({var: None for var in self.sweep})

        sweep_grids = np.meshgrid(*self.sweep.values())
        sweep_shape = sweep_grids[0].shape
        N_passes = sweep_grids[0].size
        sweep_grids = dict(zip(self.sweep, sweep_grids))
        sweep_vects = {var: grid.reshape(N_passes)
                       for (var, grid) in sweep_grids.iteritems()}
        result_2d_array = np.empty((N_passes, len(self.freevars)))

        for i in xrange(N_passes):
            this_pass = {var: sweep_vect[i]
                         for (var, sweep_vect) in sweep_vects.iteritems()}
            self.constants_update(this_pass)

            result = self.__run_solver()
            self.check_result(result)
            result_2d_array[i,:] = result['primal_sol']

        solution = {var: result_2d_array[:,j].reshape(sweep_shape)
                    for (j, var) in enumerate(self.freevars)}
        solution.update(sweep_grids)
        return solution

    def __run_solver(self):
        if self.solver == 'cvxopt':
            return cvxoptimize(self.c,
                               self.A,
                               self.k,
                               self.options)
        elif self.solver == "mosek_cli":
            import _mosek.cli_expopt
            filename = self.options.get('filename', 'gpkit_mosek')
            return _mosek.cli_expopt.imize(self.c,
                                           self.A,
                                           self.p_idx,
                                           filename)
        elif self.solver == "mosek":
            import _mosek.expopt
            return _mosek.expopt.imize(self.c,
                                       self.A,
                                       self.p_idx)
        elif self.solver == "attached":
            return self.options['solver'](self.c,
                                          self.A,
                                          self.p_idx)
        else:
            raise Exception("Solver %s is not implemented!"
                            % self.solver)

    def check_result(self, result):
        assert result['success']
        self.check_feasibility(result['primal_sol'])

    def check_feasibility(self, primal_sol):
        allconsts = dict(self.constants)
        allconsts.update(dict(zip(self.freevars, primal_sol)))
        if not set(allconsts) == self.variables:
            raise RuntimeWarning("Did not solve for all variables!")
        for p in self.constraints:
            val = sum([m.sub(allconsts).c for m in p.monomials])
            if not val <= 1 + 1e-4:
                raise RuntimeWarning("Constraint broken:"
                                     " %s = 1 + %0.2e" % (p, val-1))


def cvxoptimize(c, A, k, options):
    from cvxopt import solvers, spmatrix, matrix, log, exp
    solvers.options.update(options)
    k = k
    g = log(matrix(c))
    F = spmatrix(A.data, A.row, A.col)
    solution = solvers.gp(k, F, g)
    # TODO: catch errors, delays, etc.
    return dict(success=True,
                primal_sol=exp(solution['x']))

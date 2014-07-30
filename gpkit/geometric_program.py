from collections import defaultdict
from collections import namedtuple
from collections import Iterable
from helpers import *

from gpkit.nomials import Posynomial

try:
    import numpy as np
except ImportError:
    print "Could not import numpy: will not be able to sweep variables"

try:
    from scipy.sparse import coo_matrix
except ImportError:
    print "Could not import scipy: will not be able to use splines"

CootMatrix = namedtuple('CootMatrix', ['row', 'col', 'data'])
PosyTuple = namedtuple('PosyTuple', ['exps', 'cs', 'var_locs', 'var_descrs',
                                     'substitutions'])


import time


class CootMatrix(CootMatrix):
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
                          "     %s" % self.cost._string(),
                          "  subject to"] +
                         ["     %s <= 1" % p._string()
                          for p in self.constraints] +
                         ["  via the %s solver" % self.solver]
                         )

    def __init__(self, cost, constraints, constants={},
                 solver=None, options={}):
        self.cost = cost
        self.constraints = tuple(constraints)

        self.options = options
        if solver is not None:
            self.solver = solver
        else:
            from gpkit import settings
            self.solver = settings['defaultsolver']

        self.sweep = {}
        self._gen_unsubbed_vars()

        if constants:
            self.sub(constants, tobase='initialsub')

    def print_boundwarnings(self):
        for var, bound in self.missingbounds.iteritems():
            print "%s (%s) has no %s bound" % (
                  var, self.var_descrs[var], bound)

    def add_constraints(self, constraints):
        if isinstance(constraints, Posynomial):
            constraints = [constraints]
        self.constraints += tuple(constraints)
        self._gen_unsubbed_vars()

    def rm_constraints(self, constraints):
        if isinstance(constraints, Posynomial):
            constraints = [constraints]
        for p in constraints:
            self.constraints.remove(p)
        self._gen_unsubbed_vars()

    def _gen_unsubbed_vars(self):
        posynomials = (self.cost,)+self.constraints

        var_descrs = defaultdict(str)
        for p in posynomials:
            var_descrs.update(p.var_descrs)

        exps = sumlist(posynomials, attr='exps')
        cs = sumlist(posynomials, attr='cs')
        var_locs = locate_vars(exps)

        self.unsubbed = PosyTuple(exps, cs, var_locs, var_descrs, {})
        self.load(self.unsubbed)

        # k [j]: number of monomials (columns of F) present in each constraint
        self.k = [len(p.cs) for p in posynomials]

        # p_idxs [i]: posynomial index of each monomial
        p_idx = 0
        self.p_idxs = []
        for p_len in self.k:
            self.p_idxs += [p_idx]*p_len
            p_idx += 1

    def sub(self, substitutions, val=None, frombase='last', tobase='subbed'):
        # look for sweep variables
        sweepdescrs = {}
        if isinstance(substitutions, dict):
            subs = dict(substitutions)
            for var, sub in substitutions.iteritems():
                try:
                    if sub[0] == 'sweep':
                        del subs[var]
                        if isinstance(sub[1], Iterable):
                            self.sweep.update({var: sub[1]})
                            if isinstance(sub[-1], str):
                                if isinstance(sub[-2], str):
                                    sweepdescrs.update({var: sub[-2:]})
                                else:
                                    sweepdescrs.update({var: [None, sub[-1]]})
                    else:
                        raise ValueError("sweep variables must be iterable.")
                except: pass
        else:
            subs = substitutions

        base = getattr(self, frombase)

        # perform substitution
        var_locs, exps, cs, newdescrs, subs = substitution(base.var_locs,
                                                           base.exps,
                                                           base.cs,
                                                           subs, val)
        if not subs:
            raise ValueError("could not find anything to substitute")

        var_descrs = base.var_descrs
        var_descrs.update(newdescrs)
        var_descrs.update(sweepdescrs)
        substitutions = base.substitutions
        substitutions.update(subs)

        newbase = PosyTuple(exps, cs, var_locs, var_descrs, substitutions)
        setattr(self, tobase, self.last)
        self.load(newbase)
        self.print_boundwarnings()

    def load(self, posytuple):
        self.last = posytuple
        for attr in ['exps', 'cs', 'var_locs', 'var_descrs', 'substitutions']:
            new = getattr(posytuple, attr)
            setattr(self, attr, new)

        # A: exponents of the various free variables for each monomial
        #    rows of A are variables, columns are monomials
        self.missingbounds = {}
        self.A = CootMatrix([], [], [])
        for j, var in enumerate(self.var_locs):
            varsign = None
            for i in self.var_locs[var]:
                exp = self.exps[i][var]
                self.A.append(j, i, exp)
                if varsign is None: varsign = np.sign(exp)
                elif varsign is "both": pass
                elif np.sign(exp) != varsign: varsign = "both"
            if varsign != "both" and var not in self.sweep:
                if varsign == 1: bound = "lower"
                elif varsign == -1: bound = "upper"
                self.missingbounds[var] = bound
        self.A.update_shape()

    def solve(self, printing=True):
        if printing: print "Using solver '%s'" % self.solver
        self.starttime = time.time()
        self.data = {}
        if self.sweep:
            self.solution = self._solve_sweep(printing)
        else:
            result = self.__run_solver()
            self.check_result(result)
            self.solution = dict(zip(self.var_locs,
                                     result['primal_sol']))
            self.sensitivities = self._sensitivities(result)
            self.solution.update(self.sensitivities)

        self.data.update(self.substitutions)
        self.data.update(self.solution)
        self.endtime = time.time()
        if printing: print "Solving took %.3g seconds   " % (self.endtime - self.starttime)
        return self.data

    def _sensitivities(self, result):
        dss = result['dual_sol']
        senstuple = [('S{%s}' % var, sum([self.unsubbed.exps[i][var]*dss[i]
                                          for i in locs]))
                     for (var, locs) in self.unsubbed.var_locs.iteritems()]
        sensdict = {var: val for (var, val) in
                    filter(lambda x: abs(x[1]) >= 0.01, senstuple)}
        return sensdict

    def _solve_sweep(self, printing):
        self.presweep = self.last
        self.sub({var: 1 for var in self.sweep}, tobase='swept')

        sweep_dims = len(self.sweep)
        if sweep_dims == 1:
            sweep_grids = self.sweep.values()
        else:
            sweep_grids = np.meshgrid(*self.sweep.values())
        sweep_shape = sweep_grids[0].shape
        N_passes = sweep_grids[0].size
        if printing:
            print "Sweeping %i variables over %i passes" % (
                  sweep_dims, N_passes)
        sweep_grids = dict(zip(self.sweep, sweep_grids))
        sweep_vects = {var: grid.reshape(N_passes)
                       for (var, grid) in sweep_grids.iteritems()}
        result_2d_array = np.empty((N_passes, len(self.var_locs)))

        for i in xrange(N_passes):
            this_pass = {var: sweep_vect[i]
                         for (var, sweep_vect) in sweep_vects.iteritems()}
            self.sub(this_pass, frombase='presweep', tobase='swept')

            result = self.__run_solver()
            self.check_result(result)
            result_2d_array[i, :] = result['primal_sol']

        solution = {var: result_2d_array[:, j].reshape(sweep_shape)
                    for (j, var) in enumerate(self.var_locs)}
        solution.update(sweep_grids)

        self.load(self.presweep)
        return solution

    def __run_solver(self):
        if self.solver == 'cvxopt':
            result = cvxoptimize(self.cs,
                                 self.A,
                                 self.k,
                                 self.options)
        elif self.solver == "mosek_cli":
            import _mosek.cli_expopt
            filename = self.options.get('filename', 'gpkit_mosek')
            result = _mosek.cli_expopt.imize(self.cs,
                                             self.A,
                                             self.p_idxs,
                                             filename)
        elif self.solver == "mosek":
            import _mosek.expopt
            result = _mosek.expopt.imize(self.cs,
                                         self.A,
                                         self.p_idxs)
        elif self.solver == "attached":
            result = self.options['solver'](self.cs,
                                            self.A,
                                            self.p_idxs,
                                            self.k)
        else:
            raise Exception("Solver %s is not implemented!" % self.solver)

        self.result = result
        return result

    def check_result(self, result):
        assert result['success']
        # TODO: raise InfeasibilityWarning
        # self.check_feasibility(result['primal_sol'])

    def check_feasibility(self, primal_sol):
        allsubs = dict(self.substitutions)
        allsubs.update(dict(zip(self.var_locs, primal_sol)))
        for p in self.constraints:
            val = p.sub(allsubs).c
            if not val <= 1 + 1e-4:
                raise RuntimeWarning("Constraint broken:"
                                     " %s = 1 + %0.2e" % (p, val-1))


def cvxoptimize(c, A, k, options):
    from cvxopt import solvers, spmatrix, matrix, log, exp
    solvers.options.update({'show_progress': False})
    solvers.options.update(options)
    k = k
    g = log(matrix(c))
    F = spmatrix(A.data, A.col, A.row, tc='d')
    solution = solvers.gp(k, F, g)
    # TODO: catch errors, delays, etc.
    return dict(success=True,
                primal_sol=exp(solution['x']),
                dual_sol=solution['y'])

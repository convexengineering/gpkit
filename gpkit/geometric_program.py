from models import Model

try:
    import numpy as np
except ImportError:
    print "Could not import numpy: will not be able to sweep variables"

import time
import pprint

import gpkit.plotting


class GP(Model):

    def __eq__(self, other):
        return str(self) == str(other)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "\n".join(["gpkit.GP( # minimize",
                          "          %s," % self.cost._string(),
                          "          [   # subject to"] +
                         ["              %s," % p._string()
                          for p in self.constraints] +
                         ['          ],',
                          "          substitutions={ %s }," %
                          pprint.pformat(self.substitutions, indent=26)[26:-1],
                          '          solver="%s")' % self.solver]
                         )

    def _latex(self, unused=None):
        return "\n".join(["\\begin{array}[ll]",
                          "\\text{}"
                          "\\text{minimize}",
                          "    & %s \\\\" % self.cost._latex(),
                          "\\text{subject to}"] +
                         ["    & %s \\\\" % c._latex() for c in self.constraints] +
                         ["\\end{array}"])

    def __init__(self, cost, constraints, substitutions={},
                 solver=None, options={}):
        self.cost = cost
        self.constraints = tuple(constraints)
        self.posynomials = (self.cost,)+self.constraints

        self.options = options
        if solver is not None:
            self.solver = solver
        else:
            from gpkit import settings
            self.solver = settings['defaultsolver']

        self.sweep = {}
        self._gen_unsubbed_vars()

        if substitutions:
            self.sub(substitutions, tobase='initialsub')

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

    def plot_frontiers(self, Zs, x, y, figsize):
        if len(self.sweep) == 2:
            gpkit.plotting.contour_array(self.data,
                                         self.var_descrs,
                                         self.sweep.keys()[0],
                                         self.sweep.keys()[1],
                                         Zs, x, y, figsize,
                                         xticks=self.sweep.values()[0],
                                         yticks=self.sweep.values()[1])


def cvxoptimize(c, A, k, options):
    from cvxopt import solvers, spmatrix, matrix, log, exp
    solvers.options.update({'show_progress': False})
    solvers.options.update(options)
    g = log(matrix(c))
    F = spmatrix(A.data, A.col, A.row, tc='d')
    solution = solvers.gp(k, F, g)
    # TODO: catch errors, delays, etc.
    return dict(success=True,
                primal_sol=exp(solution['x']),
                dual_sol=solution['y'])

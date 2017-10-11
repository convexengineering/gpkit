"""Implement the SequentialGeometricProgram class"""
from time import time
from ..exceptions import InvalidGPConstraint
from ..keydict import KeyDict
from ..nomials import Variable
from .costed import CostedConstraintSet
from .gp import GeometricProgram
from ..solution_array import SolutionArray
from ..nomials import SignomialInequality, PosynomialInequality


# pylint: disable=too-many-instance-attributes
class SequentialGeometricProgram(CostedConstraintSet):
    """Prepares a collection of signomials for a SP solve.

    Arguments
    ---------
    cost : Posynomial
        Objective to minimize when solving
    constraints : list of Constraint or SignomialConstraint objects
        Constraints to maintain when solving (implicitly Signomials <= 1)
    verbosity : int (optional)
        Currently has no effect: SequentialGeometricPrograms don't know
        anything new after being created, unlike GeometricPrograms.

    Attributes with side effects
    ----------------------------
    `gps` is set during a solve
    `result` is set at the end of a solve

    Examples
    --------
    >>> gp = gpkit.geometric_program.SequentialGeometricProgram(
                        # minimize
                        x,
                        [   # subject to
                            1/x - y/x,  # <= 1, implicitly
                            y/10  # <= 1
                        ])
    >>> gp.solve()
    """

    def __init__(self, cost, constraints, substitutions=None, verbosity=1):
        # pylint: disable=unused-argument
        self.gps = []
        self.results = []
        self.result = None
        self._spconstrs = []
        self._approx_lt = []
        self._numgpconstrs = None
        self._gp = None

        if cost.any_nonpositive_cs:
            raise TypeError("""Sequential GPs need Posynomial objectives.

    The equivalent of a Signomial objective can be constructed by constraining
    a dummy variable `z` to be greater than the desired Signomial objective `s`
    (z >= s) and then minimizing that dummy variable.""")
        CostedConstraintSet.__init__(self, cost, constraints, substitutions,
                                     add_cost_values_to_substitutions=False)
        self.externalfn_vars = frozenset(Variable(newvariable=False, **v.descr)
                                         for v in self.varkeys if v.externalfn)
        self.not_sp = bool(self.externalfn_vars)
        if not self.not_sp:
            self._gp = self.init_gp(self.substitutions, verbosity)
            if not (self.not_sp or self._gp[0][1]):  # [0][1]: sp constraints
                raise ValueError("""Model valid as a Geometric Program.

    SequentialGeometricPrograms should only be created with Models containing Signomial
    Constraints, since Models without Signomials have global solutions and can
    be solved with 'Model.solve()'.""")

    # pylint: disable=too-many-locals
    def localsolve(self, solver=None, verbosity=1, x0=None, reltol=1e-4,
                   iteration_limit=50, mutategp=True, **kwargs):
        """Locally solves a SequentialGeometricProgram and returns the solution.

        Arguments
        ---------
        solver : str or function (optional)
            By default uses one of the solvers found during installation.
            If set to "mosek", "mosek_cli", or "cvxopt", uses that solver.
            If set to a function, passes that function cs, A, p_idxs, and k.
        verbosity : int (optional)
            If greater than 0, prints solve time and number of iterations.
            Each GP is created and solved with verbosity one less than this, so
            if greater than 1, prints solver name and time for each GP.
        x0 : dict (optional)
            Initial location to approximate signomials about.
        reltol : float
            Iteration ends when this is greater than the distance between two
            consecutive solve's objective values.
        iteration_limit : int
            Maximum GP iterations allowed.
        *args, **kwargs :
            Passed to solver function.


        Returns
        -------
        result : dict
            A dictionary containing the translated solver result.
        """
        starttime = time()
        if verbosity > 0:
            print("Beginning signomial solve.")
        self.gps = []  # NOTE: SIDE EFFECTS
        self.results = []
        if x0 and mutategp:
            self._gp = self.init_gp(self.substitutions, verbosity, x0)
        slackvar = Variable()
        prevcost, cost, rel_improvement = None, None, None
        while rel_improvement is None or rel_improvement > reltol:
            if len(self.gps) > iteration_limit:
                raise RuntimeWarning("""problem unsolved after %s iterations.

    The last result is available in Model.program.gps[-1].result. If the gps
    appear to be converging, you may wish to increase the iteration limit by
    calling .localsolve(..., iteration_limit=NEWLIMIT).""" % len(self.gps))
            gp = self.gp(x0, verbosity-1, mutategp)
            self.gps.append(gp)  # NOTE: SIDE EFFECTS
            try:
                result = gp.solve(solver, verbosity-1, **kwargs)
                self.results.append(result)
            except (RuntimeWarning, ValueError):
                feas_constrs = ([slackvar >= 1] +
                                [posy <= slackvar
                                 for posy in gp.posynomials[1:]])
                primal_feas = GeometricProgram(slackvar**100 * gp.cost,
                                               feas_constrs,
                                               verbosity=verbosity-1)
                self.gps.append(primal_feas)
                result = primal_feas.solve(solver, verbosity=verbosity-1)
                result["cost"] = None  # reset the cost-counting
            x0 = result["freevariables"]
            prevcost, cost = cost, result["cost"]
            if prevcost and cost:
                rel_improvement = abs(prevcost-cost)/(prevcost + cost)
            else:
                rel_improvement = None
        # solved successfully!
        soltime = time() - starttime
        if verbosity > 0:
            print("Solving took %i GP solves" % len(self.gps)
                  + " and %.3g seconds." % soltime)
        self.process_result(result)
        self.result = SolutionArray(result.copy())  # NOTE: SIDE EFFECTS
        self.result["soltime"] = soltime
        return self.result

    def _fill_x0(self, x0):
        "Returns a copy of x0 with subsitutions and sp_inits added."
        x0 = KeyDict(x0) if x0 else KeyDict()
        for key in self.varkeys:
            if key in x0:
                continue  # already specified by input dict
            elif key in self.substitutions:
                x0[key] = self.substitutions[key]
            elif key.sp_init:
                x0[key] = key.sp_init
            # undeclared variables are handled by individual constraints
        return x0

    def init_gp(self, substitutions, verbosity=1, x0=None):
        "Generates a simplified GP representation for later modification"
        gpconstrs = []
        self._spconstrs = []
        self._approx_lt = []
        approx_gt = []
        x0 = self._fill_x0(x0)
        for cs in self.flat(constraintsets=False):
            try:
                if not isinstance(cs, PosynomialInequality):
                    cs.as_posyslt1(substitutions)  # is it gp-compatible?
                gpconstrs.append(cs)
            except InvalidGPConstraint:
                if isinstance(cs, SignomialInequality):
                    self._spconstrs.append(cs)
                    self._approx_lt.extend(cs.as_approxslt())
                    # assume unspecified negy variables have a value of 1.0
                    x0.update({vk: 1.0 for vk in cs.varkeys if vk not in x0})
                    approx_gt.extend(cs.as_approxsgt(x0))
                else:
                    self.not_sp = True
                    return
        spapproxs = [p/m <= 1 for p, m in zip(self._approx_lt, approx_gt)]
        gp = GeometricProgram(self.cost, [gpconstrs, spapproxs],
                              substitutions, verbosity=verbosity)
        gp.x0 = x0
        self._numgpconstrs = len(gp.hmaps) - len(spapproxs)
        return gp

    def gp(self, x0=None, verbosity=1, mutategp=False):
        "The GP approximation of this SP at x0."
        if mutategp and not self.not_sp:
            if self.gps:  # update self._gp with new x0
                self._gp.x0.update(x0)
                mono_gts = []
                for spc in self._spconstrs:
                    mono_gts.extend(spc.as_approxsgt(self._gp.x0))
                for i, mono_gt in enumerate(mono_gts):
                    posy_lt = self._approx_lt[i]
                    unsubbed = posy_lt/mono_gt
                    # the index [0][1] gets the set of all sp constraints
                    self._gp[0][1][i].unsubbed = [unsubbed]
                    # TODO: cache parsed self.substitutions for each spmono
                    smap = unsubbed.hmap.sub(self.substitutions,
                                             unsubbed.varkeys)
                    self._gp.hmaps[self._numgpconstrs+i] = smap
                    # TODO: WHY ON EARTH IS THIS LINE REQUIRED:
                    self._gp.posynomials[self._numgpconstrs+i].hmap = smap
                self._gp.gen()
            return self._gp
        else:
            x0 = self._fill_x0(x0)
            gp_constrs = self.as_gpconstr(x0, self.substitutions)
            if self.externalfn_vars:
                gp_constrs.extend([v.key.externalfn(v, x0)
                                   for v in self.externalfn_vars])
            gp = GeometricProgram(self.cost, gp_constrs,
                                  self.substitutions, verbosity=verbosity)
            gp.x0 = x0  # NOTE: SIDE EFFECTS
            return gp

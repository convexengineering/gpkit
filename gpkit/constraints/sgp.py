"""Implement the SequentialGeometricProgram class"""
from time import time
import numpy as np
from collections import OrderedDict
from ..exceptions import InvalidGPConstraint
from ..keydict import KeyDict
from ..nomials import Variable
from .gp import GeometricProgram
from ..nomials import SignomialInequality, PosynomialInequality
from .costed import CostedConstraintSet
from ..small_scripts import mag


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
    def __init__(self, cost, constraints, substitutions):
        # pylint:disable=super-init-not-called
        self.gps = []
        self.solver_outs = []
        self._results = []
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
        self.__bare_init__(cost, constraints, substitutions, varkeys=True)
        self.externalfn_vars = frozenset(Variable(v) for v in self.varkeys
                                         if v.externalfn)
        self.externalfns = bool(self.externalfn_vars)
        if not self.externalfns:
            self._gp = self.init_gp(self.substitutions)
            if self._gp and not self._gp["SP approximations"]:
                raise ValueError("""Model valid as a Geometric Program.

    SequentialGeometricPrograms should only be created with Models containing
    Signomial Constraints, since Models without Signomials have global
    solutions and can be solved with 'Model.solve()'.""")

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
        self.solver_outs = []
        self._results = []
        # if there's external functions we can't mutate the GP
        mutategp = mutategp and not self.externalfns
        if x0 and not mutategp:
            self._gp = self.init_gp(self.substitutions, x0)
        slackvar = Variable()
        prevcost, cost, rel_improvement = None, None, None
        while rel_improvement is None or rel_improvement > reltol:
            if len(self.gps) > iteration_limit:
                raise RuntimeWarning("""problem unsolved after %s iterations.

    The last result is available in Model.program.gps[-1].result. If the gps
    appear to be converging, you may wish to increase the iteration limit by
    calling .localsolve(..., iteration_limit=NEWLIMIT).""" % len(self.gps))
            gp = self.gp(x0, mutategp)
            self.gps.append(gp)  # NOTE: SIDE EFFECTS
            try:
                solver_out = gp.solve(solver, verbosity-1,
                                      warn_on_check=True,
                                      gen_result=False, **kwargs)
                self.solver_outs.append(solver_out)
                x0 = KeyDict(zip(gp.varlocs, np.exp(solver_out["primal"])))
                if "objective" in solver_out:
                    cost = float(solver_out["objective"])
                else:
                    cost = mag(gp.posynomials[0].sub(x0).c)
            except (RuntimeWarning, ValueError):
                feas_constrs = ([slackvar >= 1] +
                                [posy <= slackvar
                                 for posy in gp.posynomials[1:]])
                primal_feas = GeometricProgram(slackvar**100 * gp.cost,
                                               feas_constrs, None)
                self.gps.append(primal_feas)
                solver_out = primal_feas.solve(solver, verbosity-1,
                                               gen_result=False, **kwargs)
                x0 = KeyDict(zip(primal_feas.varlocs,
                                 np.exp(solver_out["primal"])))
                cost = None  # reset the cost-counting
            if prevcost is None or cost is None:
                rel_improvement = None
            elif prevcost < (1-reltol)*cost:
                print("SP is not converging! Last GP iteration had a higher"
                      " cost (%.2g) than the previous one (%.2g). Results for"
                      " each iteration are in (Model).program.results. If your"
                      " model contains SignomialEqualities, note that"
                      " convergence is not guaranteed: try replacing any"
                      " SigEqs you can and solving again." % (cost, prevcost))
            else:
                rel_improvement = abs(prevcost-cost)/(prevcost + cost)
            prevcost = cost
        # solved successfully!
        self.result = gp.generate_result(solver_out, verbosity)
        soltime = time() - starttime
        if verbosity > 0:
            print("Solving took %i GP solves" % len(self.gps)
                  + " and %.3g seconds." % soltime)
        self.process_result(self.result)
        self.result["soltime"] = soltime
        if self.externalfn_vars:
            for v in self.externalfn_vars:
                self[0].insert(0, v.key.externalfn)  # for constraint senss
        return self.result

    @property
    def results(self):
        "Creates and caches results from the raw solver_outs"
        if not self._results:
            self._results = [so["gen_result"]() for so in self.solver_outs]
        return self._results

    def _fill_x0(self, x0):
        "Returns a copy of x0 with subsitutions added."
        x0 = KeyDict(x0) if x0 else KeyDict()
        for key in self.varkeys:
            if key in x0:
                continue  # already specified by input dict
            elif key in self.substitutions:
                x0[key] = self.substitutions[key]
            # undeclared variables are handled by individual constraints
        return x0

    def init_gp(self, substitutions, x0=None):
        "Generates a simplified GP representation for later modification"
        gpconstrs, self._spconstrs, self._approx_lt, approx_gt = [], [], [], []
        x0 = self._fill_x0(x0)
        for cs in self.flat():
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
                    self.externalfns = True
                    return None
        spapproxs = [p/m <= 1 for p, m in zip(self._approx_lt, approx_gt)]
        for pconstr, spconstr in zip(spapproxs, self._spconstrs):
            pconstr.sgp_parent = spconstr
        gp = GeometricProgram(self.cost, OrderedDict((
            ("GP constraints", gpconstrs),
            ("SP approximations", spapproxs))), substitutions)
        gp.x0 = x0
        self._numgpconstrs = len(gp.hmaps) - len(spapproxs)
        return gp

    def gp(self, x0=None, mutategp=False):
        "The GP approximation of this SP at x0."
        if mutategp:
            if not self.gps:
                return self._gp  # we've already generated the first gp
            gp = self._gp        # otherwise, update it with a new x0
            gp.x0.update(x0)
            mono_gts = []
            for spc in self._spconstrs:
                mono_gts.extend(spc.as_approxsgt(gp.x0))
            for i, mono_gt in enumerate(mono_gts):
                posy_lt = self._approx_lt[i]
                unsubbed = posy_lt/mono_gt
                gp["SP approximations"][i].unsubbed = [unsubbed]
                smap = unsubbed.hmap.sub(self.substitutions,
                                         unsubbed.varkeys)
                gp.hmaps[self._numgpconstrs+i] = smap
            gp.gen()
        else:
            x0 = self._fill_x0(x0)
            gp_constrs = self.as_gpconstr(x0)
            if self.externalfn_vars:
                for v in self.externalfn_vars:
                    posyconstr = v.key.externalfn(v, x0)
                    posyconstr.sgp_parent = v.key.externalfn
                    gp_constrs.append(posyconstr)
            gp = GeometricProgram(self.cost, gp_constrs, self.substitutions)
            gp.x0 = x0  # NOTE: SIDE EFFECTS
        return gp

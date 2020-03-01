"""Implement the SequentialGeometricProgram class"""
from time import time
from collections import OrderedDict
import numpy as np
from ..exceptions import InvalidGPConstraint, Infeasible
from ..keydict import KeyDict
from ..nomials import Variable
from .gp import GeometricProgram
from ..nomials import PosynomialInequality
from .. import NamedVariables
from .costed import CostedConstraintSet
from ..small_scripts import mag


EPS = 1e-6  # determines what counts as "convergence"

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
    gps = solver_outs = _results = result = None
    _gp = _spvars = _sp_constraints = _lt_approxs = None
    with NamedVariables("PCCP"):
        slack = Variable("slack")

    def __init__(self, cost, constraints, substitutions, **initgpargs):
        # pylint:disable=super-init-not-called
        self.__bare_init__(cost, constraints, substitutions)
        if cost.any_nonpositive_cs:
            raise TypeError("""Sequential GPs need Posynomial objectives.

    The equivalent of a Signomial objective can be constructed by constraining
    a dummy variable `z` to be greater than the desired Signomial objective `s`
    (z >= s) and then minimizing that dummy variable.""")
        self.externalfn_vars = \
            frozenset(Variable(v) for v in self.varkeys if v.externalfn)
        if self.externalfn_vars:
            self.blackboxconstraints = True
        else:
            try:
                self._gp = self.init_gp(self.substitutions, **initgpargs)
                self.blackboxconstraints = False
            except AttributeError:
                self.blackboxconstraints = True
            else:
                if not self._gp["SP approximations"]:
                    raise ValueError("""Model valid as a Geometric Program.

    SequentialGeometricPrograms should only be created with Models containing
    Signomial Constraints, since Models without Signomials have global
    solutions and can be solved with 'Model.solve()'.""")

    # pylint: disable=too-many-locals,too-many-branches
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-statements
    def localsolve(self, solver=None, *, verbosity=1, x0=None, reltol=1e-4,
                   iteration_limit=50, mutategp=True, **solveargs):
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
        mutategp: boolean
            Prescribes whether to mutate the previously generated GP
            or to create a new GP with every solve.
        **solveargs :
            Passed to solver function.

        Returns
        -------
        result : dict
            A dictionary containing the translated solver result.
        """
        starttime = time()
        if verbosity > 0:
            print("Beginning signomial solve.")
        self.gps, self.solver_outs = [], []  # NOTE: SIDE EFFECTS
        # if there's external functions we can't mutate the GP
        mutategp = mutategp and not self.blackboxconstraints
        if not mutategp and not x0:
            raise ValueError("Solves with arbitrary constraint generators"
                             " must specify an initial starting point x0.")
        if mutategp:
            if x0:
                self._gp = self.init_gp(self.substitutions, x0)
            gp = self._gp
        prevcost, cost, rel_improvement = None, None, None
        while rel_improvement is None or rel_improvement > reltol:
            prevcost = cost
            if len(self.gps) > iteration_limit:
                raise Infeasible(
                    "Unsolved after %s iterations. Check `m.program.results`;"
                    " if they're converging, try `.localsolve(...,"
                    " iteration_limit=NEWLIMIT)`." % len(self.gps))
            if mutategp:
                self.update_gp(x0)
            else:
                gp = self.gp(x0)
            self.gps.append(gp)  # NOTE: SIDE EFFECTS
            solver_out = gp.solve(solver, verbosity=verbosity-1,
                                  gen_result=False, **solveargs)
            self.solver_outs.append(solver_out)
            cost = float(solver_out["objective"])
            x0 = dict(zip(gp.varlocs, np.exp(solver_out["primal"])))
            if prevcost is None or cost is None:
                continue
            if cost*(1 - EPS) > prevcost + EPS and verbosity >= 0:
                print(
                    "SP is not converging! Last GP iteration had a higher cost"
                    " (%.2e) than the previous one (%+.2e). Check `m.program"
                    ".results`. If your model has SignomialEqualities,"
                    " convergence is not guaranteed: try replacing any SigEqs"
                    " you can and solving again." % (cost, cost - prevcost))
            rel_improvement = abs(prevcost - cost)/(prevcost + cost)
        # solved successfully!
        self.result = gp.generate_result(solver_out, verbosity=verbosity-1)
        self.result["soltime"] = time() - starttime
        if verbosity > 0:
            print("Solving took %i GP solves" % len(self.gps)
                  + " and %.3g seconds." % self.result["soltime"])
        self.process_result(self.result)
        if self.externalfn_vars:
            for v in self.externalfn_vars:
                self[0].insert(0, v.key.externalfn)  # for constraint senss
        try:  # check that there's not too much slack
            excess_slack = self.result["variables"][self.slack.key] - 1
            if excess_slack >= EPS:
                raise Infeasible(
                    "final slack on SP constraints was 1%+.2e. Result(s)"
                    " stored in `m.program.result(s)`." % excess_slack)
            del self.result["variables"][self.slack.key]
            del self.result["freevariables"][self.slack.key]
        except KeyError:
            pass  # not finding the slack key is just fine
        return self.result

    @property
    def results(self):
        "Creates and caches results from the raw solver_outs"
        if not self._results:
            self._results = [o["generate_result"]() for o in self.solver_outs]
        return self._results

    def _fill_x0(self, x0):
        "Returns a copy of x0 with subsitutions added."
        x0kd = KeyDict()
        x0kd.varkeys = self.varkeys
        if x0:
            x0kd.update(x0)  # has to occur after the setting of varkeys
        x0kd.update(self.substitutions)
        return x0kd

    def init_gp(self, substitutions, x0=None, **initgpargs):
        "Generates a simplified GP representation for later modification"
        x0 = self._fill_x0(x0)
        use_pccp = initgpargs.pop("use_pccp", True)
        pccp_penalty = initgpargs.pop("pccp_penalty", 10)
        constraints = OrderedDict((("SP approximations", []),
                                   ("GP constraints", [])))
        self._sp_constraints, self._lt_approxs = [], []
        self._spvars = set()
        for cs in self.flat():
            try:
                if not isinstance(cs, PosynomialInequality):
                    cs.as_hmapslt1(substitutions)  # is it gp-compatible?
                constraints["GP constraints"].append(cs)
            except InvalidGPConstraint:
                self._spvars.update(cs.varkeys)
                self._sp_constraints.append(cs)
                if use_pccp:
                    lts = [lt/self.slack for lt in cs.as_approxlts()]
                else:
                    lts = cs.as_approxlts()
                self._lt_approxs.extend(lts)
                for lt, gt in zip(lts, cs.as_approxgts(x0)):
                    constraint = (lt <= gt)
                    constraint.sgp_parent = cs
                    constraints["SP approximations"].append(constraint)
        if use_pccp:
            cost = self.cost * self.slack**pccp_penalty
            constraints["Slack restriction"] = (self.slack >= 1)
        else:
            cost = self.cost
        gp = GeometricProgram(cost, constraints, substitutions, **initgpargs)
        gp.x0 = x0
        return gp

    def update_gp(self, x0):
        "Update self._gp for x0."
        if not self.gps:
            return  # we've already generated the first gp
        gp = self._gp
        gp.x0.update({k: v for (k, v) in x0.items() if k in self._spvars})
        lt_idx = 0
        for sp_constraint in self._sp_constraints:
            for mono_gt in sp_constraint.as_approxgts(gp.x0):
                unsubbed = self._lt_approxs[lt_idx]/mono_gt
                gp["SP approximations"][lt_idx].unsubbed = [unsubbed]
                lt_idx += 1  # here because gp.hmaps[0] is the cost hmap
                gp.hmaps[lt_idx] = unsubbed.hmap.sub(self.substitutions,
                                                     unsubbed.varkeys)
        gp.gen()

    def gp(self, x0=None, **gpinitargs):
        "The GP approximation of this SP at x0."
        x0 = self._fill_x0(x0)
        constraints = OrderedDict(
            {"Approximations of existing constraints": self.as_gpconstr(x0)})
        if self.externalfn_vars:
            constraints["Constraints generated by externalfns"] = []
            for v in self.externalfn_vars:
                con = v.key.externalfn(v, x0)
                con.sgp_parent = v.key.externalfn
                constraints["Constraints generated by externalfns"].append(con)
        gp = GeometricProgram(self.cost, constraints, self.substitutions,
                              **gpinitargs)
        gp.x0 = x0  # NOTE: SIDE EFFECTS
        return gp

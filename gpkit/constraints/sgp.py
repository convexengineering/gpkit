"""Implement the SequentialGeometricProgram class"""
from time import time
from collections import OrderedDict
import numpy as np
from ..exceptions import InvalidGPConstraint, Infeasible, UnnecessarySGP
from ..keydict import KeyDict
from ..nomials import Variable
from .gp import GeometricProgram
from .set import sort_constraints_dict
from ..nomials import PosynomialInequality
from .. import NamedVariables
from .costed import CostedConstraintSet


EPS = 1e-6  # 1 +/- this is used in a few relative differences

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
    gps = solver_outs = _results = result = model = None
    _gp = _spvars = _lt_approxs = pccp_penalty = None
    with NamedVariables("SGP"):
        slack = Variable("PCCPslack")

    def __init__(self, cost, model, substitutions, *,
                 use_pccp=True, pccp_penalty=2e2, **initgpargs):
        if cost.any_nonpositive_cs:
            raise UnnecessarySGP("""Sequential GPs need Posynomial objectives.

    The equivalent of a Signomial objective can be constructed by constraining
    a dummy variable `z` to be greater than the desired Signomial objective `s`
    (z >= s) and then minimizing that dummy variable.""")
        self._original_cost = cost
        self.model = model
        self.substitutions = substitutions
        self.externalfn_vars = \
            frozenset(Variable(v) for v in self.model.varkeys if v.externalfn)
        if self.externalfn_vars:  # a non-SP-constraint generating variable
            self.blackboxconstraints = True
            super().__init__(cost, model, substitutions)
            return

        self._lt_approxs = []
        sgpconstraints = {"SP constraints": [], "GP constraints": []}
        for cs in model.flat():
            try:
                if not isinstance(cs, PosynomialInequality):
                    cs.as_hmapslt1(substitutions)  # gp-compatible?
                sgpconstraints["GP constraints"].append(cs)
            except InvalidGPConstraint:
                sgpconstraints["SP constraints"].append(cs)
                try:
                    if use_pccp:
                        lts = [lt/self.slack for lt in cs.as_approxlts()]
                    else:
                        lts = cs.as_approxlts()
                    self._lt_approxs.append(lts)
                except AttributeError:  # some custom non-SP constraint
                    self.blackboxconstraints = True
                    super().__init__(cost, model, substitutions)
                    return
        # all constraints seem SP-compatible
        self.blackboxconstraints = False
        if not sgpconstraints["SP constraints"]:
            raise UnnecessarySGP("""Model valid as a Geometric Program.

SequentialGeometricPrograms should only be created with Models containing
Signomial Constraints, since Models without Signomials have global
solutions and can be solved with 'Model.solve()'.""")

        if not use_pccp:
            self.cost = cost
        else:
            self.pccp_penalty = pccp_penalty
            self.cost = cost * self.slack**pccp_penalty
            sgpconstraints["GP constraints"].append(self.slack >= 1)

        keys, sgpconstraints = sort_constraints_dict(sgpconstraints)
        self.idxlookup = {k: i for i, k in enumerate(keys)}
        list.__init__(self, sgpconstraints)  # pylint: disable=non-parent-init-called
        self._gp = self.init_gp(**initgpargs)

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
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
        self.gps, self.solver_outs, self._results = [], [], []
        # if there's external functions we can't mutate the GP
        mutategp = mutategp and not self.blackboxconstraints
        if not mutategp and not x0:
            raise ValueError("Solves with arbitrary constraint generators"
                             " must specify an initial starting point x0.")
        if mutategp:
            if x0:
                self._gp = self.init_gp(x0)
            gp = self._gp
        starttime = time()
        if verbosity > 0:
            print("Starting a sequence of GP solves")
            if self.externalfn_vars:
                print(" for %i variables defined by externalfns"
                      % len(self.externalfn_vars))
            elif mutategp:
                print(" for %i free variables" % len(self._spvars))
                print("  in %i signomial constraints"
                      % len(self["SP constraints"]))
            print("  and for %i free variables" % len(gp.varlocs))
            print("       in %i posynomial inequalities." % len(gp.k))
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
            gp.model = self.model
            self.gps.append(gp)  # NOTE: SIDE EFFECTS
            if verbosity > 1:
                print("\nGP Solve %i" % len(self.gps))
            if verbosity > 2:
                print("===============")
            solver_out = gp.solve(solver, verbosity=verbosity-1,
                                  gen_result=False, **solveargs)
            self.solver_outs.append(solver_out)
            cost = float(solver_out["objective"])
            x0 = dict(zip(gp.varlocs, np.exp(solver_out["primal"])))
            if verbosity > 2 and self._spvars:
                result = gp.generate_result(solver_out, verbosity=verbosity-3)
                self._results.append(result)
                print(result.table(self._spvars))
            elif verbosity > 1:
                print("Solved cost was %.4g." % cost)
            if prevcost is None:
                continue
            rel_improvement = (prevcost - cost)/(prevcost + cost)
            if cost*(1 - EPS) > prevcost + EPS and verbosity > -1:
                print("SGP not convergent: Cost rose by %.2g%% on GP solve %i."
                      " Details can be found in `m.program.results` or by"
                      " solving at a higher verbosity. Note that convergence is"
                      " not guaranteed for models with SignomialEqualities.\n"
                      % (100*(cost - prevcost)/prevcost, len(self.gps)))
                rel_improvement = cost = None
        # solved successfully!
        self.result = gp.generate_result(solver_out, verbosity=verbosity-3)
        self.result["soltime"] = time() - starttime
        if verbosity > 1:
            print()
        if verbosity > 0:
            print("Solving took %.3g seconds and %i GP solves."
                  % (self.result["soltime"], len(self.gps)))
        self.model.process_result(self.result)
        if self.externalfn_vars:
            for v in self.externalfn_vars:
                self[0].insert(0, v.key.externalfn)  # for constraint senss
        if self.slack.key in self.result["variables"]:
            excess_slack = self.result["variables"][self.slack.key] - 1
            if excess_slack <= EPS:
                del self.result["freevariables"][self.slack.key]
                del self.result["variables"][self.slack.key]
                del self.result["sensitivities"]["variables"][self.slack.key]
                slackconstraint = self["GP constraints"][-1]
                del self.result["sensitivities"]["constraints"][slackconstraint]
            elif verbosity > -1:
                print("Final solution let signomial constraints slacken by"
                      " %.2g%%. Calling .localsolve with a higher"
                      " `pccp_penalty` (it was %.3g this time) will reduce"
                      " final slack if the model is solvable with less. If"
                      " you think it might not be, check by solving with "
                      "`use_pccp=False, x0=(this model's final solution)`.\n"
                      % (100*excess_slack, self.pccp_penalty))
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
        x0kd.varkeys = self.model.varkeys
        if x0:
            x0kd.update(x0)  # has to occur after the setting of varkeys
        x0kd.update(self.substitutions)
        return x0kd

    def init_gp(self, x0=None, **initgpargs):
        "Generates a simplified GP representation for later modification"
        x0 = self._fill_x0(x0)
        constraints = OrderedDict({"SP approximations": []})
        constraints["GP constraints"] = self["GP constraints"]
        self._spvars = set([self.slack])
        for cs, lts in zip(self["SP constraints"], self._lt_approxs):
            for lt, gt in zip(lts, cs.as_approxgts(x0)):
                constraint = (lt <= gt)
                constraint.generated_by = cs
                constraints["SP approximations"].append(constraint)
                self._spvars.update({vk for vk in gt.varkeys
                                     if vk not in self.substitutions})
        gp = GeometricProgram(self.cost, constraints, self.substitutions,
                              **initgpargs)
        gp.x0 = x0
        return gp

    def update_gp(self, x0):
        "Update self._gp for x0."
        if not self.gps:
            return  # we've already generated the first gp
        gp = self._gp
        gp.x0.update({k: v for (k, v) in x0.items() if k in self._spvars})
        hmap_idx = 0
        for sp_constraint, lts in zip(self["SP constraints"], self._lt_approxs):
            for lt, gt in zip(lts, sp_constraint.as_approxgts(gp.x0)):
                unsubbed = lt/gt
                gp["SP approximations"][hmap_idx].unsubbed = [unsubbed]
                hmap = unsubbed.hmap.sub(self.substitutions, unsubbed.varkeys)
                hmap.parent = gp["SP approximations"][hmap_idx]
                hmap_idx += 1  # here because gp.hmaps[0] is the cost hmap
                gp.hmaps[hmap_idx] = hmap
        gp.gen()

    def gp(self, x0=None, **gpinitargs):
        "The GP approximation of this SP at x0."
        x0 = self._fill_x0(x0)
        constraints = OrderedDict(
            {"SP constraints": [c.as_gpconstr(x0) for c in self.model.flat()]})
        if self.externalfn_vars:
            constraints["Generated by externalfns"] = []
            for v in self.externalfn_vars:
                constraint = v.key.externalfn(v, x0)
                constraint.generated_by = v.key.externalfn
                constraints["Generated by externalfns"].append(constraint)
        gp = GeometricProgram(self._original_cost,
                              constraints, self.substitutions, **gpinitargs)
        gp.x0 = x0
        return gp

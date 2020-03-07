"""Implement the SequentialGeometricProgram class"""
from time import time
from collections import OrderedDict, defaultdict
import numpy as np
from ..exceptions import (InvalidGPConstraint, Infeasible, UnnecessarySGP,
                          InvalidSGP)
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
    _gp = _spvars = pccp_penalty = None
    with NamedVariables("SGP"):
        slack = Variable("PCCPslack")

    def __init__(self, cost, model, substitutions, *,
                 use_pccp=True, pccp_penalty=2e2, checkbounds=True):
        if cost.any_nonpositive_cs:
            raise InvalidSGP("""Sequential GPs need Posynomial objectives.

    The equivalent of a Signomial objective can be constructed by constraining
    a dummy variable `z` to be greater than the desired Signomial objective `s`
    (z >= s) and then minimizing that dummy variable.""")
        self._original_cost = cost
        self.model = model
        self.substitutions = substitutions

        sgpconstraints = {"SP constraints": [], "GP constraints": []}
        for cs in model.flat():
            try:
                if not isinstance(cs, PosynomialInequality):
                    cs.as_hmapslt1(substitutions)  # gp-compatible?
                sgpconstraints["GP constraints"].append(cs)
            except InvalidGPConstraint:
                if not hasattr(cs, "approx_as_posyslt1"):
                    raise InvalidSGPConstraint()
                sgpconstraints["SP constraints"].append(cs)
        # all constraints seem SP-compatible
        if not sgpconstraints["SP constraints"]:
            raise UnnecessarySGP("""Model valid as a Geometric Program.

SequentialGeometricPrograms should only be created with Models containing
Signomial Constraints, since Models without Signomials have global
solutions and can be solved with 'Model.solve()'.""")

        if not use_pccp:
            self.cost = cost
            from ..nomials import Monomial
            self.slack = Monomial(1)
        else:
            self.pccp_penalty = pccp_penalty
            self.cost = cost * self.slack**pccp_penalty
            sgpconstraints["GP constraints"].append(self.slack >= 1)

        keys, sgpconstraints = sort_constraints_dict(sgpconstraints)
        self.idxlookup = {k: i for i, k in enumerate(keys)}
        list.__init__(self, sgpconstraints)  # pylint: disable=non-parent-init-called
        self._gp = self.init_gp(checkbounds=checkbounds)

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def localsolve(self, solver=None, *, verbosity=1, x0=None, reltol=1e-4,
                   iteration_limit=50, **solveargs):
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
        starttime = time()
        if verbosity > 0:
            print("Starting a sequence of GP solves")
            print(" for %i free variables" % len(self._spvars))
            print("  in %i signomial constraints"
                  % len(self["SP constraints"]))
            print("  and for %i free variables" % len(self._gp.varlocs))
            print("       in %i posynomial inequalities." % len(self._gp.k))
        prevcost, cost, rel_improvement = None, None, None
        while rel_improvement is None or rel_improvement > reltol:
            prevcost = cost
            if len(self.gps) > iteration_limit:
                raise Infeasible(
                    "Unsolved after %s iterations. Check `m.program.results`;"
                    " if they're converging, try `.localsolve(...,"
                    " iteration_limit=NEWLIMIT)`." % len(self.gps))
            gp = self.gp(x0, cleanx0=True)
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
            if verbosity > 2:
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
        if getattr(self.slack, "key", None) in self.result["variables"]:
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

    def init_gp(self, checkbounds=True):
        "Generates a simplified GP representation for later modification"
        x0 = KeyDict()
        x0.varkeys = self.model.varkeys
        x0.update(self.substitutions)
        # OrderedDict so that SP constraints are at the first indices
        constraints = OrderedDict({"SP approximations": []})
        constraints["GP constraints"] = self["GP constraints"]
        self._spvars = set()
        for cs in self["SP constraints"]:
            for posylt1 in cs.approx_as_posyslt1(x0):
                constraint = (posylt1 <= self.slack)
                constraint.generated_by = cs
                constraints["SP approximations"].append(constraint)
                self._spvars.update(posylt1.varkeys)
        gp = GeometricProgram(self.cost, constraints, self.substitutions,
                              checkbounds=checkbounds)
        gp.x0 = x0
        self.a_idxs = defaultdict(list)
        cost_mons, sp_mons = gp.k[0], sum(gp.k[:1+len(self["SP constraints"])])
        for row_idx, m_idx in enumerate(gp.A.row):
            if cost_mons <= m_idx <= sp_mons:
                self.a_idxs[gp.p_idxs[m_idx]].append(row_idx)
        return gp

    def gp(self, x0={}, *, cleanx0=False):
        "Update self._gp for x0 and return it."
        if not x0:
            return self._gp  # return last generated
        if not cleanx0:
            x0 = KeyDict(x0)
        gp = self._gp
        gp.x0.update({vk: x0[vk] for vk in self._spvars if vk in x0})
        p_idx = 0
        for sp_constraint in self["SP constraints"]:
            for posylt1 in sp_constraint.approx_as_posyslt1(gp.x0):
                approx_constraint = gp["SP approximations"][p_idx]
                approx_constraint.unsubbed = [posylt1/self.slack]
                p_idx += 1  # p_idx=0 is the cost; sp constraints are after it
                hmap, = approx_constraint.as_hmapslt1(self.substitutions)
                gp.hmaps[p_idx] = hmap
                m_idx = gp.m_idxs[p_idx].start
                a_idxs = list(self.a_idxs[p_idx])  # A's entries we can modify
                for i, (exp, c) in enumerate(hmap.items()):
                    gp.exps[m_idx + i] = exp
                    gp.cs[m_idx + i] = c
                    for var, x in exp.items():
                        row_idx = a_idxs.pop()  # modify a particular A entry
                        gp.A.row[row_idx] = m_idx + i
                        gp.A.col[row_idx] = gp.varidxs[var]
                        gp.A.data[row_idx] = x
        return gp

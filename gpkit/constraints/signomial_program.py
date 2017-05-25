"""Implement the SignomialProgram class"""
from time import time
from ..exceptions import InvalidGPConstraint
from ..keydict import KeyDict
from ..nomials import Variable
from .costed import CostedConstraintSet
from .geometric_program import GeometricProgram
from ..solution_array import SolutionArray
from ..nomials import SignomialInequality


# pylint: disable=too-many-instance-attributes
class SignomialProgram(CostedConstraintSet):
    """Prepares a collection of signomials for a SP solve.

    Arguments
    ---------
    cost : Posynomial
        Objective to minimize when solving
    constraints : list of Constraint or SignomialConstraint objects
        Constraints to maintain when solving (implicitly Signomials <= 1)
    verbosity : int (optional)
        Currently has no effect: SignomialPrograms don't know anything new
        after being created, unlike GeometricPrograms.

    Attributes with side effects
    ----------------------------
    `gps` is set during a solve
    `result` is set at the end of a solve

    Examples
    --------
    >>> gp = gpkit.geometric_program.SignomialProgram(
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
        self.lastgp = None
        self.is_sgp = False
        self._spconstrs = []
        self._approx_lt = []
        self._gppos = None

        if cost.any_nonpositive_cs:
            raise TypeError("""SignomialPrograms need Posyomial objectives.

    The equivalent of a Signomial objective can be constructed by constraining
    a dummy variable z to be greater than the desired Signomial objective s
    (z >= s) and then minimizing that dummy variable.""")
        CostedConstraintSet.__init__(self, cost, constraints, substitutions)
        try:
            self.__parse_externalfnvars()
            if self.externalfn_vars:  # not a GP! Skip to the `except`
                raise InvalidGPConstraint("some variables have externalfns")
            _ = self.as_posyslt1(substitutions)
        except InvalidGPConstraint:
            pass
        else:  # this is a GP
            raise ValueError("""Model valid as a Geometric Program.

    SignomialPrograms should only be created with Models containing Signomial
    Constraints, since Models without Signomials have global solutions and can
    be solved with 'Model.solve()'.""")

    # pylint: disable=too-many-locals
    def localsolve(self, solver=None, verbosity=1, x0=None, reltol=1e-4,
                   iteration_limit=50, modifylastgp=True, **kwargs):
        """Locally solves a SignomialProgram and returns the solution.

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
        slackvar = Variable()
        prevcost, cost, rel_improvement = None, None, None
        while rel_improvement is None or rel_improvement > reltol:
            if len(self.gps) > iteration_limit:
                raise RuntimeWarning("""problem unsolved after %s iterations.

    The last result is available in Model.program.gps[-1].result. If the gps
    appear to be converging, you may wish to increase the iteration limit by
    calling .localsolve(..., iteration_limit=NEWLIMIT).""" % len(self.gps))
            gp = self.gp(x0, verbosity-1, modifylastgp)
            self.gps.append(gp)  # NOTE: SIDE EFFECTS
            try:
                result = gp.solve(solver, verbosity-1, **kwargs)
                self.lastgp = gp
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
        x0 = KeyDict(x0) if x0 is not None else KeyDict()
        for key in self.varkeys:
            if key in x0:
                continue  # already specified by input dict
            elif key in self.substitutions:
                x0[key] = self.substitutions[key]
            elif key.sp_init:
                x0[key] = key.sp_init
            # for now, variables not declared elsewhere are
            # left for the individual constraints to handle
        return x0

    def firstgp(self, x0, substitutions):
        "Generates a simplified GP representation for later modification"
        gpposys = []
        gpconstrs = []
        self._spconstrs = []
        self._approx_lt = []
        approx_gt = []
        for cs in self.flat(constraintsets=False):
            try:
                gpposys.extend(cs.as_posyslt1(substitutions))
                gpconstrs.append(cs)
            except InvalidGPConstraint:
                if isinstance(cs, SignomialInequality):
                    self._spconstrs.append(cs)
                    self._approx_lt.extend(cs.as_approxslt())
                    # assume unspecified negy variables have a value of 1.0
                    x0.update({vk: 1.0 for vk in cs.varkeys if vk not in x0})
                    approx_gt.extend(cs.as_approxsgt(x0))
                else:
                    self.is_sgp = True
                    return
        self._gppos = 1 + len(gpposys)
        return [gpconstrs,
                [p/m <= 1 for p, m in zip(self._approx_lt, approx_gt)]]

    def gp(self, x0=None, verbosity=1, modifylastgp=False):
        "The GP approximation of this SP at x0."
        x0 = self._fill_x0(x0)
        if not hasattr(self, "externalfn_vars"):
            self.__parse_externalfnvars()
        if modifylastgp and self.lastgp:
            spmonos = []
            for spc in self._spconstrs:
                spmonos.extend(spc.as_approxsgt(x0))
            for i, spmono in enumerate(spmonos):
                firstposy = self._approx_lt[i]
                unsubbed = firstposy/spmono
                self.lastgp[0][1][i].unsubbed = [unsubbed]
                localposylt1 = unsubbed.sub(self.substitutions)
                self.lastgp.posynomials[self._gppos+i] = localposylt1
            self.lastgp.gen()
            return self.lastgp
        else:
            if modifylastgp and not self.lastgp:
                gp_constrs = self.firstgp(x0, self.substitutions)
            if not modifylastgp or self.is_sgp:  # may be set by the above
                gp_constrs = self.as_gpconstr(x0, self.substitutions)  # pylint: disable=redefined-variable-type
                if self.externalfn_vars:
                    gp_constrs.extend([v.key.externalfn(v, x0)
                                       for v in self.externalfn_vars])
            gp = GeometricProgram(self.cost, gp_constrs,
                                  self.substitutions, verbosity=verbosity)
            gp.x0 = x0  # NOTE: SIDE EFFECTS
            return gp

    def __parse_externalfnvars(self):
        "If this hasn't already been done, look for vars with externalfns"
        self.externalfn_vars = frozenset(Variable(newvariable=False,
                                                  **v.descr)
                                         for v in self.varkeys
                                         if v.externalfn)
        self.is_sgp = bool(self.externalfn_vars)

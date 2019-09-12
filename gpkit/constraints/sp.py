"""Implement the SignomialProgram class"""
from __future__ import unicode_literals, print_function
from time import time
from collections import OrderedDict
import numpy as np
from ..exceptions import InvalidGPConstraint
from ..keydict import KeyDict
from ..nomials import Variable, VectorVariable
from .gp import GeometricProgram
from ..nomials import SignomialInequality, PosynomialInequality
from ..nomials import SingleSignomialEquality
from .. import SignomialsEnabled, NamedVariables
from .costed import CostedConstraintSet
from ..small_scripts import mag


# pylint: disable=too-many-instance-attributes
class SignomialProgram(CostedConstraintSet):
    """Prepares a collection of signomials for a SP solve.

    Arguments
    ---------
    cost : Posynomial
        Objective to minimize when solving
    constraints : list of Constraint or SignomialConstraint objects
    verbosity : int (optional)
        Currently has no effect: SignomialPrograms don't know
        anything new after being created, unlike GeometricPrograms.

    Attributes with side effects
    ----------------------------
    TODO: DETERMINE WHAT TO SET AFTER MEETING WITH RILEY
     `result` is set at the end of a solve

    """
    def __init__(self, cost, constraints, substitutions):
        # pylint:disable=super-init-not-called
        self.gps = []
        self.solver_outs = []
        self._results = []
        self.result = None
        self._spconstrs = []
        self._spvars = set()
        self._approx_lt = []
        self._numgpconstrs = None
        self._gp = None

        if cost.any_nonpositive_cs:
            raise TypeError("""SPs still need Posynomial objectives.

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

    SignomialPrograms should only be created with Models containing
    Signomial Constraints, since Models without Signomials have global
    solutions and can be solved with 'Model.solve()'.""")

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-statements
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
        mutategp: boolean
            Prescribes whether to mutate the previously generated GP
            or to create a new GP with every solve.
        *args, **kwargs :
            Passed to solver function.

        Returns
        -------
        result : dict
            A dictionary containing the translated solver result.
        """
        return

    def generate_matrices(self):
        return

    @property
    def results(self):
        "Creates and caches results from the raw solver_outs"
        if not self._results:
            self._results = [so["gen_result"]() for so in self.solver_outs]
        return self._results

    def _fill_x0(self, x0):
        "Returns a copy of x0 with subsitutions added."
        x0kd = KeyDict()
        x0kd.varkeys = self.varkeys
        if x0:
            x0kd.update(x0)
        for key in self.varkeys:
            if key in x0kd:
                continue  # already specified by input dict
            elif key in self.substitutions:
                x0kd[key] = self.substitutions[key]
            # undeclared variables are handled by individual constraints
        return x0kd


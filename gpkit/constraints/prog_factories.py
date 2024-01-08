"Scripts for generating, solving and sweeping programs"
from time import time
import warnings as pywarnings
import numpy as np
from adce import adnumber
from ..nomials import parse_subs
from ..solution_array import SolutionArray
from ..keydict import KeyDict
from ..small_scripts import maybe_flatten
from ..small_classes import FixedScalar
from ..exceptions import Infeasible
from ..globals import SignomialsEnabled


def evaluate_linked(constants, linked):
    # pylint: disable=too-many-branches
    "Evaluates the values and gradients of linked variables."
    kdc = KeyDict({k: adnumber(maybe_flatten(v), k)
                   for k, v in constants.items()})
    kdc_plain = None
    array_calulated = {}
    for key in constants:  # remove gradients from constants
        key.descr.pop("gradients", None)
    for v, f in linked.items():
        try:
            if v.veckey and v.veckey.vecfn:
                if v.veckey not in array_calulated:
                    with SignomialsEnabled():  # to allow use of gpkit.units
                        vecout = v.veckey.vecfn(kdc)
                    if not hasattr(vecout, "shape"):
                        vecout = np.array(vecout)
                    array_calulated[v.veckey] = vecout
                out = array_calulated[v.veckey][v.idx]
            else:
                with SignomialsEnabled():  # to allow use of gpkit.units
                    out = f(kdc)
            if isinstance(out, FixedScalar):  # to allow use of gpkit.units
                out = out.value
            if hasattr(out, "units"):
                out = out.to(v.units or "dimensionless").magnitude
            elif out != 0 and v.units:
                pywarnings.warn(
                    f"Linked function for {v} did not return a united value."
                    " Modifying it to do so (e.g. by using `()` instead of `[]`"
                    " to access variables) will reduce errors.")
            out = maybe_flatten(out)
            if not hasattr(out, "x"):
                constants[v] = out
                continue  # a new fixed variable, not a calculated one
            constants[v] = out.x
            v.descr["gradients"] = {adn.tag: grad
                                    for adn, grad in out.d().items()
                                    if adn.tag}
        except Exception as exception:  # pylint: disable=broad-except
            from .. import settings  # pylint: disable=import-outside-toplevel
            if settings.get("ad_errors_raise", None):
                raise
            if kdc_plain is None:
                kdc_plain = KeyDict(constants)
            constants[v] = f(kdc_plain)
            v.descr.pop("gradients", None)
            print("Warning: skipped auto-differentiation of linked variable"
                  f" {v} because {exception!r} was raised. Set `gpkit.settings"
                  "[\"ad_errors_raise\"] = True` to raise such Exceptions"
                  " directly.\n")
            if ("Automatic differentiation not yet supported for <class "
                    "'gpkit.nomials.math.Monomial'> objects") in str(exception):
                print("This particular warning may have come from using"
                      f" gpkit.units.* in the function for {v}; try using"
                      " gpkit.ureg.* or gpkit.units.*.units instead.")


def progify(program, return_attr=None):
    """Generates function that returns a program() and optionally an attribute.

    Arguments
    ---------
    program: NomialData
        Class to return, e.g. GeometricProgram or SequentialGeometricProgram
    return_attr: string
        attribute to return in addition to the program
    """
    def programfn(self, constants=None, **initargs):
        "Return program version of self"
        if not constants:
            constants, _, linked = parse_subs(self.varkeys, self.substitutions)
            if linked:
                evaluate_linked(constants, linked)
        prog = program(self.cost, self, constants, **initargs)
        prog.model = self  # NOTE SIDE EFFECTS
        if return_attr:
            return prog, getattr(prog, return_attr)
        return prog
    return programfn


def solvify(genfunction):
    "Returns function for making/solving/sweeping a program."
    def solvefn(self, solver=None, *, verbosity=1, skipsweepfailures=False,
                **kwargs):
        """Forms a mathematical program and attempts to solve it.

         Arguments
         ---------
         solver : string or function (default None)
             If None, uses the default solver found in installation.
         verbosity : int (default 1)
             If greater than 0 prints runtime messages.
             Is decremented by one and then passed to programs.
         skipsweepfailures : bool (default False)
             If True, when a solve errors during a sweep, skip it.
         **kwargs : Passed to solve and program init calls

         Returns
         -------
         sol : SolutionArray
             See the SolutionArray documentation for details.

         Raises
         ------
         ValueError if the program is invalid.
         RuntimeWarning if an error occurs in solving or parsing the solution.
         """
        constants, sweep, linked = parse_subs(self.varkeys, self.substitutions)
        solution = SolutionArray()
        solution.modelstr = str(self)

        # NOTE SIDE EFFECTS: self.program and self.solution set below
        if sweep:
            run_sweep(genfunction, self, solution, skipsweepfailures,
                      constants, sweep, linked, solver, verbosity, **kwargs)
        else:
            self.program, progsolve = genfunction(self, **kwargs)
            result = progsolve(solver, verbosity=verbosity, **kwargs)
            if kwargs.get("process_result", True):
                self.process_result(result)
            solution.append(result)
        solution.to_arrays()
        self.solution = solution
        return solution
    return solvefn


# pylint: disable=too-many-locals,too-many-arguments,too-many-branches,too-many-statements
def run_sweep(genfunction, self, solution, skipsweepfailures,
              constants, sweep, linked, solver, verbosity, **kwargs):
    "Runs through a sweep."
    # sort sweeps by the eqstr of their varkey
    sweepvars, sweepvals = zip(*sorted(list(sweep.items()),
                                       key=lambda vkval: vkval[0].eqstr))
    if len(sweep) == 1:
        sweep_grids = np.array(list(sweepvals))
    else:
        sweep_grids = np.meshgrid(*list(sweepvals))

    n_passes = sweep_grids[0].size
    sweep_vects = {var: grid.reshape(n_passes)
                   for (var, grid) in zip(sweepvars, sweep_grids)}

    if verbosity > 0:
        tic = time()

    self.program = []
    last_error = None
    for i in range(n_passes):
        constants.update({var: sweep_vect[i]
                          for (var, sweep_vect) in sweep_vects.items()})
        if linked:
            evaluate_linked(constants, linked)
        program, solvefn = genfunction(self, constants, **kwargs)
        program.model = None  # so it doesn't try to debug
        self.program.append(program)  # NOTE: SIDE EFFECTS
        if i == 0 and verbosity > 0:  # wait for successful program gen
            # TODO: use full string when minimum lineage is set automatically
            sweepvarsstr = ", ".join([str(var)
                                      for var, val in zip(sweepvars, sweepvals)
                                      if not np.isnan(val).all()])
            print(f"Sweeping {sweepvarsstr} with {n_passes} solves:")
        try:
            if verbosity > 1:
                print(f"\nSolve {i}:")
            result = solvefn(solver, verbosity=verbosity-1, **kwargs)
            if kwargs.get("process_result", True):
                self.process_result(result)
            solution.append(result)
            if verbosity == 1:
                print(".", end="", flush=True)
        except Infeasible as e:
            last_error = e
            if not skipsweepfailures:
                raise RuntimeWarning(
                    f"Solve {i} was infeasible; progress saved to m.program."
                    " To continue sweeping after failures, solve with"
                    " skipsweepfailures=True.") from e
            if verbosity > 1:
                print(f"Solve {i} was {e.__class__.__name__}.")
            if verbosity == 1:
                print("!", end="", flush=True)
    if not solution:
        raise RuntimeWarning("All solves were infeasible.") from last_error
    if verbosity == 1:
        print()

    solution["sweepvariables"] = KeyDict()
    ksweep = KeyDict(sweep)
    for var, val in list(solution["constants"].items()):
        if var in ksweep:
            solution["sweepvariables"][var] = val
            del solution["constants"][var]
        elif linked:  # if any variables are linked, we check all of them
            if hasattr(val[0], "shape"):
                differences = ((l != val[0]).any() for l in val[1:])
            else:
                differences = (l != val[0] for l in val[1:])
            if not any(differences):
                solution["constants"][var] = [val[0]]
        else:
            solution["constants"][var] = [val[0]]

    if verbosity > 0:
        soltime = time() - tic
        print(f"Sweeping took {soltime:.3g} seconds.")

"Scripts for generating, solving and sweeping programs"
from time import time
import numpy as np
from ..nomials import parse_subs
from ..solution_array import SolutionArray
from ..keydict import KeyDict

# try:
#     from ipyparallel import Client
#     CLIENT = Client(timeout=0.01)
#     assert len(CLIENT) > 0  # pylint:disable=len-as-condition
#     POOL = CLIENT[:]
#     POOL.use_dill()
#     print("Using parallel execution of sweeps on %s clients" % len(CLIENT))
# except (ImportError, IOError, AssertionError):
POOL = None  # TODO: fix or remove the above


def _progify_fctry(program, return_attr=None):
    "Generates function that returns a program() and optionally an attribute."
    def programify(self, constants=None, **kwargs):
        """Return program version of self

        Arguments
        ---------
        program: NomialData
            Class to return, e.g. GeometricProgram or SequentialGeometricProgram
        return_attr: string
            attribute to return in addition to the program
        """
        if not constants:
            constants, _, linked = parse_subs(self.varkeys, self.substitutions)
            if linked:
                kdc = KeyDict(constants)
                constants.update({v: f(kdc) for v, f in linked.items()})
        prog = program(self.cost, self, constants, **kwargs)
        if return_attr:
            return prog, getattr(prog, return_attr)
        return prog
    return programify


def _solve_fctry(genfunction):
    "Returns function for making/solving/sweeping a program."
    def solvefn(self, solver=None, verbosity=1, skipsweepfailures=False,
                **kwargs):
        """Forms a mathematical program and attempts to solve it.

         Arguments
         ---------
         solver : string or function (optional)
             If None, uses the default solver found in installation.
         verbosity : int (optional)
             If greater than 0 prints runtime messages.
             Is decremented by one and then passed to programs.
         skipsweepfailures : bool (optional)
             If True, when a solve errors during a sweep, skip it.
         **kwargs : Passed to solver

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

        # NOTE: SIDE EFFECTS: self.program is set below
        if sweep:
            run_sweep(genfunction, self, solution, skipsweepfailures,
                      constants, sweep, linked,
                      solver, verbosity, **kwargs)
        else:
            self.program, progsolve = genfunction(self)
            result = progsolve(solver, verbosity, **kwargs)
            solution.append(result)
        solution.program = self.program
        solution.to_arrays()
        if self.cost.units:
            solution["cost"] = solution["cost"] * self.cost.units
        self.solution = solution  # NOTE: SIDE EFFECTS
        # TODO: run process_result here, seperately for each i in a sweep?
        return solution
    return solvefn


# pylint: disable=too-many-locals,too-many-arguments
def run_sweep(genfunction, self, solution, skipsweepfailures,
              constants, sweep, linked,
              solver, verbosity, **kwargs):
    "Runs through a sweep."
    if len(sweep) == 1:
        sweep_grids = np.array(list(sweep.values()))
    else:
        sweep_grids = np.meshgrid(*list(sweep.values()))

    N_passes = sweep_grids[0].size
    sweep_vects = {var: grid.reshape(N_passes)
                   for (var, grid) in zip(sweep, sweep_grids)}

    if verbosity > 0:
        print("Solving over %i passes." % N_passes)
        tic = time()

    def solve_pass(i):
        "Solves one pass of a sweep."
        this_pass = {var: sweep_vect[i]
                     for (var, sweep_vect) in sweep_vects.items()}
        constants.update(this_pass)
        if linked:
            kdc = KeyDict(constants)
            constants.update({v: f(kdc) for v, f in linked.items()})
        program, solvefn = genfunction(self, constants)
        try:
            result = solvefn(solver, verbosity-1, **kwargs)
            # add localmodel here
            return program, result
        except (RuntimeWarning, ValueError):
            return program, None

    mapfn = POOL.map_sync if POOL else map

    self.program = []
    for program, result in mapfn(solve_pass, range(N_passes)):
        self.program.append(program)  # NOTE: SIDE EFFECTS
        if result:  # solve succeeded
            solution.append(result)
        elif not skipsweepfailures:
            raise RuntimeWarning("solve failed during sweep; program"
                                 " has been saved to m.program[-1]."
                                 " To ignore such failures, solve with"
                                 " skipsweepfailures=True.")

    if not solution:
        raise RuntimeWarning("no sweeps solved successfully.")

    solution["sweepvariables"] = KeyDict()
    ksweep = KeyDict(sweep)
    delvars = set()
    for var, val in solution["constants"].items():
        if var in ksweep:
            solution["sweepvariables"][var] = val
            delvars.add(var)
        else:
            solution["constants"][var] = [val[0]]
    for var in delvars:
        del solution["constants"][var]

    if verbosity > 0:
        soltime = time() - tic
        print("Sweeping took %.3g seconds." % (soltime,))

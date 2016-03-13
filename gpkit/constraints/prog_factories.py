"Scripts for generating, solving and sweeping programs"
from time import time
import numpy as np
from ..nomials.substitution import parse_subs
from ..solution_array import SolutionArray
from ..keydict import KeyDict
from ..varkey import VarKey

try:
    from ipyparallel import Client
    CLIENT = Client(timeout=0.01)
    assert len(CLIENT) > 0
    POOL = CLIENT[:]
    POOL.use_dill()
    print("Using parallel execution of sweeps on %s clients" % len(CLIENT))
except (ImportError, IOError, AssertionError):
    POOL = None


def _progify_fctry(program, return_attr=None):
    "Generates function that returns a program() and optionally an attribute."
    def programify(self, verbosity=1, substitutions=None):
        """Return program version of self

        Arguments
        ---------
        program: NomialData
            Class to return, e.g. GeometricProgram or SignomialProgram
        return_attr: string
            attribute to return in addition to the program
        """
        if not substitutions:
            substitutions = self.substitutions
        prog = program(self.cost, self, substitutions, verbosity)
        if return_attr:
            return prog, getattr(prog, return_attr)
        else:
            return prog
    return programify


def _solve_fctry(genfunction):
    "Returns function for making/solving/sweeping a program."
    def solvefn(self, solver=None, verbosity=2, skipsweepfailures=False,
                *args, **kwargs):
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
         *args, **kwargs : Passed to solver

         Returns
         -------
         sol : SolutionArray
             See the SolutionArray documentation for details.

         Raises
         ------
         ValueError if the program is invalid.
         RuntimeWarning if an error occurs in solving or parsing the solution.
         """
        constants, sweep, linkedsweep = parse_subs(self.varkeys,
                                                   self.substitutions)
        solution = SolutionArray()

        if not sweep:
            # NOTE: SIDE EFFECTS IN LINE BELOW
            self.program, solvefn = genfunction(self, verbosity-1)
            result = solvefn(solver, verbosity-1, *args, **kwargs)
            # add localmodel here
            solution.append(result)
        else:
            if len(sweep) == 1:
                sweep_grids = np.array(list(sweep.values()))
            else:
                sweep_grids = np.meshgrid(*list(sweep.values()))

            N_passes = sweep_grids[0].size
            sweep_vects = {var: grid.reshape(N_passes)
                           for (var, grid) in zip(sweep, sweep_grids)}

            if verbosity > 1:
                print("Solving over %i passes." % N_passes)
                tic = time()

            def solve_pass(i):
                this_pass = {var: sweep_vect[i]
                             for (var, sweep_vect) in sweep_vects.items()}
                linked = {var: fn(*[this_pass[VarKey(v)]
                                    for v in var.descr["args"]])
                          for var, fn in linkedsweep.items()}
                this_pass.update(linked)
                constants.update(this_pass)
                program, solvefn = genfunction(self, verbosity-2, constants)
                try:
                    result = solvefn(solver, verbosity-2, *args, **kwargs)
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

            solution["sweepvariables"] = KeyDict()
            ksweep, klinkedsweep = KeyDict(sweep), KeyDict(linkedsweep)
            for var, val in solution["constants"].items():
                if var in ksweep or var in klinkedsweep:
                    solution["sweepvariables"][var] = val
                    del solution["constants"][var]
                else:
                    solution["constants"][var] = [val[0]]
            if not solution["constants"]:
                del solution["constants"]

            if verbosity > 1:
                soltime = time() - tic
                print("Sweeping took %.3g seconds." % (soltime,))
        solution.program = self.program
        solution.to_united_array(unitless_keys=["sensitivities"], united=True)
        # solution["localmodel"] = NomialArray(solution["localmodel"])
        self.solution = solution  # NOTE: SIDE EFFECTS
        if verbosity > 0:
            print(solution.table())
        return solution
    return solvefn

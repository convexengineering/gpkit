"implements SolutionEnsemble class"
import pickle
import numpy as np
from gpkit.keydict import KeyDict
from gpkit.varkey import VarKey

def varsort(diff):
    "Sort function for variables"
    var, *_ = diff
    return var.str_without({"hiddenlineage"})


def vardescr(var):
    "Returns a string fully describing a variable"
    return f"{var.label} ({var})"

class OpenedSolutionEnsemble:
    "Helper class for use with `with` to handle opening/closing an ensemble"
    def __init__(self, filename="solensemble.pkl"):
        self.filename = filename
        try:
            self.solensemble = SolutionEnsemble.load(filename)
        except (EOFError, FileNotFoundError):
            self.solensemble = SolutionEnsemble()

    def __enter__(self):
        return self.solensemble

    def __exit__(self, type_, val, traceback):
        self.solensemble.save(self.filename)

class SolutionEnsemble:
    """An ensemble of solutions.

    Attributes:
      "solutions" : all solutions, keyed by modified variables
      "labels" : solution labels, keyed by modified variables

    SolutionEnsemble[varstr]  : will return the relevant varkey

    """

    def __str__(self):
        nmods = len(self.solutions) - 1
        out = ("Solution ensemble with a baseline and"
               f"{nmods} modified solutions:")
        for differences in self.solutions:
            if differences:
                out += "\n    " + self.labels[differences]
        return out

    def __init__(self):
        self.baseline = None
        self.solutions = {}
        self.labels = {}

    def save(self, filename="solensemble.pkl", **pickleargs):
        "Pickle a file and then compress it into a file with extension."
        with open(filename, "wb") as f:
            pickle.dump(self, f, **pickleargs)

    @staticmethod
    def load(filename):
        "Loads a SolutionEnsemble"
        return pickle.load(open(filename, "rb"))

    def __getitem__(self, var):
        nameref = self.baseline["variables"]
        k, _ = nameref.parse_and_index(var)
        if isinstance(k, str):
            kstr = k
        else:
            kstr = k.str_without({"lineage", "idx"})
            if k.lineage:
                kstr = k.lineagestr() + "." + kstr
        keys = nameref.keymap[kstr]
        if len(keys) != 1:
            raise KeyError(var)
        basevar, = keys
        return basevar

    def filter(self, *requirements):
        "Filters by requirements, returning another solution ensemble"
        candidates = set(self.solutions)
        for requirement in requirements:
            if (isinstance(requirement, str)
                    or not hasattr(requirement, "__len__")):
                requirement = [requirement]
            subreqs = []
            for subreq in requirement:
                try:
                    subreqs.append(self[subreq])
                except (AttributeError, KeyError):
                    subreqs.append(subreq)
            for candidate in set(candidates):
                found_requirement = False
                for difference in candidate:
                    if all(subreq in difference for subreq in subreqs):
                        found_requirement = True
                        break
                if not found_requirement:
                    candidates.remove(candidate)
        se = SolutionEnsemble()
        se.append(self.baseline)
        for candidate in candidates:
            se.append(self.solutions[candidate], verbosity=0)
        return se

    def get_solutions(self, *requirements):
        "Filters by requirements, returning a list of solutions."
        return [sol
                for diff, sol in self.filter(*requirements).solutions.items()
                if diff]

    def append(self, solution, verbosity=1):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        "Appends solution to the Ensemble"
        solution.set_necessarylineage()
        for var in solution["variables"]:
            var.descr.pop("vecfn", None)
            var.descr.pop("evalfn", None)
        if self.baseline is None:
            if "sweepvariables" in solution:
                raise ValueError("baseline solution cannot be a sweep")
            self.baseline = self.solutions[()] = solution
            self.labels[()] = "Baseline Solution"
            return

        solconstraintstr, baseconstraintstr = (
            sol.modelstr[sol.modelstr.find("Constraints"):]
            for sol in [solution, self.baseline])
        if solconstraintstr != baseconstraintstr:
            raise ValueError("the new model's constraints are not identical"
                             " to the base model's constraints."
                             " (Use .baseline.diff(sol) to compare.)")

        solution.pop("warnings", None)
        solution.pop("freevariables", None)
        solution["sensitivities"].pop("constants", None)
        for subd, value in solution.items():
            if isinstance(value, KeyDict):
                solution[subd] = KeyDict()
                for oldkey, val in value.items():
                    solution[subd][self[oldkey]] = val
        for subd, value in solution["sensitivities"].items():
            if subd == "constraints":
                solution["sensitivities"][subd] = {}
                cstrs = {str(c): c
                         for c in self.baseline["sensitivities"][subd]}
                for oldkey, val in value.items():
                    if np.abs(val).max() < 1e-2:
                        if hasattr(val, "shape"):
                            val = np.zeros(val.shape, dtype=np.bool_)
                        else:
                            val = 0
                    elif hasattr(val, "shape"):
                        val = np.array(val, dtype=np.float16)
                    solution["sensitivities"][subd][cstrs[str(oldkey)]] = val
            elif isinstance(value, KeyDict):
                solution["sensitivities"][subd] = KeyDict()
                for oldkey, val in value.items():
                    if np.abs(val).max() < 1e-2:
                        if hasattr(val, "shape"):
                            val = np.zeros(val.shape, dtype=np.bool_)
                        else:
                            val = 0
                    elif hasattr(val, "shape"):
                        val = np.array(val, dtype=np.float16)
                    solution["sensitivities"][subd][self[oldkey]] = val

        differences = []
        labels = []
        solcostfun = solution["cost function"]
        if len(solution) > 1:
            solcostfun = solcostfun[0]
        solcoststr = solcostfun.str_without({"units"})
        basecoststr = self.baseline["cost function"].str_without({"units"})
        if basecoststr != solcoststr:
            differences.append(("cost", solcoststr))
            labels.append(f"Cost function set to {solcoststr}")

        freedvars = set()
        setvars = set()
        def check_var(var,):
            fixed_in_baseline = var in self.baseline["constants"]
            fixed_in_solution = var in solution["constants"]
            bval = self.baseline["variables"][var]
            if fixed_in_solution:
                sval = solution["constants"][var]
            else:
                sval = solution["variables"][var]
            if fixed_in_solution and getattr(sval, "shape", None):
                pass  # calculated constant that depends on a sweep variable
            elif fixed_in_solution and sval != bval:
                setvars.add((var, sval))  # whether free or fixed before
            elif not fixed_in_solution and fixed_in_baseline:
                if var not in solution["sweepvariables"]:
                    freedvars.add((var,))

        for var in self.baseline["variables"]:
            if var not in solution["variables"]:
                print("Variable", var, "removed (relative to baseline)")
                continue
            if not var.shape:
                check_var(var)
            else:
                it = np.nditer(np.empty(var.shape), flags=["multi_index"])
                while not it.finished:
                    check_var(VarKey(idx=it.multi_index, **var.descr))
                    it.iternext()

        for freedvar, in sorted(freedvars, key=varsort):
            differences.append((freedvar, "freed"))
            labels.append(vardescr(freedvar) + " freed")
        for setvar, setval in sorted(setvars, key=varsort):
            differences.append((setvar, setval))
            ustr = setvar.unitstr(into=' %s')
            labels.append(vardescr(setvar) + f" set to {setval:.5g}" + ustr)
        if "sweepvariables" in solution:
            for var, vals in sorted(solution["sweepvariables"].items(),
                                    key=varsort):
                var = self[var]
                if var.shape:
                    it = np.nditer(np.empty(var.shape), flags=["multi_index"])
                    while not it.finished:
                        valsi = vals[(...,)+it.multi_index]
                        if not np.isnan(valsi).any():
                            idxvar = VarKey(idx=it.multi_index, **var.descr)
                            differences.append((idxvar, "sweep",
                                                (min(valsi), max(valsi))))
                            labels.append(vardescr(idxvar) + " swept from"
                                          + f" {min(valsi):.5g} to"
                                          + f" {max(valsi):.5g}"
                                          + idxvar.unitstr(into=' %s'))
                        it.iternext()
                else:
                    differences.append((var, "sweep", (min(vals), max(vals))))
                    labels.append(vardescr(var) + " swept from"
                                  + f" {min(vals):.5g} to"
                                  + f" {max(vals):.5g}"
                                  + var.unitstr(into=' %s'))
        difference = tuple(differences)
        label = ", ".join(labels)
        if verbosity > 0:
            if difference in self.solutions:
                if not difference:
                    print("The baseline in this ensemble cannot be replaced.")
                else:
                    print(label + " will be replaced in the ensemble.")
            else:
                print(label + " added to the ensemble.")

        self.solutions[difference] = solution
        self.labels[difference] = label

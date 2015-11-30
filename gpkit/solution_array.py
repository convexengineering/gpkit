import numpy as np

from collections import Iterable
from functools import reduce as functools_reduce
from operator import mul

from .posyarray import PosyArray
from .nomials import Monomial
from .varkey import VarKey
from .small_classes import Strings, Quantity
from .small_classes import DictOfLists
from .small_scripts import unitstr
from .small_scripts import mag


class SolutionArray(DictOfLists):
    """A dictionary (of dictionaries) of lists, with convenience methods.

    Items
    -----
    cost : array
    variables: dict of arrays
    sensitivities: dict containing:
        monomials : array
        posynomials : array
        variables: dict of arrays
    localmodels : PosyArray
        Local power-law fits (small sensitivities are cut off)

    Example
    -------
    >>> import gpkit
    >>> import numpy as np
    >>> x = gpkit.Variable("x")
    >>> x_min = gpkit.Variable("x_{min}", 2)
    >>> sol = gpkit.Model(x, [x >= x_min]).solve(verbosity=0)
    >>>
    >>> # VALUES
    >>> values = [sol(x), sol.subinto(x), sol["variables"]["x"]]
    >>> assert all(np.array(values) == 2)
    >>>
    >>> # SENSITIVITIES
    >>> senss = [sol.sens(x_min), sol.sens(x_min)]
    >>> senss.append(sol["sensitivities"]["variables"]["x_{min}"])
    >>> assert all(np.array(senss) == 1)

    """

    def __len__(self):
        try:
            return len(self["cost"])
        except TypeError:
            return 1

    def getvars(self, *args):
        out = [self["variables"][arg] for arg in args]
        return out[0] if len(out) == 1 else out

    def __call__(self, p):
        return mag(self.subinto(p).c)

    def subinto(self, p):
        "Returns PosyArray of each solution substituted into p."
        if p in self["variables"]:
            return PosyArray(self["variables"][p])
        elif len(self) > 1:
            return PosyArray([self.atindex(i).subinto(p)
                             for i in range(len(self))])
        else:
            return p.sub(self["variables"])

    def sens(self, p):
        """Returns array of each solution's sensitivity substituted into p

        Note: this does not return monomial sensitivities if you pass it a
        signomial; it returns each variable's sensitivity substituted in for it
        in that signomial.

        Returns scalar, unitless values.
        """
        if p in self["variables"]["sensitivities"]:
            return PosyArray(self["variables"]["sensitivities"][p])
        elif len(self) > 1:
            return PosyArray([self.atindex(i).subinto(p)
                             for i in range(len(self))])
        else:
            subbed = p.sub(self["variables"]["sensitivities"],
                           require_positive=False)
            assert isinstance(subbed, Monomial)
            assert not subbed.exp
            return mag(subbed.c)

    def table(self, tables=["cost", "freevariables", "sweepvariables",
                            "constants", "sensitivities"], fixedcols=True,
                            included_models=None, excluded_models=None):
        if isinstance(tables, Strings):
            tables = [tables]
        strs = []
        if "cost" in tables:
            strs += ["\nCost\n----"]
            if len(self) > 1:
                costs = ["%-8.3g" % c for c in self["cost"][:4]]
                strs += [" [ %s %s ]" % ("  ".join(costs),
                                         "..." if len(self) > 4 else "")]
                cost_units = self.program[0].cost.units
            else:
                strs += [" %-.4g" % self["cost"]]
                cost_units = self.program.cost.units
            strs[-1] += unitstr(cost_units, into=" [%s] ", dimless="")
            strs += [""]
        if in_both_and_truthy_in_second("sweepvariables", tables, self):
            strs += [results_table(self["sweepvariables"],
                                   "Sweep Variables",
                                   fixedcols=fixedcols,
                                   included_models=included_models,
                                   excluded_models=excluded_models)]
        if in_both_and_truthy_in_second("freevariables", tables, self):
            strs += [results_table(self["freevariables"],
                                   "Free Variables",
                                   fixedcols=fixedcols,
                                   included_models=included_models,
                                   excluded_models=excluded_models)]
        if in_both_and_truthy_in_second("constants", tables, self):
            strs += [results_table(self["constants"],
                                   "Constants",
                                   fixedcols=fixedcols,
                                   included_models=included_models,
                                   excluded_models=excluded_models)]
        if in_both_and_truthy_in_second("variables", tables, self):
            strs += [results_table(self["variables"],
                                   "Variables",
                                   fixedcols=fixedcols,
                                   included_models=included_models,
                                   excluded_models=excluded_models)]
        if (in_both_and_truthy_in_second("sensitivities", tables, self)
                and "constants" in self["sensitivities"]
                and self["sensitivities"]["constants"]):
            strs += [results_table(self["sensitivities"]["constants"],
                                   "Sensitivities",
                                   fixedcols=fixedcols,
                                   included_models=included_models,
                                   excluded_models=excluded_models,
                                   minval=1e-2,
                                   printunits=False)]
        return "\n".join(strs)


def in_both_and_truthy_in_second(key, tables, dictionary):
    return bool(key in tables and key in dictionary and dictionary[key])


def results_table(data, title, minval=0, printunits=True, fixedcols=True,
                  varfmt="%s : ", valfmt="%-.4g ", vecfmt="%-8.3g",
                  included_models=None, excluded_models=None):
    """
    Pretty string representation of a dict of VarKeys
    Iterable values are handled specially (partial printing)

    Arguments
    ---------
    data: dict whose keys are VarKey's
        data to represent in table
    title: string
    minval: float
        skip values with all(abs(value)) < minval
    printunits: bool
    fixedcols: bool
        if True, print rhs (val, units, label) in fixed-width cols
    varfmt: string
        format for variable names
    valfmt: string
        format for scalar values
    vecfmt: string
        format for vector values
    """
    lines = []
    decorated = []
    models = set()
    for i, (k, v) in enumerate(data.items()):
        notnan = ~np.isnan([v])
        if np.any(notnan) and np.max(np.abs(np.array([v])[notnan])) >= minval:
            b = isinstance(v, Iterable) and bool(v.shape)
            model = k.descr.get("model", "")
            models.add(model)
            decorated.append((model, b, (varfmt % k.nomstr), i, k, v))
    if included_models:
        included_models = set(included_models)
        included_models.add("")
        models = models.intersection(included_models)
    if excluded_models:
        models = models.difference(excluded_models)
    decorated.sort()
    oldmodel = None
    for model, isvector, varstr, _, var, val in decorated:
        if model not in models:
            continue
        if model != oldmodel and len(models) > 1:
            if oldmodel is not None:
                lines.append(["", "", "", ""])
            if model is not "":
                lines.append([model+" | ", "", "", ""])
            oldmodel = model
        label = var.descr.get('label', '')
        units = unitstr(var, into=" [%s] ", dimless="") if printunits else ""
        if isvector:
            vals = [vecfmt % v for v in val.flatten()[:4]]
            ellipsis = " ..." if len(val) > 4 else ""
            valstr = "[ %s%s ] " % ("  ".join(vals), ellipsis)
        else:
            valstr = valfmt % val
        valstr = valstr.replace("nan", " - ")
        lines.append([varstr, valstr, units, label])
    if lines:
        maxlens = np.max([list(map(len, line)) for line in lines], axis=0)
        if not fixedcols:
            maxlens = [maxlens[0], 0, 0, 0]
        dirs = ['>', '<', '<', '<']
        # check lengths before using zip
        assert len(list(dirs)) == len(list(maxlens))
        fmts = ['{0:%s%s}' % (direc, L) for direc, L in zip(dirs, maxlens)]
    lines = [[fmt.format(s) for fmt, s in zip(fmts, line)] for line in lines]
    lines = [title] + ["-"*len(title)] + [''.join(l) for l in lines] + [""]
    return "\n".join(lines)

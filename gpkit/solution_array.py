import numpy as np

from collections import Iterable

from .posyarray import PosyArray
from .nomials import Monomial
from .small_classes import Strings
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

    Example
    -------
    >>> import gpkit
    >>> import numpy as np
    >>> x = gpkit.Variable("x")
    >>> x_min = gpkit.Variable("x_{min}", 2)
    >>> sol = gpkit.Model(x, [x >= x_min]).solve(verbosity=0)
    >>> # ACCESS VALUES
    >>> values = [sol(x), sol.subinto(x), sol["variables"]["x"]]
    >>> assert all(np.array(values) == 2)
    >>> # ACCESS SENSITIVITIES
    >>> senss = [sol.sens(x_min), sol.senssubinto(x_min)]
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
        if len(out) == 1:
            return out[0]
        else:
            return out

    def __call__(self, p):
        return mag(self.subinto(p).c)

    def subinto(self, p):
        "Returns PosyArray of each solution substituted into p."
        if p in self["variables"]:
            return PosyArray(self["variables"][p])
        if len(self) > 1:
            return PosyArray([p.sub(self.atindex(i)["variables"])
                              for i in range(len(self["cost"]))])
        else:
            return p.sub(self["variables"])

    def sens(self, p):
        return self.senssubinto(p)

    def senssubinto(self, p):
        """Returns array of each solution's sensitivity substituted into p

        Returns only scalar values.
        """
        if len(self) > 1:
            subbeds = [p.sub(self.atindex(i)["sensitivities"]["variables"],
                             require_positive=False) for i in range(len(self))]
            assert not any([subbed.exp for subbed in subbeds])
            return np.array([mag(subbed.c) for subbed in subbeds],
                            np.dtype('float'))
        else:
            subbed = p.sub(self["sensitivities"]["variables"],
                           require_positive=False)
            assert isinstance(subbed, Monomial)
            assert not subbed.exp
            return mag(subbed.c)

    def table(self,
              tables=["cost", "freevariables", "sweepvariables",
                      "constants", "sensitivities"],
              fixedcols=True):
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
        if "freevariables" in tables:
            strs += [results_table(self["freevariables"],
                                   "Free Variables",
                                   fixedcols=fixedcols)]
        if "sweepvariables" in tables and self["sweepvariables"]:
            strs += [results_table(self["sweepvariables"],
                                   "Sweep Variables",
                                   fixedcols=fixedcols)]
        if "constants" in tables and self["constants"]:
            strs += [results_table(self["constants"],
                                   "Constants",
                                   fixedcols=fixedcols)]
        if "variables" in tables:
            strs += [results_table(self["variables"],
                                   "Variables",
                                   fixedcols=fixedcols)]
        if "sensitivities" in tables:
            strs += [results_table(self["sensitivities"]["variables"],
                                   "Constant and swept variable sensitivities",
                                   fixedcols=fixedcols,
                                   minval=1e-2,
                                   printunits=False)]
        return "\n".join(strs)


def results_table(data, title, minval=0, printunits=True, fixedcols=True,
                  varfmt="%s : ", valfmt="%-.4g ", vecfmt="%-8.3g"):
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
    for i, (k, v) in enumerate(data.items()):
        notnan = ~np.isnan([v])
        if np.any(notnan) and np.max(np.abs(np.array([v])[notnan])) >= minval:
            b = isinstance(v, Iterable) and bool(v.shape)
            decorated.append((b, (varfmt % k), i, k, v))
    decorated.sort()
    for isvector, varstr, _, var, val in decorated:
        label = var.descr.get('label', '')
        units = unitstr(var, into=" [%s] ", dimless="") if printunits else ""
        if isvector:
            vals = [vecfmt % v for v in val[:4]]
            ellipsis = " ..." if len(val) > 4 else ""
            valstr = "[ %s%s ] " % ("  ".join(vals), ellipsis)
        else:
            valstr = valfmt % val
        valstr = valstr.replace("nan", " - ")
        lines.append([varstr, valstr, units, label])
    if lines:
        maxlens = np.max([map(len, line) for line in lines], axis=0)
        if not fixedcols:
            maxlens = [maxlens[0], 0, 0, 0]
        dirs = ['>', '<', '<', '<']
        assert len(dirs) == len(maxlens)  # check lengths before using zip
        fmts = ['{0:%s%s}' % (direc, L) for direc, L in zip(dirs, maxlens)]
    lines = [[fmt.format(s) for fmt, s in zip(fmts, line)]
             for line in lines]
    lines = [title] + ["-"*len(title)] + [''.join(l) for l in lines] + [""]
    return "\n".join(lines)

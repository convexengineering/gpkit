"""Defines SolutionArray class"""
from collections import Iterable
import numpy as np
from .nomials import NomialArray, Monomial
from .small_classes import Strings, DictOfLists
from .small_scripts import unitstr, mag


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
    localmodels : NomialArray
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
    program = None
    table_titles = {"cost": "Cost",
                    "sweepvariables": "Sweep Variables",
                    "freevariables": "Free Variables",
                    "constants": "Constants",
                    "variables": "Variables",
                    "sensitivities": "Sensitivities"}

    def __len__(self):
        try:
            return len(self["cost"])
        except TypeError:
            return 1
        except KeyError:
            return 0

    def __call__(self, posy):
        posy_subbed = self.subinto(posy)
        if hasattr(posy_subbed, "exp") and not posy_subbed.exp:
            # it's a constant monomial
            return posy_subbed.c
        elif hasattr(posy_subbed, "c"):
            # it's a posyarray, which'll throw an error if non-constant...
            return posy_subbed.c
        return posy_subbed

    def subinto(self, posy):
        "Returns NomialArray of each solution substituted into posy."
        if posy in self["variables"]:
            return self["variables"][posy]
        elif len(self) > 1:
            return NomialArray([self.atindex(i).subinto(posy)
                                for i in range(len(self))])
        else:
            return posy.sub(self["variables"])

    def sens(self, nomial):
        """Returns array of each solution's sensitivity substituted into nomial

        Note: this does not return monomial sensitivities if you pass it a
        signomial; it returns each variable's sensitivity substituted in for it
        in that signomial.

        Returns scalar, unitless values.
        """
        if nomial in self["variables"]["sensitivities"]:
            return NomialArray(self["variables"]["sensitivities"][nomial])
        elif len(self) > 1:
            return NomialArray([self.atindex(i).subinto(nomial)
                                for i in range(len(self))])
        else:
            subbed = nomial.sub(self["variables"]["sensitivities"],
                                require_positive=False)
            assert isinstance(subbed, Monomial)
            assert not subbed.exp
            return mag(subbed.c)

    def table(self, tables=("cost", "sweepvariables", "freevariables",
                            "constants", "sensitivities"),
              latex=False, **kwargs):
        """A table representation of this SolutionArray

        Arguments
        ---------
        tables: Iterable
            Which to print of ("cost", "sweepvariables", "freevariables",
                               "constants", "sensitivities")
        fixedcols: If true, print vectors in fixed-width format
        latex: int
            If > 0, return latex format (options 1-3); otherwise plain text
        included_models: Iterable of strings
            If specified, the models (by name) to include
        excluded_models: Iterable of strings
            If specified, model names to exclude

        Returns
        -------
        str
        """
        if isinstance(tables, Strings):
            tables = [tables]
        strs = []
        for table in tables:
            subdict = self.get(table, None)
            table_title = self.table_titles[table]
            if table == "cost":
                # pylint: disable=unsubscriptable-object
                if latex:
                    # TODO should probably print a small latex cost table here
                    continue
                strs += ["\n%s\n----" % table_title]
                if len(self) > 1:
                    costs = ["%-8.3g" % c for c in subdict[:4]]
                    strs += [" [ %s %s ]" % ("  ".join(costs),
                                             "..." if len(self) > 4 else "")]
                    cost_units = self.program[0].cost.units
                else:
                    strs += [" %-.4g" % subdict]
                    if hasattr(self.program, "cost"):
                        cost_units = self.program.cost.units
                    else:
                        # we're in a skipsweepfailures that only solved once
                        cost_units = self.program[0].cost.units
                strs[-1] += unitstr(cost_units, into=" [%s] ", dimless="")
                strs += [""]
            elif not subdict:
                continue
            elif table == "sensitivities":
                strs += results_table(subdict["constants"], table_title,
                                      minval=1e-2,
                                      sortbyvals=True,
                                      printunits=False,
                                      latex=latex,
                                      **kwargs)
            else:
                strs += results_table(subdict, table_title,
                                      latex=latex, **kwargs)
        if latex:
            preamble = "\n".join(("% \\documentclass[12pt]{article}",
                                  "% \\usepackage{booktabs}",
                                  "% \\usepackage{longtable}",
                                  "% \\usepackage{amsmath}",
                                  "% \\begin{document}\n"))
            strs = [preamble] + strs + ["% \\end{document}"]
        return "\n".join(strs)


# pylint: disable=too-many-statements,too-many-arguments
# pylint: disable=too-many-branches,too-many-locals
def results_table(data, title, minval=0, printunits=True, fixedcols=True,
                  varfmt="%s : ", valfmt="%-.4g ", vecfmt="%-8.3g",
                  included_models=None, excluded_models=None, latex=False,
                  sortbyvals=False):
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
    latex: int
        If > 0, return latex format (options 1-3); otherwise plain text
    included_models: Iterable of strings
        If specified, the models (by name) to include
    excluded_models: Iterable of strings
        If specified, model names to exclude
    sortbyvals : boolean
        If true, rows are sorted by their average value instead of by name.
    """
    lines = []
    decorated = []
    models = set()
    for i, (k, v) in enumerate(data.items()):
        v_ = mag(v)
        notnan = ~np.isnan([v_])
        if np.any(notnan) and np.max(np.abs(np.array([v_])[notnan])) >= minval:
            b = isinstance(v, Iterable) and bool(v.shape)
            model = ", ".join(k.descr.get("models", ""))
            models.add(model)
            s = k.str_without("models")
            if not sortbyvals:
                decorated.append((model, b, (varfmt % s), i, k, v))
            else:
                decorated.append((model, np.mean(v), b, (varfmt % s), i, k, v))
    if included_models:
        included_models = set(included_models)
        included_models.add("")
        models = models.intersection(included_models)
    if excluded_models:
        models = models.difference(excluded_models)
    decorated.sort(reverse=sortbyvals)
    oldmodel = None
    for varlist in decorated:
        if not sortbyvals:
            model, isvector, varstr, _, var, val = varlist
        else:
            model, _, isvector, varstr, _, var, val = varlist
        if model not in models:
            continue
        if model != oldmodel and len(models) > 1:
            if oldmodel is not None:
                lines.append(["", "", "", ""])
            if model is not "":
                if not latex:
                    lines.append([model+" | ", "", "", ""])
                else:
                    lines.append([r"\multicolumn{3}{l}{\textbf{" +
                                  model + r"}} \\"])
            oldmodel = model
        label = var.descr.get('label', '')
        units = unitstr(var, into=" [%s] ", dimless="") if printunits else ""
        if isvector:
            vals = [vecfmt % v for v in mag(val).flatten()[:4]]
            ellipsis = " ..." if len(val) > 4 else ""
            valstr = "[ %s%s ] " % ("  ".join(vals), ellipsis)
        else:
            valstr = valfmt % mag(val)
        valstr = valstr.replace("nan", " - ")
        if not latex:
            lines.append([varstr, valstr, units, label])
        else:
            varstr = "$%s$" % varstr.replace(" : ", "")
            if latex == 1:  # normal results table
                lines.append([varstr, valstr, "$%s$" % var.unitstr(), label])
                coltitles = [title, "Value", "Units", "Description"]
            elif latex == 2:  # no values
                lines.append([varstr, "$%s$" % var.unitstr(), label])
                coltitles = [title, "Units", "Description"]
            elif latex == 3:  # no description
                lines.append([varstr, valstr, "$%s$" % var.unitstr()])
                coltitles = [title, "Value", "Units"]
            else:
                raise ValueError("Unexpected latex option, %s." % latex)
    if not latex:
        if lines:
            maxlens = np.max([list(map(len, line)) for line in lines], axis=0)
            if not fixedcols:
                maxlens = [maxlens[0], 0, 0, 0]
            dirs = ['>', '<', '<', '<']
            # check lengths before using zip
            assert len(list(dirs)) == len(list(maxlens))
            fmts = ['{0:%s%s}' % (direc, L) for direc, L in zip(dirs, maxlens)]
        lines = [[fmt.format(s) for fmt, s in zip(fmts, line)]
                 for line in lines]
        lines = [title] + ["-"*len(title)] + [''.join(l) for l in lines] + [""]
    else:
        colfmt = {1: "llcl", 2: "lcl", 3: "llc"}
        lines = (["\n".join(["{\\footnotesize",
                             "\\begin{longtable}{%s}" % colfmt[latex],
                             "\\toprule",
                             " & ".join(coltitles) + " \\\\ \\midrule"])] +
                 [" & ".join(l) + " \\\\" for l in lines] +
                 ["\n".join(["\\bottomrule", "\\end{longtable}}", ""])])
    return lines

"""Defines SolutionArray class"""
from collections import Iterable
import numpy as np
from .nomials import NomialArray
from .small_classes import DictOfLists
from .small_scripts import unitstr, mag


def senss_table(data, showvars=(), title="Sensitivities", **kwargs):
    "Returns sensitivity table lines"
    if "constants" in data.get("sensitivities", {}):
        data = data["sensitivities"]["constants"]
    if showvars:
        data = {k: data[k] for k in showvars if k in data}
    return results_table(data, title, sortbyvals=True,
                         valfmt="%+-.2g ", vecfmt="%+-8.2g",
                         printunits=False, **kwargs)


def topsenss_table(data, showvars, nvars=5, **kwargs):
    "Returns top sensitivity table lines"
    data, filtered = topsenss_filter(data, showvars, nvars)
    title = "Most Sensitive" if not filtered else "Next Largest Sensitivities"
    return senss_table(data, title=title, **kwargs)


def topsenss_filter(data, showvars, nvars=5):
    "Filters sensitivities down to top N vars"
    if "constants" in data.get("sensitivities", {}):
        data = data["sensitivities"]["constants"]
    mean_abs_senss = {k: np.abs(s).mean() for k, s in data.items()
                      if not np.isnan(s).any()}
    topk = [k for k, _ in sorted(mean_abs_senss.items(), key=lambda l: l[1])]
    filter_already_shown = showvars.intersection(topk)
    for k in filter_already_shown:
        topk.remove(k)
        if nvars > 3:  # always show at least 3
            nvars -= 1
    return {k: data[k] for k in topk[-nvars:]}, filter_already_shown


def insenss_table(data, _, maxval=0.1, **kwargs):
    "Returns insensitivity table lines"
    if "constants" in data.get("sensitivities", {}):
        data = data["sensitivities"]["constants"]
    data = {k: s for k, s in data.items() if np.mean(np.abs(s)) < maxval}
    return senss_table(data, title="Insensitive Fixed Variables", **kwargs)

TABLEFNS = {"sensitivities": senss_table,
            "topsensitivities": topsenss_table,
            "insensitivities": insenss_table,
           }


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
    table_titles = {"sweepvariables": "Sweep Variables",
                    "freevariables": "Free Variables",
                    "constants": "Constants",
                    "variables": "Variables"}

    def __len__(self):
        try:
            return len(self["cost"])
        except TypeError:
            return 1
        except KeyError:
            return 0

    def __call__(self, posy):
        posy_subbed = self.subinto(posy)
        return getattr(posy_subbed, "c", posy_subbed)

    def subinto(self, posy):
        "Returns NomialArray of each solution substituted into posy."
        if posy in self["variables"]:
            return self["variables"][posy]
        elif not hasattr(posy, "sub"):
            raise ValueError("no variable '%s' found in the solution" % posy)
        elif len(self) > 1:
            return NomialArray([self.atindex(i).subinto(posy)
                                for i in range(len(self))])
        else:
            return posy.sub(self["variables"])

    def _parse_showvars(self, showvars):
        showvars_out = set()
        if showvars:
            for k in showvars:
                k, _ = self["variables"].parse_and_index(k)
                keys = self["variables"].keymap[k]
                showvars_out.update(keys)
        return showvars_out

    def summary(self, showvars=(), ntopsenss=5):
        "Print summary table, showing top sensitivities and no constants"
        showvars = self._parse_showvars(showvars)
        out = self.table(showvars, ["cost", "sweepvariables", "freevariables"])
        constants_in_showvars = showvars.intersection(self["constants"])
        senss_tables = []
        if len(self["constants"]) < ntopsenss+2 or constants_in_showvars:
            senss_tables.append("sensitivities")
        if len(self["constants"]) >= ntopsenss+2:
            senss_tables.append("topsensitivities")
        senss_str = self.table(showvars, senss_tables, nvars=ntopsenss)
        if senss_str:
            out += "\n" + senss_str
        return out

    def table(self, showvars=(),
              tables=("cost", "sweepvariables", "freevariables",
                      "constants", "sensitivities"), **kwargs):
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
        showvars = self._parse_showvars(showvars)
        strs = []
        for table in tables:
            if table == "cost":
                cost = self["cost"]
                # pylint: disable=unsubscriptable-object
                if kwargs.get("latex", None):
                    # TODO should probably print a small latex cost table here
                    continue
                strs += ["\n%s\n----" % "Cost"]
                if len(self) > 1:
                    costs = ["%-8.3g" % c for c in mag(cost[:4])]
                    strs += [" [ %s %s ]" % ("  ".join(costs),
                                             "..." if len(self) > 4 else "")]
                else:
                    strs += [" %-.4g" % mag(cost)]
                strs[-1] += unitstr(cost, into=" [%s] ", dimless="")
                strs += [""]
            elif table in TABLEFNS:
                strs += TABLEFNS[table](self, showvars, **kwargs)
            elif table in self:
                data = self[table]
                if showvars:
                    data = {k: data[k] for k in showvars if k in data}
                strs += results_table(data, self.table_titles[table], **kwargs)
        if kwargs.get("latex", None):
            preamble = "\n".join(("% \\documentclass[12pt]{article}",
                                  "% \\usepackage{booktabs}",
                                  "% \\usepackage{longtable}",
                                  "% \\usepackage{amsmath}",
                                  "% \\begin{document}\n"))
            strs = [preamble] + strs + ["% \\end{document}"]
        return "\n".join(strs)

    def plot(self, posys=None, axes=None):
        "Plots a sweep for each posy"
        if len(self["sweepvariables"]) != 1:
            print "SolutionArray.plot only supports 1-dimensional sweeps"
        if not hasattr(posys, "__len__"):
            posys = [posys]
        for i, posy in enumerate(posys):
            if posy in [None, "cost"]:
                posys[i] = self.program[0].cost   # pylint: disable=unsubscriptable-object
        import matplotlib.pyplot as plt
        from .interactive.plot_sweep import assign_axes
        from . import GPBLU
        (swept, x), = self["sweepvariables"].items()
        posys, axes = assign_axes(swept, posys, axes)
        for posy, ax in zip(posys, axes):
            y = self(posy) if posy not in [None, "cost"] else self["cost"]
            ax.plot(x, y, color=GPBLU)
        if len(axes) == 1:
            axes, = axes
        return plt.gcf(), axes


# pylint: disable=too-many-statements,too-many-arguments
# pylint: disable=too-many-branches,too-many-locals
def results_table(data, title, minval=0, printunits=True, fixedcols=True,
                  varfmt="%s : ", valfmt="%-.4g ", vecfmt="%-8.3g",
                  included_models=None, excluded_models=None, latex=False,
                  sortbyvals=False, **_):  # **_ catches unused tablefn args
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
    if not data:
        return []
    from . import units
    if not units:
        # disable units printing
        printunits = False
    lines = []
    decorated = []
    models = set()
    for i, (k, v) in enumerate(data.items()):
        v_ = mag(v)
        notnan = ~np.isnan([v_])
        if np.any(notnan) and np.sum(np.abs(np.array([v_])[notnan])) >= minval:
            b = isinstance(v, Iterable) and bool(v.shape)
            kmodels = k.descr.get("models", [])
            kmodelnums = k.descr.get("modelnums", [])
            model = "/".join([kstr + (".%i" % knum if knum != 0 else "")
                              for kstr, knum in zip(kmodels, kmodelnums)
                              if kstr])
            models.add(model)
            s = k.str_without("models")
            if not sortbyvals:
                decorated.append((model, b, (varfmt % s), i, k, v))
            else:  # for consistent sorting, add small offset to negative vals
                val = np.mean(np.abs(v)) - (1e-9 if np.mean(v) < 0 else 0)
                decorated.append((model, -val, b, (varfmt % s), i, k, v))
    if included_models:
        included_models = set(included_models)
        included_models.add("")
        models = models.intersection(included_models)
    if excluded_models:
        models = models.difference(excluded_models)
    decorated.sort()
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
                    lines.append([("modelname",), model, "", ""])
                else:
                    lines.append([r"\multicolumn{3}{l}{\textbf{" +
                                  model + r"}} \\"])
            oldmodel = model
        label = var.descr.get('label', '')
        units = var.unitstr() if printunits else ""
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
                lines.append([varstr, valstr, "$%s$" % var.latex_unitstr(),
                              label])
                coltitles = [title, "Value", "Units", "Description"]
            elif latex == 2:  # no values
                lines.append([varstr, "$%s$" % var.latex_unitstr(), label])
                coltitles = [title, "Units", "Description"]
            elif latex == 3:  # no description
                lines.append([varstr, valstr, "$%s$" % var.latex_unitstr()])
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
            fmts = [u'{0:%s%s}' % (direc, L) for direc, L in zip(dirs, maxlens)]
        for i, line in enumerate(lines):
            if line[0] == ("modelname",):
                line = [fmts[0].format(" | "), line[1]]
            else:
                line = [fmt.format(s) for fmt, s in zip(fmts, line)]
            lines[i] = "".join(line).rstrip()
        lines = [title] + ["-"*len(title)] + lines + [""]
    elif lines:
        colfmt = {1: "llcl", 2: "lcl", 3: "llc"}
        lines = (["\n".join(["{\\footnotesize",
                             "\\begin{longtable}{%s}" % colfmt[latex],
                             "\\toprule",
                             " & ".join(coltitles) + " \\\\ \\midrule"])] +
                 [" & ".join(l) + " \\\\" for l in lines] +
                 ["\n".join(["\\bottomrule", "\\end{longtable}}", ""])])
    return lines

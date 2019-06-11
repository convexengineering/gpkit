"""Defines SolutionArray class"""
import re
from collections import Iterable
import cPickle as pickle
import numpy as np
from .nomials import NomialArray
from .small_classes import DictOfLists, Strings
from .small_scripts import mag, isnan
from .repr_conventions import unitstr


CONSTRSPLITPATTERN = re.compile(r"([^*]\*[^*])|( \+ )|( >= )|( <= )|( = )")

VALSTR_REPLACES = [
    ("+nan", " - "),
    ("nan", " - "),
    ("-nan", " - "),
    ("+0 ", " 0 "),
    ("+0.00 ", " 0.00 "),
    ("-0.00 ", " 0.00 "),
    ("+0.0% ", " 0.0  "),
    ("-0.0% ", " 0.0  ")
]


def senss_table(data, showvars=(), title="Sensitivities", **kwargs):
    "Returns sensitivity table lines"
    if "constants" in data.get("sensitivities", {}):
        data = data["sensitivities"]["constants"]
    if showvars:
        data = {k: data[k] for k in showvars if k in data}
    return var_table(data, title, sortbyvals=True,
                     valfmt="%+-.2g  ", vecfmt="%+-8.2g",
                     printunits=False, minval=1e-3, **kwargs)


def topsenss_table(data, showvars, nvars=5, **kwargs):
    "Returns top sensitivity table lines"
    data, filtered = topsenss_filter(data, showvars, nvars)
    title = "Most Sensitive" if not filtered else "Next Largest Sensitivities"
    return senss_table(data, title=title, hidebelowminval=True, **kwargs)


def topsenss_filter(data, showvars, nvars=5):
    "Filters sensitivities down to top N vars"
    if "constants" in data.get("sensitivities", {}):
        data = data["sensitivities"]["constants"]
    mean_abs_senss = {k: np.abs(s).mean() for k, s in data.items()
                      if not isnan(s).any()}
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


def tight_table(self, _, ntightconstrs=5, tight_senss=1e-2, **kwargs):
    "Return constraint tightness lines"
    if not self.model:
        return []
    title = "Tightest Constraints"
    data = [(-float("%+6.2g" % c.relax_sensitivity) + 1e-30*hash(str(c)),
             "%+6.2g" % c.relax_sensitivity, c)
            for c in self.model.flat(constraintsets=False)
            if c.relax_sensitivity >= tight_senss]
    if not data:
        lines = ["No constraints had a sensitivity above %+5.1g."
                 % tight_senss]
    else:
        data = sorted(data)[:ntightconstrs]
        lines = constraint_table(data, **kwargs)
    lines = [title] + ["-"*len(title)] + lines + [""]
    if "sweepvariables" in self:
        lines.insert(1, "(for the last sweep only)")
    return lines


def loose_table(self, _, loose_senss=1e-5, **kwargs):
    "Return constraint tightness lines"
    if not self.model:
        return []
    title = "All Loose Constraints"
    data = [(0, "", c) for c in self.model.flat(constraintsets=False)
            if c.relax_sensitivity <= loose_senss]
    if not data:
        lines = ["No constraints had a sensitivity below %+6.2g."
                 % loose_senss]
    else:
        lines = constraint_table(data, **kwargs)
    return [title] + ["-"*len(title)] + lines + [""]


# pylint: disable=too-many-branches,too-many-locals,too-many-statements
def constraint_table(data, sortbymodels=True, showmodels=True, **_):
    "Creates lines for tables where the right side is a constraint."
    models = {}
    decorated = []
    for sortby, openingstr, constraint in data:
        if sortbymodels and getattr(constraint, "naming", None):
            model = "/".join([kstr + (".%i" % knum if knum != 0 else "")
                              for kstr, knum in zip(*constraint.naming)
                              if kstr])
        else:
            model = ""
        if model not in models:
            models[model] = len(models)
        if showmodels:
            constrstr = str(constraint)
            if " at 0x" in constrstr:  # don't print memory addresses
                constrstr = constrstr[:constrstr.find(" at 0x")] + ">"
        else:
            try:
                constrstr = constraint.str_without(["units", "models"])
            except AttributeError:
                constrstr = str(constraint)
        decorated.append((models[model], model, sortby, constrstr, openingstr))
    decorated.sort()
    oldmodel = None
    lines = []
    for varlist in decorated:
        _, model, _, constrstr, openingstr = varlist
        if model not in models:
            continue
        if model != oldmodel and len(models) > 1:
            if oldmodel is not None:
                lines.append(["", ""])
            lines.append([("modelname",), model])
            oldmodel = model
        if model and len(models) == 1:  # fully remove
            constrstr = constrstr.replace("_"+model, "")
        else:  # partially remove
            constrstr = constrstr.replace(model, "")
        minlen, maxlen = 25, 80
        segments = [s for s in CONSTRSPLITPATTERN.split(constrstr) if s]
        splitlines = []
        line = ""
        next_idx = 0
        while next_idx < len(segments):
            segment = segments[next_idx]
            next_idx += 1
            if CONSTRSPLITPATTERN.match(segment) and next_idx < len(segments):
                segments[next_idx] = segment[1:] + segments[next_idx]
                segment = segment[0]
            elif len(line) + len(segment) > maxlen and len(line) > minlen:
                splitlines.append(line)
                line = "  "
            line += segment
            while len(line) > maxlen:
                splitlines.append(line[:maxlen])
                line = "  " + line[maxlen:]
        splitlines.append(line)
        lines += [(openingstr + " : ", splitlines[0])]
        lines += [("", l) for l in splitlines[1:]]

    maxlens = np.max([list(map(len, line)) for line in lines
                      if line[0] != ("modelname",)], axis=0)
    dirs = ['>', '<']
    # check lengths before using zip
    assert len(list(dirs)) == len(list(maxlens))
    fmts = [u'{0:%s%s}' % (direc, L) for direc, L in zip(dirs, maxlens)]
    for i, line in enumerate(lines):
        if line[0] == ("modelname",):
            line = [fmts[0].format(" | "), line[1]]
        else:
            line = [fmt.format(s) for fmt, s in zip(fmts, line)]
        lines[i] = "".join(line).rstrip()
    return lines


def warnings_table(self, _, **kwargs):
    "Makes a table for all warnings in the solution."
    title = "Warnings"
    lines = [title, "="*len(title)]
    if "warnings" not in self or not self["warnings"]:
        return []
    for wtype in self["warnings"]:
        lines += [wtype] + ["-"*len(wtype)]
        data_vec = self["warnings"][wtype]
        if not hasattr(data_vec, "shape"):
            data_vec = [data_vec]
        for i, data in enumerate(data_vec):
            if len(data_vec) > 1:
                lines += ["| for sweep %i |" % i]
            if wtype == "Unexpectedly Tight Constraints" and data[0][1]:
                data = [(-int(1e5*c.relax_sensitivity),
                         "%+6.2g" % c.relax_sensitivity, c) for _, c in data]
                data = sorted(data)
                lines += constraint_table(data, **kwargs)
            elif wtype == "Unexpectedly Loose Constraints" and data[0][1]:
                data = [(-int(1e5*c.rel_diff),
                         "%.4g %s %.4g" % c.tightvalues, c) for _, c in data]
                data = sorted(data)
                lines += constraint_table(data, **kwargs)
            else:
                for msg, _ in data:
                    lines += [msg, ""]
            lines += [""]
    return lines


TABLEFNS = {"sensitivities": senss_table,
            "top sensitivities": topsenss_table,
            "insensitivities": insenss_table,
            "tightest constraints": tight_table,
            "loose constraints": loose_table,
            "warnings": warnings_table,
           }


def reldiff(val1, val2):
    "Relative difference between val1 and val2 (positive if val2 is larger)"
    if hasattr(val1, "shape") or hasattr(val2, "shape") or val1.magnitude != 0:
        if hasattr(val1, "shape") and val1.shape:
            val1_dims = len(val1.shape)
            if (hasattr(val2, "shape") and val1.shape != val2.shape
                    and val2.shape[:val1_dims] == val1.shape):
                val1_ = np.tile(val1.magnitude, val2.shape[val1_dims:]+(1,)).T
                val1 = val1_ * val1.units
        # numpy division will warn but return infs
        return (val2/val1 - 1).to("dimensionless").magnitude
    elif val2.magnitude == 0:  # both are scalar zeroes
        return 0
    return np.inf  # just val1 is a scalar zero


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
    model = None
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

    def almost_equal(self, sol, reltol=1e-3, sens_abstol=0.01):
        "Checks for almost-equality between two solutions"
        selfvars = set(self["variables"])
        solvars = set(sol["variables"])
        if selfvars != solvars:
            return False
        for key in selfvars:
            if abs(reldiff(self(key), sol(key))) >= reltol:
                return False
            if abs(sol["sensitivities"]["variables"][key]
                   - self["sensitivities"]["variables"][key]) >= sens_abstol:
                return False
        return True

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def diff(self, sol, showvars=None, min_percent=1.0,
             show_sensitivities=True, min_senss_delta=0.1,
             sortbymodel=True):
        """Outputs differences between this solution and another

        Arguments
        ---------
        sol : solution or string
            Strings are treated as paths to valid pickled solutions
        min_percent : float
            The smallest percentage difference in the result to consider
        show_sensitivities : boolean
            if True, also computer sensitivity deltas
        min_senss_delta : float
            The smallest absolute difference in sensitivities to consider

        Returns
        -------
        str
        """
        if isinstance(sol, Strings):
            sol = pickle.load(open(sol))
        selfvars = set(self["variables"])
        solvars = set(sol["variables"])
        if showvars:
            showvars = self._parse_showvars(showvars)
            selfvars = set([k for k in showvars
                            if k in self["variables"]])
            solvars = set([k for k in showvars
                           if k in sol["variables"]])
        sol_diff = {}
        for key in selfvars.intersection(solvars):
            sol_diff[key] = 100*reldiff(self(key), sol(key))
        lines = var_table(sol_diff, "Solution difference", sortbyvals=True,
                          valfmt="%+6.1f%%  ", vecfmt="%+6.1f%% ",
                          printunits=False, minval=min_percent,
                          sortbymodel=sortbymodel)
        if showvars:
            lines[0] += " for variables given in `showvars`"
            lines[1] += "----------------------------------"
        if len(lines) > 3:
            lines.insert(1, "(positive means the argument is bigger)")
        elif sol_diff:
            values = []
            for v in sol_diff.values():
                if hasattr(v, "shape"):
                    values.extend(v.flatten().tolist())
                else:
                    values.append(v)
            values = np.array(values)
            i = np.unravel_index(np.argmax(np.abs(values)), values.shape)
            lines.insert(2, "The largest difference is %g%%" % values[i])

        if show_sensitivities:
            senss_delta = {}
            for key in selfvars.intersection(solvars):
                if key in sol["sensitivities"]["variables"]:
                    val1 = self["sensitivities"]["variables"][key]
                    val2 = sol["sensitivities"]["variables"][key]
                    if hasattr(val1, "shape") and val1.shape:
                        val1_dims = len(val1.shape)
                        if (hasattr(val2, "shape") and val1.shape != val2.shape
                                and val2.shape[:val1_dims] == val1.shape):
                            val1 = np.tile(val1,
                                           val2.shape[val1_dims:]+(1,)).T
                    senss_delta[key] = val2 - val1
                elif key in sol["sensitivities"]["variables"]:
                    print("Key %s is not in this solution's sensitivities"
                          " but is in those of the argument.")
                else:  # for variables that just aren't in any constraints
                    senss_delta[key] = 0

            primal_lines = len(lines)
            lines += var_table(senss_delta, "Solution sensitivity delta",
                               sortbyvals=True,
                               valfmt="%+-6.2f  ", vecfmt="%+-6.2f",
                               printunits=False, minval=min_senss_delta,
                               sortbymodel=sortbymodel)
            if showvars:
                lines[primal_lines] += " for variables given in `showvars`"
                lines[primal_lines + 1] += "----------------------------------"
            if len(lines) > primal_lines + 3:
                lines.insert(
                    primal_lines + 1,
                    "(positive means the argument has a higher sensitivity)")
            elif senss_delta:
                absmaxvalue, maxvalue = 0, 0
                for valarray in senss_delta.values():
                    if not getattr(valarray, "shape", None):
                        value = valarray
                    else:
                        flatvalarray = valarray.flatten()
                        value = flatvalarray[np.argmax(np.abs(valarray))]
                    absvalue = abs(value)
                    if absvalue > absmaxvalue:
                        maxvalue = value
                        absmaxvalue = absvalue
                lines.insert(
                    primal_lines + 2,
                    "The largest sensitivity delta is %+g" % maxvalue)

        if selfvars-solvars:
            lines.append("Variable(s) of this solution"
                         " which are not in the argument:")
            lines.append("\n".join("  %s" % key for key in selfvars-solvars))
            lines.append("")
        if solvars-selfvars:
            lines.append("Variable(s) of the argument"
                         " which are not in this solution:")
            lines.append("\n".join("  %s" % key for key in solvars-selfvars))
            lines.append("")

        out = "\n".join(lines)
        out = out.replace("+0.", " +.")
        out = out.replace("-0.", " -.")
        return out

    def pickle_prep(self):
        "After calling this, the SolutionArray is ready to pickle"
        program, model = self.program, self.model
        self.program, self.model = None, None
        cost = self["cost"]
        self["cost"] = mag(cost)
        warnings = {}
        if "warnings" in self:
            for wtype in self["warnings"]:
                warnings[wtype] = self["warnings"][wtype]
                warnarray = np.array(self["warnings"][wtype])
                warnarray.T[1] = None  # remove pointer to exact constraint
                if len(warnarray.shape) == 2:
                    warnarray = warnarray.tolist()
                self["warnings"][wtype] = warnarray
        return program, model, cost, warnings

    def save(self, filename="solution.pkl"):
        """Pickles the solution and saves it to a file.

        The saved solution is identical except for two things:
            - the cost is made unitless
            - the solution's 'program' attribute is removed
            - the solution's 'model' attribute is removed
            - the data field is removed from the solution's warnings
                (the "message" field is preserved)

        Solution can then be loaded with e.g.:
        >>> import cPickle as pickle
        >>> pickle.load(open("solution.pkl"))
        """
        program, model, cost, warnings = self.pickle_prep()
        pickle.dump(self, open(filename, "w"))
        self["cost"], self["warnings"] = cost, warnings
        self.program, self.model = program, model

    def varnames(self, include):
        "Returns list of variables, optionally with minimal unique names"
        self["variables"].update_keymap()
        keymap = self["variables"].keymap
        names = {}
        for key in (include or self["variables"]):
            if include:
                key, _ = self["variables"].parse_and_index(key)
            keys = keymap[key.name]
            names.update((str(k), k) for k in keys)
        return names

    def savemat(self, filename="solution.mat", include=None):
        "Saves primal solution as matlab file"
        from scipy.io import savemat
        savemat(filename,
                {name.replace("/", "_").replace(".", "__"):
                 np.array(self["variables"][key], "f")
                 for name, key in self.varnames(include).items()})

    def todataframe(self, include=None):
        "Returns primal solution as pandas dataframe"
        import pandas as pd  # pylint:disable=import-error
        rows = []
        cols = ["Name", "Index", "Value", "Units", "Label",
                "Models", "Model Numbers", "Other"]
        for _, key in sorted(self.varnames(include).items(),
                             key=lambda k: k[0]):
            value = self["variables"][key]
            if key.shape:
                idxs = []
                it = np.nditer(np.empty(key.shape), flags=['multi_index'])
                while not it.finished:
                    idx = it.multi_index
                    idxs.append(idx[0] if len(idx) == 1 else idx)
                    it.iternext()
            else:
                idxs = [None]
            for idx in idxs:
                row = [
                    key.name,
                    "" if idx is None else idx,
                    value if idx is None else value[idx],
                ]
                rows.append(row)
                row.extend([
                    key.unitstr(),
                    key.label or "",
                    key.models or "",
                    key.modelnums or "",
                    ", ".join("%s=%s" % (k, v) for (k, v) in key.descr.items()
                              if k not in ["name", "units", "unitrepr",
                                           "idx", "shape", "veckey",
                                           "value", "original_fn",
                                           "models", "modelnums", "label"])
                ])
        return pd.DataFrame(rows, columns=cols)

    def savetxt(self, filename="solution.txt", *args, **kwargs):
        "Saves solution table as a text file"
        with open(filename, "w") as f:
            f.write(self.table(*args, **kwargs).encode("utf-8"))

    def savecsv(self, showvars=None, filename="solution.csv", valcols=5,
                **kwargs):
        "Saves primal solution as a CSV sorted by modelname, like the tables."
        data = self["variables"]
        if showvars:
            showvars = self._parse_showvars(showvars)
            data = {k: data[k] for k in showvars if k in data}
        # if the columns don't capture any dimensions, skip them
        minspan, maxspan = None, 1
        for v in data.values():
            if (getattr(v, "shape", None)
                    and not all(di == 1 for di in v.shape)):
                minspan_ = min((di for di in v.shape if di != 1))
                maxspan_ = min((di for di in v.shape if di != 1))
                if minspan is None or minspan_ < minspan:
                    minspan = minspan_
                if maxspan is None or maxspan_ > maxspan:
                    maxspan = maxspan_
        if minspan is not None and minspan > valcols:
            valcols = 1
        if maxspan < valcols:
            valcols = maxspan
        lines = var_table(data, "", rawlines=True, maxcolumns=valcols,
                          **kwargs)
        with open(filename, "w") as f:
            f.write("Model Name,Variable Name,Value(s)" + ","*valcols
                    + "Units,Description\n")
            for line in lines:
                if line[0] == ("modelname",):
                    f.write(line[1])
                elif not line[1]:  # spacer line
                    f.write("\n")
                else:
                    f.write("," + line[0].replace(" : ", "") + ",")
                    vals = line[1].replace("[", "").replace("]", "").strip()
                    for el in vals.split():
                        f.write(el + ",")
                    f.write(","*(valcols - len(vals.split())))
                    f.write((line[2].replace("[", "").replace("]", "").strip()
                             + ",").encode("utf8"))
                    f.write(line[3].strip() + "\n")

    def subinto(self, posy):
        "Returns NomialArray of each solution substituted into posy."
        if posy in self["variables"]:
            return self["variables"](posy)
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

    def summary(self, showvars=(), ntopsenss=5, **kwargs):
        "Print summary table, showing top sensitivities and no constants"
        showvars = self._parse_showvars(showvars)
        out = self.table(showvars, ["cost", "warnings",
                                    "sweepvariables", "freevariables"],
                         **kwargs)
        constants_in_showvars = showvars.intersection(self["constants"])
        senss_tables = []
        if len(self["constants"]) < ntopsenss+2 or constants_in_showvars:
            senss_tables.append("sensitivities")
        if len(self["constants"]) >= ntopsenss+2:
            senss_tables.append("top sensitivities")
        senss_tables.append("tightest constraints")
        senss_str = self.table(showvars, senss_tables, nvars=ntopsenss,
                               **kwargs)
        if senss_str:
            out += "\n" + senss_str
        return out

    def table(self, showvars=(),
              tables=("cost", "warnings", "sweepvariables", "freevariables",
                      "constants", "sensitivities", "tightest constraints"),
              **kwargs):
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
                strs[-1] += unitstr(cost, into=" [%s]", dimless="")
                strs += [""]
            elif table in TABLEFNS:
                strs += TABLEFNS[table](self, showvars, **kwargs)
            elif table in self:
                data = self[table]
                if showvars:
                    showvars = self._parse_showvars(showvars)
                    data = {k: data[k] for k in showvars if k in data}
                strs += var_table(data, self.table_titles[table], **kwargs)
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
            print("SolutionArray.plot only supports 1-dimensional sweeps")
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
def var_table(data, title, printunits=True, fixedcols=True,
              varfmt="%s : ", valfmt="%-.4g ", vecfmt="%-8.3g",
              included_models=None, excluded_models=None, latex=False,
              minval=0, sortbyvals=False, hidebelowminval=False,
              columns=None, maxcolumns=5, rawlines=False,
              sortbymodel=True, **_):
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
    lines = []
    decorated = []
    models = set()
    for i, (k, v) in enumerate(data.items()):
        v_arr = np.array([v])
        notnan = ~isnan(v_arr)
        if notnan.any() and np.sum(np.abs(v_arr[notnan])) >= minval:
            if minval and hidebelowminval and len(notnan.shape) > 1:
                less_than_min = np.abs(v) <= minval
                v[np.logical_and(~isnan(v), less_than_min)] = 0
            b = isinstance(v, Iterable) and bool(v.shape)
            kmodels = k.descr.get("models", [])
            kmodelnums = k.descr.get("modelnums", [])
            model = "/".join([kstr + (".%i" % knum if knum != 0 else "")
                              for kstr, knum in zip(kmodels, kmodelnums)
                              if kstr])
            if not sortbymodel:
                model = "null"
            models.add(model)
            s = k.str_without("models")
            if not sortbyvals:
                decorated.append((model, b, (varfmt % s), i, k, v))
            else:  # for consistent sorting, add small offset to negative vals
                val = np.mean(np.abs(v)) - (1e-9 if np.mean(v) < 0 else 0)
                val -= hash(k.name)*1e-30
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
            if model != "":
                if not latex:
                    lines.append([("modelname",), model, "", ""])
                else:
                    lines.append([r"\multicolumn{3}{l}{\textbf{" +
                                  model + r"}} \\"])
            oldmodel = model
        label = var.descr.get('label', '')
        units = var.unitstr(" [%s] ") if printunits else ""
        if isvector:
            # TODO: pretty n-dimensional printing?
            if columns is not None:
                ncols = columns
            else:
                last_dim_index = len(val.shape)-1
                horiz_dim = last_dim_index  # default alignment
                ncols = 1
                for i, dim_size in enumerate(val.shape):
                    if dim_size >= ncols and dim_size <= maxcolumns:
                        horiz_dim = i
                        ncols = dim_size
                # align the array with horiz_dim by making it the last one
                dim_order = range(last_dim_index)
                dim_order.insert(horiz_dim, last_dim_index)
                val = val.transpose(dim_order)
            flatval = val.flatten()
            vals = [vecfmt % v for v in flatval[:ncols]]
            bracket = " ] " if len(flatval) <= ncols else ""
            valstr = "[ %s%s" % ("  ".join(vals), bracket)
        else:
            valstr = valfmt % val
        for before, after in VALSTR_REPLACES:
            valstr = valstr.replace(before, after)
        if not latex:
            lines.append([varstr, valstr, units, label])
            if isvector and len(flatval) > ncols:
                values_remaining = len(flatval) - ncols
                while values_remaining > 0:
                    idx = len(flatval)-values_remaining
                    vals = [vecfmt % v for v in flatval[idx:idx+ncols]]
                    values_remaining -= ncols
                    valstr = "  " + "  ".join(vals)
                    for before, after in VALSTR_REPLACES:
                        valstr = valstr.replace(before, after)
                    if values_remaining <= 0:
                        spaces = (-values_remaining
                                  * len(valstr)/(values_remaining + ncols))
                        valstr = valstr + "  ]" + " "*spaces
                    lines.append(["", valstr, "", ""])
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
    if rawlines:
        return lines
    if not latex:
        if lines:
            maxlens = np.max([list(map(len, line)) for line in lines
                              if line[0] != ("modelname",)], axis=0)
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

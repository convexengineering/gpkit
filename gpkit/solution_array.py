"""Defines SolutionArray class"""
import re
from operator import sub
import warnings as pywarnings
import pickle
import numpy as np
from .nomials import NomialArray
from .small_classes import DictOfLists, Strings
from .small_scripts import mag, try_str_without
from .repr_conventions import unitstr, lineagestr


CONSTRSPLITPATTERN = re.compile(r"([^*]\*[^*])|( \+ )|( >= )|( <= )|( = )")

VALSTR_REPLACES = [
    ("+nan", " nan"),
    ("-nan", " nan"),
    ("nan%", "nan "),
    ("nan", " - "),
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


def tight_table(self, _, ntightconstrs=5, tight_senss=1e-2, **kwargs):
    "Return constraint tightness lines"
    if not self.model:
        return []
    title = "Tightest Constraints"
    data = []
    for c in self.model.flat():
        try:
            if c.relax_sensitivity and c.relax_sensitivity >= tight_senss:
                data.append(((-float("%+6.2g" % c.relax_sensitivity), str(c)),
                             "%+6.2g" % c.relax_sensitivity, id(c), c))
        except AttributeError:
            print("Constraint %s had no `relax_sensitivity` attribute." % c)
            return []
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


def loose_table(self, _, min_senss=1e-5, **kwargs):
    "Return constraint tightness lines"
    if not self.model:
        return []
    title = "All Loose Constraints"
    data = [(0, "", c) for c in self.model.flat()
            if c.relax_sensitivity <= min_senss]
    if not data:
        lines = ["No constraints had a sensitivity below %+6.2g." % min_senss]
    else:
        lines = constraint_table(data, **kwargs)
    return [title] + ["-"*len(title)] + lines + [""]


# pylint: disable=too-many-branches,too-many-locals,too-many-statements
def constraint_table(data, sortbymodel=True, showmodels=True, **_):
    "Creates lines for tables where the right side is a constraint."
    excluded = ("units", "unnecessary lineage")
    if not showmodels:
        excluded = ("units", "lineage")  # hide all of it
    models, decorated = {}, []
    for sortby, openingstr, _, constraint in sorted(data):
        model = lineagestr(constraint) if sortbymodel else ""
        if model not in models:
            models[model] = len(models)
        constrstr = try_str_without(constraint, excluded)
        if " at 0x" in constrstr:  # don't print memory addresses
            constrstr = constrstr[:constrstr.find(" at 0x")] + ">"
        decorated.append((models[model], model, sortby, constrstr, openingstr))
    decorated.sort()
    previous_model, lines = None, []
    for varlist in decorated:
        _, model, _, constrstr, openingstr = varlist
        if model != previous_model:
            if lines:
                lines.append(["", ""])
            if model or lines:
                lines.append([("modelname",), model])
            previous_model = model
        constrstr = constrstr.replace(model, "")
        minlen, maxlen = 25, 80
        segments = [s for s in CONSTRSPLITPATTERN.split(constrstr) if s]
        constraintlines = []
        line = ""
        next_idx = 0
        while next_idx < len(segments):
            segment = segments[next_idx]
            next_idx += 1
            if CONSTRSPLITPATTERN.match(segment) and next_idx < len(segments):
                segments[next_idx] = segment[1:] + segments[next_idx]
                segment = segment[0]
            elif len(line) + len(segment) > maxlen and len(line) > minlen:
                constraintlines.append(line)
                line = "  "  # start a new line
            line += segment
            while len(line) > maxlen:
                constraintlines.append(line[:maxlen])
                line = "  " + line[maxlen:]
        constraintlines.append(line)
        lines += [(openingstr + " : ", constraintlines[0])]
        lines += [("", l) for l in constraintlines[1:]]
    maxlens = np.max([list(map(len, line)) for line in lines
                      if line[0] != ("modelname",)], axis=0)
    dirs = [">", "<"]  # we'll check lengths before using zip
    assert len(list(dirs)) == len(list(maxlens))
    fmts = ["{0:%s%s}" % (direc, L) for direc, L in zip(dirs, maxlens)]
    for i, line in enumerate(lines):
        if line[0] == ("modelname",):
            linelist = [fmts[0].format(" | "), line[1]]
        else:
            linelist = [fmt.format(s) for fmt, s in zip(fmts, line)]
        lines[i] = "".join(linelist).rstrip()
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
                         "%+6.2g" % c.relax_sensitivity, id(c), c)
                        for _, c in data]
                lines += constraint_table(data, **kwargs)
            elif wtype == "Unexpectedly Loose Constraints" and data[0][1]:
                data = [(-int(1e5*c.rel_diff),
                         "%.4g %s %.4g" % c.tightvalues, id(c), c)
                        for _, c in data]
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

def unrolled_absmax(values):
    "From an iterable of numbers and arrays, returns the largest magnitude"
    finalval, absmaxest = None, 0
    for val in values:
        absmaxval = np.abs(val).max()
        if absmaxval >= absmaxest:
            absmaxest, finalval = absmaxval, val
    if getattr(finalval, "shape", None):
        return finalval[np.argmax(np.abs(finalval))]
    return finalval


def cast(function, val1, val2):
    "Relative difference between val1 and val2 (positive if val2 is larger)"
    with pywarnings.catch_warnings():  # skip those pesky divide-by-zeros
        pywarnings.simplefilter("ignore")
        if hasattr(val1, "shape") and hasattr(val2, "shape"):
            if val1.ndim == val2.ndim:
                return function(val1, val2)
            lessdim, dimmest = sorted([val1, val2], key=lambda v: v.ndim)
            dimdelta = dimmest.ndim - lessdim.ndim
            add_axes = (slice(None),)*lessdim.ndim + (np.newaxis,)*dimdelta
            if dimmest is val1:
                return function(dimmest, lessdim[add_axes])
            if dimmest is val2:
                return function(lessdim[add_axes], dimmest)
        return function(val1, val2)


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
    _name_collision_varkeys = None
    table_titles = {"sweepvariables": "Sweep Variables",
                    "freevariables": "Free Variables",
                    "constants": "Constants",
                    "variables": "Variables"}

    def name_collision_varkeys(self):
        "Returns the set of contained varkeys whose names are not unique"
        if self._name_collision_varkeys is None:
            self["variables"].update_keymap()
            keymap = self["variables"].keymap
            self._name_collision_varkeys = set()
            for key in list(keymap):
                if hasattr(key, "key"):
                    if len(keymap[key.str_without(["lineage", "vec"])]) > 1:
                        self._name_collision_varkeys.add(key)
        return self._name_collision_varkeys

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

    def almost_equal(self, other, reltol=1e-3, sens_abstol=0.01):
        "Checks for almost-equality between two solutions"
        svars, ovars = self["variables"], other["variables"]
        svks, ovks = set(svars), set(ovars)
        if svks != ovks:
            return False
        for key in svks:
            if abs(cast(np.divide, svars[key], ovars[key]) - 1) >= reltol:
                return False
            if abs(self["sensitivities"]["variables"][key]
                   - other["sensitivities"]["variables"][key]) >= sens_abstol:
                return False
        return True

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def diff(self, other, showvars=None, *, senssdiff=True, sensstol=0.1,
             absdiff=False, abstol=0, reldiff=True, reltol=1.0, **tableargs):
        """Outputs differences between this solution and another

        Arguments
        ---------
        other : solution or string
            strings will be treated as paths to pickled solutions
        senssdiff : boolean
            if True, show sensitivity differences
        sensstol : float
            the smallest sensitivity difference worth showing
        abssdiff : boolean
            if True, show absolute differences
        absstol : float
            the smallest absolute difference worth showing
        reldiff : boolean
            if True, show relative differences
        reltol : float
            the smallest relative difference worth showing

        Returns
        -------
        str
        """
        tableargs.update({"hidebelowminval": True, "sortbyvals": True})
        if isinstance(other, Strings):
            other = pickle.load(open(other, "rb"))
        svars, ovars = self["variables"], other["variables"]
        lines = ["Solution difference",
                 "===================",
                 "(positive means the argument is smaller)", ""]
        svks, ovks = set(svars), set(ovars)
        if showvars:
            lines[0] += " for selected variables"
            lines[1] += "======================="
            showvars = self._parse_showvars(showvars)
            svks = {k for k in showvars if k in svars}
            ovks = {k for k in showvars if k in ovars}
        if svks - ovks:
            lines.append("Variable(s) of this solution"
                         " which are not in the argument:")
            lines.append("\n".join("  %s" % key for key in svks - ovks))
            lines.append("")
        if ovks - svks:
            lines.append("Variable(s) of the argument"
                         " which are not in this solution:")
            lines.append("\n".join("  %s" % key for key in ovks - svks))
            lines.append("")
        sharedvks = svks.intersection(ovks)
        # relative difference #
        if reldiff:
            rel_diff = {vk: 100*(cast(np.divide, svars[vk], ovars[vk]) - 1)
                        for vk in sharedvks}
            lines += var_table(rel_diff, "Relative differences >%g%%" % reltol,
                               valfmt="%+.1f%%  ", vecfmt="%+6.1f%% ",
                               minval=reltol, printunits=False, **tableargs)
            if lines[-2][:10] == "-"*10:  # nothing larger than sensstol
                lines.insert(-1, ("The largest is %g%%"
                                  % unrolled_absmax(rel_diff.values())))
        # absolute difference #
        if absdiff:
            abs_diff = {vk: cast(sub, svars[vk], ovars[vk]) for vk in sharedvks}
            lines += var_table(abs_diff, "Absolute differences >%g" % abstol,
                               valfmt="%+.2g", vecfmt="%+8.2g",
                               minval=abstol, **tableargs)
        # sensitivity difference #
        if senssdiff:
            ssenss = self["sensitivities"]["variables"]
            osenss = other["sensitivities"]["variables"]
            senss_delta = {vk: cast(sub, ssenss.get(vk, 0), osenss.get(vk, 0))
                           for vk in svks.intersection(ovks)}
            title = "Sensitivity differences >%g" % sensstol
            lines += var_table(senss_delta, title,
                               valfmt="%+-.2f  ", vecfmt="%+-6.2f",
                               minval=sensstol, printunits=False, **tableargs)
            if lines[-2][:10] == "-"*10:  # nothing larger than sensstol
                lines.insert(-1, ("The largest is %+g"
                                  % unrolled_absmax(senss_delta.values())))
        return "\n".join(lines)

    def pickle_prep(self):
        "After calling this, the SolutionArray is ready to pickle"
        program, model = self.program, self.model
        self.program = self.model = None
        cost = self["cost"]
        self["cost"] = mag(cost)
        warnings = {}
        if "warnings" in self:
            for wtype in self["warnings"]:
                warnings[wtype] = self["warnings"][wtype]
                warnarray = np.array(self["warnings"][wtype])
                warnarray.T[1] = None  # remove pointer to exact constraint  # pylint: disable=unsupported-assignment-operation
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
        >>> import pickle
        >>> pickle.load(open("solution.pkl"))
        """
        program, model, cost, warnings = self.pickle_prep()
        pickle.dump(self, open(filename, "wb"), protocol=1)
        self["cost"], self["warnings"] = cost, warnings
        self.program, self.model = program, model

    def varnames(self, showvars, exclude):
        "Returns list of variables, optionally with minimal unique names"
        if showvars:
            showvars = self._parse_showvars(showvars)
        for key in self.name_collision_varkeys():
            key.descr["necessarylineage"] = True
        names = {}
        for key in showvars or self["variables"]:
            for k in self["variables"].keymap[key]:
                names[k.str_without(exclude)] = k
        for key in self.name_collision_varkeys():
            del key.descr["necessarylineage"]
        return names

    def savemat(self, filename="solution.mat", showvars=None,
                excluded=("unnecessary lineage", "vec")):
        "Saves primal solution as matlab file"
        from scipy.io import savemat
        savemat(filename,
                {name.replace(".", "_"): np.array(self["variables"][key], "f")
                 for name, key in self.varnames(showvars, excluded).items()})

    def todataframe(self, showvars=None,
                    excluded=("unnecessary lineage", "vec")):
        "Returns primal solution as pandas dataframe"
        import pandas as pd  # pylint:disable=import-error
        rows = []
        cols = ["Name", "Index", "Value", "Units", "Label",
                "Lineage", "Other"]
        for _, key in sorted(self.varnames(showvars, excluded).items(),
                             key=lambda k: k[0]):
            value = self["variables"][key]
            if key.shape:
                idxs = []
                it = np.nditer(np.empty(value.shape), flags=['multi_index'])
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
                    value if idx is None else value[idx]]
                rows.append(row)
                row.extend([
                    key.unitstr(),
                    key.label or "",
                    key.lineage or "",
                    ", ".join("%s=%s" % (k, v) for (k, v) in key.descr.items()
                              if k not in ["name", "units", "unitrepr",
                                           "idx", "shape", "veckey",
                                           "value", "original_fn",
                                           "lineage", "label"])])
        return pd.DataFrame(rows, columns=cols)

    def savetxt(self, filename="solution.txt", printmodel=True, **kwargs):
        "Saves solution table as a text file"
        with open(filename, "w") as f:
            if printmodel and self.model:
                f.write(str(self.model))
            f.write(self.table(**kwargs))

    def savecsv(self, showvars=None, filename="solution.csv", valcols=5):
        "Saves primal solution as a CSV sorted by modelname, like the tables."
        data = self["variables"]
        if showvars:
            showvars = self._parse_showvars(showvars)
            data = {k: data[k] for k in showvars if k in data}
        # if the columns don't capture any dimensions, skip them
        minspan, maxspan = None, 1
        for v in data.values():
            if getattr(v, "shape", None) and any(di != 1 for di in v.shape):
                minspan_ = min((di for di in v.shape if di != 1))
                maxspan_ = max((di for di in v.shape if di != 1))
                if minspan is None or minspan_ < minspan:
                    minspan = minspan_
                if maxspan is None or maxspan_ > maxspan:
                    maxspan = maxspan_
        if minspan is not None and minspan > valcols:
            valcols = 1
        if maxspan < valcols:
            valcols = maxspan
        lines = var_table(data, "", rawlines=True, maxcolumns=valcols)
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
                             + ","))
                    f.write(line[3].strip() + "\n")

    def subinto(self, posy):
        "Returns NomialArray of each solution substituted into posy."
        if posy in self["variables"]:
            return self["variables"](posy)

        if not hasattr(posy, "sub"):
            raise ValueError("no variable '%s' found in the solution" % posy)

        if len(self) > 1:
            return NomialArray([self.atindex(i).subinto(posy)
                                for i in range(len(self))])

        return posy.sub(self["variables"])

    def _parse_showvars(self, showvars):
        showvars_out = set()
        for k in showvars:
            k, _ = self["variables"].parse_and_index(k)
            keys = self["variables"].keymap[k]
            showvars_out.update(keys)
        return showvars_out

    def summary(self, showvars=(), ntopsenss=5, **kwargs):
        "Print summary table, showing top sensitivities and no constants"
        showvars = self._parse_showvars(showvars)
        out = self.table(showvars, ["cost", "warnings", "sweepvariables",
                                    "freevariables"], **kwargs)
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
        varlist = list(self["variables"])
        has_only_one_model = True
        for var in varlist[1:]:
            if var.lineage != varlist[0].lineage:
                has_only_one_model = False
                break
        if has_only_one_model:
            kwargs["sortbymodel"] = False
        for key in self.name_collision_varkeys():
            key.descr["necessarylineage"] = True
        showvars = self._parse_showvars(showvars)
        strs = []
        for table in tables:
            if table == "cost":
                cost = self["cost"]  # pylint: disable=unsubscriptable-object
                if kwargs.get("latex", None):  # cost is not printed for latex
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
        for key in self.name_collision_varkeys():
            del key.descr["necessarylineage"]
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
def var_table(data, title, printunits=True, latex=False, rawlines=False,
              varfmt="%s : ", valfmt="%-.4g ", vecfmt="%-8.3g",
              minval=0, sortbyvals=False, hidebelowminval=False,
              included_models=None, excluded_models=None, sortbymodel=True,
              maxcolumns=5, **_):
    """
    Pretty string representation of a dict of VarKeys
    Iterable values are handled specially (partial printing)

    Arguments
    ---------
    data : dict whose keys are VarKey's
        data to represent in table
    title : string
    printunits : bool
    latex : int
        If > 0, return latex format (options 1-3); otherwise plain text
    varfmt : string
        format for variable names
    valfmt : string
        format for scalar values
    vecfmt : string
        format for vector values
    minval : float
        skip values with all(abs(value)) < minval
    sortbyvals : boolean
        If true, rows are sorted by their average value instead of by name.
    included_models : Iterable of strings
        If specified, the models (by name) to include
    excluded_models : Iterable of strings
        If specified, model names to exclude
    """
    if not data:
        return []
    decorated, models = [], set()
    for i, (k, v) in enumerate(data.items()):
        if np.nanmax(np.abs(v)) > minval:
            if minval and hidebelowminval and getattr(v, "shape", None):
                v[np.abs(v) <= minval] = np.nan
            model = lineagestr(k.lineage) if sortbymodel else ""
            models.add(model)
            b = bool(getattr(v, "shape", None))
            s = k.str_without(("lineage", "vec"))
            if not sortbyvals:
                decorated.append((model, b, (varfmt % s), i, k, v))
            else:  # for consistent sorting, add small offset to negative vals
                val = np.nanmean(np.abs(v)) - (1e-9 if np.nanmean(v) < 0 else 0)
                sort = (float("%.4g" % -val), k.name)
                decorated.append((model, sort, b, (varfmt % s), i, k, v))
    if included_models:
        included_models = set(included_models)
        included_models.add("")
        models = models.intersection(included_models)
    if excluded_models:
        models = models.difference(excluded_models)
    decorated.sort()
    previous_model, lines = None, []
    for varlist in decorated:
        if not sortbyvals:
            model, isvector, varstr, _, var, val = varlist
        else:
            model, _, isvector, varstr, _, var, val = varlist
        if model not in models:
            continue
        if model != previous_model:
            if lines:
                lines.append(["", "", "", ""])
            if model:
                if not latex:
                    lines.append([("modelname",), model, "", ""])
                else:
                    lines.append(
                        [r"\multicolumn{3}{l}{\textbf{" + model + r"}} \\"])
            previous_model = model
        label = var.descr.get('label', '')
        units = var.unitstr(" [%s] ") if printunits else ""
        if not isvector:
            valstr = valfmt % val
        else:
            last_dim_index = len(val.shape)-1
            horiz_dim, ncols = last_dim_index, 1  # starting values
            for dim_idx, dim_size in enumerate(val.shape):
                if ncols <= dim_size <= maxcolumns:
                    horiz_dim, ncols = dim_idx, dim_size
            # align the array with horiz_dim by making it the last one
            dim_order = list(range(last_dim_index))
            dim_order.insert(horiz_dim, last_dim_index)
            flatval = val.transpose(dim_order).flatten()
            vals = [vecfmt % v for v in flatval[:ncols]]
            bracket = " ] " if len(flatval) <= ncols else ""
            valstr = "[ %s%s" % ("  ".join(vals), bracket)
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
                                  * len(valstr)//(values_remaining + ncols))
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
            dirs = [">", "<", "<", "<"]
            # check lengths before using zip
            assert len(list(dirs)) == len(list(maxlens))
            fmts = ["{0:%s%s}" % (direc, L) for direc, L in zip(dirs, maxlens)]
        for i, line in enumerate(lines):
            if line[0] == ("modelname",):
                line = [fmts[0].format(" | "), line[1]]
            else:
                line = [fmt.format(s) for fmt, s in zip(fmts, line)]
            lines[i] = "".join(line).rstrip()
        lines = [title] + ["-"*len(title)] + lines + [""]
    else:
        colfmt = {1: "llcl", 2: "lcl", 3: "llc"}
        lines = (["\n".join(["{\\footnotesize",
                             "\\begin{longtable}{%s}" % colfmt[latex],
                             "\\toprule",
                             " & ".join(coltitles) + " \\\\ \\midrule"])] +
                 [" & ".join(l) + " \\\\" for l in lines] +
                 ["\n".join(["\\bottomrule", "\\end{longtable}}", ""])])
    return lines

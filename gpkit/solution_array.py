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


    def table(self, tables=["cost", "sweepvariables", "freevariables",
                            "constants", "sensitivities"], latex=False,
              fixedcols=True, included_models=None, excluded_models=None):
        if isinstance(tables, Strings):
            tables = [tables]
        strs = []
        for table in tables:
            subdict = self.get(table, None)
            table_title = self.table_titles[table]
            if table == "cost":
                strs += ["\n%s\n----" % table_title]
                if len(self) > 1:
                    costs = ["%-8.3g" % c for c in subdict[:4]]
                    strs += [" [ %s %s ]" % ("  ".join(costs),
                                             "..." if len(self) > 4 else "")]
                    cost_units = self.program[0].cost.units
                else:
                    strs += [" %-.4g" % subdict]
                    cost_units = self.program.cost.units
                strs[-1] += unitstr(cost_units, into=" [%s] ", dimless="")
                strs += [""]
            elif not subdict:
                continue
            elif table == "sensitivities":
                strs += results_table(subdict["variables"], table_title,
                                      fixedcols=fixedcols,
                                      included_models=included_models,
                                      excluded_models=excluded_models,
                                      minval=1e-2,
                                      sortbyvals=True,
                                      printunits=False,
                                      latex=latex)
            else:
                strs += results_table(subdict, table_title,
                                      fixedcols=fixedcols,
                                      included_models=included_models,
                                      excluded_models=excluded_models,
                                      latex=latex)
        return "\n".join(strs)


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
            if not sortbyvals:
                decorated.append((model, b, (varfmt % k.nomstr), i, k, v))
            else:
                decorated.append((model, np.mean(v), b, (varfmt % k.nomstr), i, k, v))
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
        if not latex:
            lines.append([varstr, valstr, units, label])
        else:
            varstr = varstr.replace(" : ", "")
            lines.append(["$", varstr, "$ & ",  valstr, " & $",  units.replace('**','^'), "$ & ", label, " \\\\"])
    if not latex:
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
    else:
        lines = ["\\toprule"] + [title + "\\\\"] + ["\\midrule"] + [''.join(l) for l in lines] + ["\\bottomrule"] + [""]
    return lines


def parse_result(result, constants, beforesubs, sweep={}, linkedsweep={},
                 freevar_sensitivity_tolerance=1e-4,
                 localmodel_sensitivity_requirement=0.1):
    "Parses a GP-like result dict into a SolutionArray-like dict."
    cost = result["cost"]
    freevariables = dict(result["variables"])
    sweepvariables = {var: val for var, val in constants.items()
                      if var in sweep or var in linkedsweep}
    constants = {var: val for var, val in constants.items()
                 if var not in sweepvariables}
    variables = dict(freevariables)
    variables.update(constants)
    variables.update(sweepvariables)
    sensitivities = dict(result["sensitivities"])

    # Remap monomials after substitution and simplification.
    #  The monomial sensitivities from the GP/SP are in terms of this
    #  smaller post-substitution list of monomials, so we need to map that
    #  back to the pre-substitution list.
    #
    #  Each "smap" is a list of HashVectors (mmaps),
    #    whose keys are monomial indexes pre-substitution,
    #    and whose values are the percentage of the simplified monomial's
    #    coefficient that came from that particular parent
    nu = result["sensitivities"]["monomials"]
    # HACK: simplified solves need a mutated beforesubs, as created in Model
    if hasattr(beforesubs, "smaps"):
        nu_ = np.zeros(len(beforesubs.cs))
        little_counter, big_counter = 0, 0
        for j, smap in enumerate(beforesubs.smaps):
            for i, mmap in enumerate(smap):
                for idx, percentage in mmap.items():
                    nu_[idx + big_counter] += percentage*nu[i + little_counter]
            little_counter += len(smap)
            big_counter += len(beforesubs.signomials[j].cs)
    sensitivities["monomials"] = nu_

    sens_vars = {var: sum([beforesubs.exps[i][var]*nu_[i] for i in locs])
                 for (var, locs) in beforesubs.varlocs.items()}
    sensitivities["variables"] = sens_vars

    # free-variable sensitivities must be <= some epsilon
    for var, S in sensitivities["variables"].items():
        if var in freevariables and abs(S) > freevar_sensitivity_tolerance:
            raise ValueError("free variable too sensitive: S_{%s} = "
                             "%0.2e" % (var, S))

    localexp = {var: S for (var, S) in sens_vars.items()
                if abs(S) >= localmodel_sensitivity_requirement}
    localcs = (variables[var]**-S for (var, S) in localexp.items())
    localc = functools_reduce(mul, localcs, cost)
    localmodel = Monomial(localexp, localc)

    # vectorvar substitution
    veckeys = set()
    for var in beforesubs.varlocs:
        if "idx" in var.descr and "shape" in var.descr:
            descr = dict(var.descr)
            idx = descr.pop("idx")
            if "value" in descr:
                descr.pop("value")
            if "units" in descr:
                units = descr.pop("units")
                veckey = VarKey(**descr)
                veckey.descr["units"] = units
            else:
                veckey = VarKey(**descr)
            veckeys.add(veckey)

            for vardict in [variables, sensitivities["variables"],
                            constants, sweepvariables, freevariables]:
                if var in vardict:
                    if veckey in vardict:
                        vardict[veckey][idx] = vardict[var]
                    else:
                        vardict[veckey] = np.full(var.descr["shape"], np.nan)
                        vardict[veckey][idx] = vardict[var]

                    del vardict[var]

    if hasattr(beforesubs, "varkeysubs"):
        for origvk, subvk in beforesubs.varkeysubs.items():
            for data in [constants, sweepvariables, freevariables, variables,
                         sensitivities["variables"]]:
                if subvk in data:
                    qty = isinstance(origvk.units, Quantity)
                    if data is sensitivities["variables"] or not qty:
                        data[origvk] = data[subvk]
                    else:
                        scale = (subvk.units/origvk.units).to("dimensionless")
                        data[origvk] = data[subvk] * scale

    return dict(cost=cost,
                constants=constants,
                sweepvariables=sweepvariables,
                freevariables=freevariables,
                variables=variables,
                sensitivities=sensitivities,
                localmodel=localmodel)

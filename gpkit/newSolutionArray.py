import numpy as np

from .posyarray import PosyArray
from .small_classes import Strings
from .small_classes import DictOfLists
from .small_scripts import results_table
from .small_scripts import unitstr
from .small_scripts import mag


class SolutionArray(DictOfLists):
    "DictofLists extended with posynomial substitution."

    def __len__(self):
        try:
            return len(self["cost"])
        except TypeError:
            return 1

    def __call__(self, p):
        return mag(self.subinto(p).c)

    def getvars(self, *args):
        out = [self["variables"][arg] for arg in args]
        if len(out) == 1:
            return out[0]
        else:
            return out

    def subinto(self, p):
        "Returns PosyArray of each solution substituted into p."
        if p in self["variables"]:
            return PosyArray(self["variables"][p])
        if len(self) > 1:
            return PosyArray([p.sub(self.atindex(i)["variables"])
                              for i in range(len(self["cost"]))])
        else:
            return p.sub(self["variables"])

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
            return subbed.c

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

            strs[-1] += unitstr(cost_units, into=" [%s] ", dimless="")
            strs += [""]
        if "freevariables" in tables:
            strs += [results_table(self["freevariables"],
                                   "Free Variables",
                                   fixedcols=fixedcols)]
        if "sweepvariables" in tables:
            strs += [results_table(self["sweepvariables"],
                                   "Sweep Variables",
                                   fixedcols=fixedcols)]
        if "constants" in tables:
            strs += [results_table({k: v[0] for (k, v)
                                    in self["constants"].items()},
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

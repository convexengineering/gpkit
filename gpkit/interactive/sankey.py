"implements Sankey"
from collections import defaultdict
import numpy as np
from ipysankeywidget import SankeyWidget  # pylint: disable=import-error
from ipywidgets import Layout
from .. import ConstraintSet, Model
from .. import GeometricProgram, SequentialGeometricProgram
from ..nomials.math import MonomialEquality
from ..small_classes import Count
from .. import GPCOLORS


INSENSITIVE = 1e-2


def getcolor(value):
    "color scheme for sensitivities"
    if abs(value) < INSENSITIVE:
        return "#cfcfcf"
    return GPCOLORS[0 if value < 0 else 1]


class Sankey(object):
    "diagrams of sensitivity flow"
    def __init__(self, model):
        self.links = defaultdict(float)
        self.counter = Count()
        if isinstance(model, Model):
            if model.program is None:
                raise ValueError("Model must be solved before a Sankey"
                                 " diagram can be made.")
            if isinstance(model.program, GeometricProgram):
                model = model.program
            elif isinstance(model.program, SequentialGeometricProgram):
                model = model.program.gps[-1]
        if not isinstance(model, GeometricProgram):
            raise ValueError("did not find a GeometricProgram in the `model`"
                             " argument: try passing a particular GP.")
        self.gp = model
        self.constr_name = {}
        self.nodes = []
        self.var_eqs = set()
        self._varprops = None

    def constrlinks(self, constrset, target=None):
        "adds links of a given constraint set to self.links"
        if target is None:  # set final target
            target = getattr(constrset, "name", None) or "Model"
            if getattr(constrset, "num", None):
                target += ".%i" % constrset.num
        for constr in constrset:
            if isinstance(constr, ConstraintSet):
                if getattr(constr, "name", None):
                    source = constr.name
                    source += ".%i" % constr.num if constr.num else ""
                    # value is negative so that the plot is GPBLU
                    self.links[source, target] -= constr.relax_sensitivity
                    self.constrlinks(constr, source)
                else:
                    self.constrlinks(constr, target)

    # pylint: disable=invalid-name, too-many-locals, too-many-branches, too-many-statements
    def varlinks(self, constrset, key, target=None, printing=True):
        "adds links of a given variable in self.gp to self.links"
        if target is None:  # set final target as the variable itself
            value = constrset.v_ss[key]  # if it's zero
            target = getattr(constrset, "name", None) or "Model"
            if getattr(constrset, "num", None):
                target += ".%i" % constrset.num
            source = str(key)
            shortname = (key.str_without(["models"])
                         + key.unitstr(into=" [%s]", dimless=" [-]"))
            self.nodes.append({"id": source, "title": shortname})
            self.links[target, source] += value
            if key in self.gp.result["sensitivities"]["cost"]:
                cost_senss = self.gp.result["sensitivities"]["cost"]
                value = -cost_senss[key]  # sensitivites flow _from_ cost
                self.links[source, "(objective)"] += value
                if printing:
                    print ("(objective) adds %+.3g to the sensitivity"
                           " of %s" % (-value, key))
                    print "(objective) is", self.gp.cost, "\n"
        for constr in constrset:
            if key not in constr.v_ss:
                continue
            value = constr.v_ss[key]
            if isinstance(constr, ConstraintSet):
                if getattr(constr, "name", None):
                    source = constr.name
                    source += ".%i" % constr.num if constr.num else ""
                    self.links[source, target] += value
                    self.varlinks(constr, key, source, printing)
                else:
                    self.varlinks(constr, key, target, printing)
            else:
                if constr not in self.constr_name:
                    # use unicode's circled letters for constraint labels
                    source = unichr(self.counter.next()+9398)
                    self.constr_name[constr] = source
                else:
                    source = self.constr_name[constr]
                if printing:
                    print ("%s adds %+.3g to the overall sensitivity of %s"
                           % (source, value, key))
                    print source, "is", constr.str_without("units"), "\n"
                if ((isinstance(constr, MonomialEquality)
                     or abs(value) >= INSENSITIVE)
                        and all(len(getattr(p, "hmap", [])) == 1
                                and p.hmap.keys()[0].values() == [1]
                                for p in [constr.left, constr.right])):
                    leftkey = constr.left.hmap.keys()[0].keys()[0]
                    if key != leftkey:
                        key2 = leftkey
                    else:
                        key2 = constr.right.hmap.keys()[0].keys()[0]
                    if key2 not in self.var_eqs:  # not already been added
                        self.var_eqs.update([key2, key])
                        self.varlinks(self.gp[0], key2, printing=printing)
                        self.nodes.append({"id": source,
                                           "passthrough": constr})
                self.links[source, target] += value

    # pylint: disable=too-many-arguments
    def diagram(self, variables=None, flowright=False, width=900, height=400,
                top=0, bottom=0, left=120, right=55, printing=True):
        "creates links and an ipython widget to show them"
        margins = dict(top=top, bottom=bottom, left=left, right=right)
        self.__init__(self.gp)
        if not variables:
            self.constrlinks(self.gp[0])
        else:
            if not getattr(variables, "__len__", False):
                variables = [variables]
            for var in variables:
                self.varlinks(self.gp[0], var.key, printing=printing)
            # if var_eqs were found, label them on the diagram
            lookup = {key: i for i, key in
                      enumerate(sorted(map(str, self.var_eqs)))}
            for node in self.nodes:
                if node["id"] in lookup:
                    # use inverted circled numbers to id the variables...
                    node["title"] += " " + unichr(0x2776+lookup[node["id"]])
                elif "passthrough" in node:
                    cn = node.pop("passthrough")
                    l_idx = lookup[str(cn.left.hmap.keys()[0].keys()[0])]
                    r_idx = lookup[str(cn.right.hmap.keys()[0].keys()[0])]
                    op = {"=": "=", ">=": u"\u2265", "<=": u"\u2264"}[cn.oper]
                    # ...so that e.g. (1) >= (2) can label the constraints
                    node["title"] = (node["id"]+u"\u2009"+unichr(l_idx+0x2776)
                                     + op + unichr(r_idx+0x2776))
        if flowright:
            r, l = margins["right"], margins["left"]
            margins["left"], margins["right"] = r, l
        links = []
        maxflow = np.abs(self.links.values()).max()
        for (source, target), value in self.links.items():
            if not flowright:
                source, target = target, source
            links.append({"source": source, "target": target,
                          "value": max(abs(value), maxflow/1e5),
                          "color": getcolor(value)})
        return SankeyWidget(nodes=self.nodes, links=links,
                            layout=Layout(width=str(width),
                                          height=str(height)),
                            margins=margins)

    @property
    def variable_properties(self):
        "Gets and caches properties for contained variables"
        if not self._varprops:
            varprops = {}
            var_eqs = set()
            for key in self.gp.v_ss:
                if key in var_eqs:
                    continue
                self.__init__(self.gp)
                self.varlinks(self.gp, key, printing=False)
                maxflow = max(self.links.values())
                if maxflow > 0.01:  # TODO: arbitrary threshold
                    varprops[key] = {"constraints": self.counter.next(),
                                     "maxflow": maxflow}
                var_eqs.update(self.var_eqs)
            self.__init__(self.gp)
            self._varprops = varprops
        return self._varprops

    def sorted_by(self, prop, idx, **kwargs):
        "chooses a variable by its rank in # of constraints or maximum flow"
        key = sorted(self.variable_properties.items(),
                     key=lambda i: (-i[1][prop], str(i[0])))[idx][0]
        return self.diagram(key, **kwargs)

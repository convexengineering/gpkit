"implements Sankey"
import string
from ipysankeywidget import SankeyWidget
from gpkit import ConstraintSet, Model
from gpkit.nomials.math import MonomialEquality
from gpkit.small_classes import Count
from gpkit import GPCOLORS


INSENSITIVE = 1e-7


def getcolor(value):
    "color scheme for sensitivities"
    if abs(value or 1e-30) < INSENSITIVE:
        return "#cfcfcf"
    return GPCOLORS[1 if value > 0 else 0]


class Sankey(object):
    "diagrams of sensitivity flow"
    def __init__(self, model):
        self.links = []
        self.counter = None
        self.model = model
        self.constr_name = {}
        self.nodes = []
        self.var_eqs = set()
        self._varprops = None

    def constrlinks(self, constrset, target=None):
        "adds links of a given constraint set to self.links"
        if target is None:  # set final target
            target = constrset.name or "[Model]"
            if constrset.num:
                target += ".%i" % constrset.num
        for constr in constrset:
            if isinstance(constr, ConstraintSet):
                if isinstance(constr, Model):
                    # value is negative so that the plot is GPBLU
                    value = -constr.relax_sensitivity
                    source = constr.name
                    source += ".%i" % constr.num if constr.num else ""
                    self.links.append({"target": target, "source": source,
                                       "value": abs(value or 1e-30),
                                       "color": getcolor(value)})
                    self.constrlinks(constr, source)
                else:
                    self.constrlinks(constr, target)

    # pylint: disable=invalid-name, too-many-locals, too-many-branches, too-many-statements
    def varlinks(self, constrset, key, target=None, printing=True):
        "adds links of a given variable in self.model to self.links"
        if target is None:  # set final target as the variable itself
            value = constrset.v_ss[key]  # if it's zero
            target = constrset.name or "[Model]"
            if constrset.num:
                target += ".%i" % constrset.num
            source = str(key)
            shortname = (key.str_without(["models"])
                         + key.unitstr(into=" [%s]", dimless=" [-]"))
            self.nodes.append({"id": source,
                               "title": shortname})
            self.links.append({"target": source, "source": target,
                               "value": abs(value or 1e-30),
                               "color": getcolor(value)})
            if key in self.model.solution["sensitivities"]["cost"]:
                cost_senss = self.model.solution["sensitivities"]["cost"]
                value = -cost_senss[key]  # sensitivites flow _from_ cost
                self.links.append({"target": "(objective)", "source": source,
                                   "signed_value": value,
                                   "value": abs(value or 1e-30),
                                   "color": getcolor(value)})
                if printing:
                    print ("(objective) adds %+.3g to the overall sensitivity"
                           " of %s" % (-value, key))
                    print "(objective) is", self.model.cost, "\n"
        for constr in constrset:
            if key not in constr.v_ss:
                continue
            value = constr.v_ss[key]
            # TODO: add filter-by-abs argument?
            if isinstance(constr, ConstraintSet):
                if isinstance(constr, Model):
                    source = constr.name
                    source += ".%i" % constr.num if constr.num else ""
                    self.links.append({"target": target, "source": source,
                                       "signed_value": value,
                                       "value": abs(value or 1e-30),
                                       "color": getcolor(value)})
                    self.varlinks(constr, key, source, printing)
                else:
                    self.varlinks(constr, key, target, printing)
            else:
                if constr not in self.constr_name:
                    source = "(%s)" % string.ascii_letters[self.counter.next()]
                    self.constr_name[constr] = source
                else:
                    source = self.constr_name[constr]
                if printing:
                    print ("%s adds %+.3g to the overall sensitivity of %s"
                           % (source, value, key))
                    print source, "is", constr.str_without("units"), "\n"
                flowcolor = getcolor(value)
                if (len(getattr(constr.left, "hmap", [])) == 1
                        and len(getattr(constr.right, "hmap", [])) == 1
                        and constr.left.hmap.keys()[0].values() == [1]
                        and constr.right.hmap.keys()[0].values() == [1]
                        and (abs(value) >= INSENSITIVE
                             or isinstance(constr, MonomialEquality))):
                    leftkey = constr.left.hmap.keys()[0].keys()[0]
                    if key != leftkey:
                        key2 = leftkey
                    else:
                        key2 = constr.right.hmap.keys()[0].keys()[0]
                    if key2 in self.var_eqs:
                        continue
                    self.var_eqs.update([key2, key])
                    value = 0  # since it's just pass-through
                    flowcolor = "black"  # to highlight it
                    # now to remove duplicate flows!
                    # TODO: does this only need to be done for constants?
                    nlinks = len(self.links)
                    self.varlinks(self.model, key2, printing=printing)
                    newlinks = len(self.links) - nlinks
                    newlinkmap = {}
                    for link in self.links[-newlinks:]:
                        s, t = link["source"], link["target"]
                        newlinkmap[(s, t)] = link
                        if (s, t) == (source, target):
                            break
                    for link in reversed(self.links[:nlinks]):
                        linkkey = (link["source"], link["target"])
                        if linkkey in newlinkmap:
                            self.links.remove(newlinkmap[linkkey])
                            newval = newlinkmap.pop(linkkey)["signed_value"]
                            link["signed_value"] += newval
                            link["value"] = abs(link["signed_value"])
                            link["color"] = getcolor(link["signed_value"])
                self.links.append({"target": target, "source": source,
                                   "signed_value": value,
                                   "value": abs(value or 1e-30),
                                   "color": flowcolor})

    def diagram(self, variables=None, flowright=False, width=900, height=400,
                top=0, bottom=0, left=100, right=25, printing=True):
        "creates links and an ipython widget to show them"
        margins = dict(top=top, bottom=bottom, left=left, right=right)
        self.counter = Count()
        self.links = []
        if not variables:
            self.constrlinks(self.model)
        else:
            if not getattr(variables, "__len__", False):
                variables = [variables]
            for var in variables:
                self.varlinks(self.model, var.key, printing=printing)
        if flowright:
            r, l = margins["right"], margins["left"]
            margins["left"], margins["right"] = r, l
        else:
            for link in self.links:
                link["source"], link["target"] = link["target"], link["source"]
        return SankeyWidget(nodes=self.nodes, links=self.links,
                            margins=margins, width=width, height=height)

    @property
    def variable_properties(self):
        "Gets and caches properties for contained variables"
        if not self._varprops:
            varprops = {}
            var_eqs = set()
            for key in self.model.v_ss:
                if key in var_eqs:
                    continue
                self.links = []
                self.counter = Count()
                self.varlinks(self.model, key, printing=False)
                maxflow = max(l["value"] for l in self.links)
                if maxflow > 0.01:  # TODO: arbitrary threshold
                    varprops[key] = {"constraints": self.counter.next(),
                                     "maxflow": maxflow}
                var_eqs.update(self.var_eqs)
                self.__init__(self.model)
            self._varprops = varprops
        return self._varprops

    def sorted_by(self, prop, idx, **kwargs):
        "chooses a variable by its rank in # of constraints or maximum flow"
        key = sorted(self.variable_properties.items(),
                     key=lambda i: (-i[1][prop], str(i[0])))[idx][0]
        return self.diagram(key, **kwargs)

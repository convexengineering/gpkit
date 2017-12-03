from ipysankeywidget import SankeyWidget
from gpkit import ConstraintSet, Model
from gpkit.nomials.math import MonomialEquality
from gpkit.small_classes import Count
from gpkit import GPCOLORS, Variable


def getcolor(value):
    if abs(value) < 1e-7:
        return "#cfcfcf"
    return GPCOLORS[1 if value > 0 else 0]


class Sankey(object):
    def __init__(self, model):
        self.links = []
        self.counter = None
        self.model = model
        self.constr_name = {}
        self.nodes = []
        self.var_eqs = set()

    def constrlinks(self, constrset, target=None):
        if target is None:  # set final target
            target = constrset.name or "[Model]"
            if constrset.num:
                target += ".%i" % constrset.num
        for constr in constrset:
            if isinstance(constr, ConstraintSet):
                if isinstance(constr, Model):
                    value = -constr.lasum  # negative to get blue color
                    source = constr.name
                    source += ".%i" % constr.num if constr.num else ""
                    self.links.append({"target": target, "source": source,
                                       "value": abs(value),
                                       "color": getcolor(value)})
                    self.constrlinks(constr, source)
                else:
                    self.constrlinks(constr, target)

    def varlinks(self, constrset, vk, target=None, printing=True, addvarlink=True):
        if target is None:  # set final target as the variable itself
            value = constrset.v_ss[vk] or 1e-30  # if it's zero
            target = constrset.name or "[Model]"
            if constrset.num:
                target += ".%i" % constrset.num
            source = str(vk)
            shortname = (vk.str_without(["models"])
                         + vk.unitstr(into=" [%s]", dimless=" [-]"))
            self.nodes.append({"id": source,
                               "title": shortname})
            self.links.append({"target": source, "source": target,
                               "value": abs(value), "color": getcolor(value)})
            if vk in self.model.solution["sensitivities"]["cost"]:
                cost_senss = self.model.solution["sensitivities"]["cost"]
                value = cost_senss[vk]
                self.links.append({"target": source, "source": "(objective)",
                                   "value": abs(value),
                                   "color": getcolor(value)})
                if printing:
                    print ("(objective) adds %+.3g to the overall sensitivity"
                           " of %s" % (value, vk))
                    print "(objective) is", self.model.cost, "\n"
        for constr in constrset:
            if vk not in constr.v_ss:
                continue
            value = constr.v_ss[vk] or 1e-30
            # TODO: add filter-by-abs argument?
            if isinstance(constr, ConstraintSet):
                if isinstance(constr, Model):
                    source = constr.name
                    source += ".%i" % constr.num if constr.num else ""
                    self.links.append({"target": target, "source": source,
                                       "value": abs(value),
                                       "color": getcolor(value)})
                    self.varlinks(constr, vk, source, printing)
                else:
                    self.varlinks(constr, vk, target, printing)
            else:
                flowvalue = abs(value)
                flowcolor = getcolor(value)
                if (isinstance(constr, MonomialEquality)
                        and constr.left.hmap.keys()[0].values() == [1]
                        and constr.right.hmap.keys()[0].values() == [1]):
                    leftkey = constr.left.hmap.keys()[0].keys()[0]
                    if vk != leftkey:
                        vk2 = leftkey
                    else:
                        vk2 = constr.right.hmap.keys()[0].keys()[0]
                    if vk2 in self.var_eqs:
                        continue
                    self.var_eqs.update([vk2, vk])
                    self.varlinks(self.model, vk2, printing=printing)
                    flowvalue = 1e-30  # since it's just pass-through
                    flowcolor = "black"  # to highlight it
                if constr not in self.constr_name:
                    source = "(%s)" % ("abcdefgijklmnopqrstuvwxyzABCDEFGHIJKLM"
                                       "NOPQRSTUVWXYZ"[self.counter.next()])
                    self.constr_name[constr] = source
                else:
                    source = self.constr_name[constr]
                self.links.append({"target": target, "source": source,
                                   "value": flowvalue,
                                   "color": flowcolor})
                if printing:
                    print ("%s adds %+.3g to the overall sensitivity of %s"
                           % (source, value, vk))
                    print source, "is", constr.str_without("units"), "\n"

    def diagram(self, variables=None, flowright=False, width=900, height=400,
                top=0, bottom=0, left=100, right=25):
        self.counter = Count()
        self.links = []
        if not variables:
            self.constrlinks(self.model)
        else:
            if not hasattr(variables, "__len__"):
                variables = [variables]
            for var in variables:
                self.varlinks(self.model, var.key)
        if flowright:
            r, l = margins["right"], margins["left"]
            margins["left"], margins["right"] = r, l
        else:
            for link in self.links:
                link["source"], link["target"] = link["target"], link["source"]
        margins = dict(top=top, bottom=bottom, left=left, right=right)
        return SankeyWidget(nodes=self.nodes, links=self.links,
                            margins=margins, width=width, height=height)

    @classmethod
    def of_vars_in_most_constraints(cls, model, minflow=0.01):
        linkcount = {}
        for key in model.v_ss:
            s = Sankey(model, key)
            s.varlinks(s.model, printing=False)
            if any(l["value"] > minflow for l in s.links):
                linkcount[key] = (s.counter.next(), s)
            s.links = []
        return [v[1] for k, v in
                sorted(linkcount.items(), key=lambda t: t[1][0], reverse=True)]

    @classmethod
    def of_highest_flow_vars(cls, model, minflow=0.01):
        linkcount = {}
        for key in model.v_ss:
            s = Sankey(model, key)
            s.varlinks(s.model, printing=False)
            maxflow = max(l["value"] for l in s.links)
            if maxflow > minflow:
                linkcount[key] = (maxflow, s)
            s.links = []
        return [v[1] for k, v in
                sorted(linkcount.items(), key=lambda t: t[1][0], reverse=True)]

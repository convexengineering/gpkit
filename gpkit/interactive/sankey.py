from ipysankeywidget import SankeyWidget
from gpkit import ConstraintSet, Model
from gpkit.small_classes import Count
from gpkit import GPCOLORS


def getcolor(value):
    if abs(value) < 1e-7:
        return "#cfcfcf"
    return GPCOLORS[1 if value > 0 else 0]


class Sankey(object):
    def __init__(self, model, var=None):
        self.links = []
        self.counter = None
        self.model = model
        self.var = var

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

    def varlinks(self, constrset, target=None, printing=True):
        if target is None:  # set final target as the variable itself
            value = constrset.v_ss[self.var.key] or 1e-30  # if it's zero
            target = constrset.name or "[Model]"
            if constrset.num:
                target += ".%i" % constrset.num
            source = (self.var.key.str_without(["models"])
                      + self.var.key.unitstr(into=" [%s]", dimless=" [-]"))
            self.counter = Count()
            self.links.append({"target": source, "source": target,
                               "value": abs(value), "color": getcolor(value)})
            if self.var.key in self.model.solution["sensitivities"]["cost"]:
                cost_senss = self.model.solution["sensitivities"]["cost"]
                value = cost_senss[self.var.key]
                self.links.append({"target": source, "source": "(objective)",
                                   "value": abs(value),
                                   "color": getcolor(value)})
                if printing:
                    print ("(objective) adds %+.3g to the overall sensitivity"
                           " of %s" % (value, self.var.key))
                    print "(objective) is", self.model.cost, "\n"
        for constr in constrset:
            if self.var.key not in constr.v_ss:
                continue
            value = constr.v_ss[self.var.key] or 1e-30
            # TODO: add filter-by-abs argument?
            if isinstance(constr, ConstraintSet):
                if isinstance(constr, Model):
                    source = constr.name
                    source += ".%i" % constr.num if constr.num else ""
                    self.links.append({"target": target, "source": source,
                                       "value": abs(value),
                                       "color": getcolor(value)})
                    self.varlinks(constr, source, printing)
                else:
                    self.varlinks(constr, target, printing)
            else:
                source = "(%s)" % ("abcdefgijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ"
                                   "RSTUVWXYZ"[self.counter.next()])
                self.links.append({"target": target, "source": source,
                                   "value": abs(value),
                                   "color": getcolor(value)})
                if printing:
                    print ("%s adds %+.3g to the overall sensitivity of %s"
                           % (source, value, self.var.key))
                    print source, "is", constr.str_without("units"), "\n"

    def widget(self, flowright=False, width=900, height=400, **kwargs):
        margins = dict(top=0, bottom=0, left=100, right=25)
        margins.update(kwargs)
        if self.var:
            self.varlinks(self.model)
        else:
            self.constrlinks(self.model)
        if flowright:
            r, l = margins["right"], margins["left"]
            margins["left"], margins["right"] = r, l
        else:
            for link in self.links:
                link["source"], link["target"] = link["target"], link["source"]
        return SankeyWidget(links=self.links, margins=margins,
                            width=width, height=height)

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

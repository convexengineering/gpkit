"implements Sankey"
from collections import defaultdict
from collections.abc import Iterable
import numpy as np
from ipywidgets import Layout
from ipysankeywidget import SankeyWidget  # pylint: disable=import-error
from ..repr_conventions import lineagestr
from .. import Model, GPCOLORS
from ..constraints.array import ArrayConstraint


INSENSITIVE = 1e-2

def isnamedmodel(constraint):
    "Checks if a constraint is a named model"
    return (isinstance(constraint, Model)
            and constraint.__class__.__name__ != "Model")

def getcolor(value):
    "color scheme for sensitivities"
    if abs(value) < INSENSITIVE:
        return "#cfcfcf"
    return GPCOLORS[0 if value < 0 else 1]

class Sankey:
    "diagrams of sensitivity flow"
    maxdepth = np.inf
    maxlinks = 50

    def __init__(self, solution, constraintset, leftlabel=None):
        self.csenss = solution["sensitivities"]["constraints"]
        self.cset = constraintset
        if leftlabel is None:
            leftlabel = lineagestr(self.cset) or self.cset.__class__.__name__
        self.leftlabel = leftlabel
        self.links = defaultdict(float)
        self.links_to_target = defaultdict(int)
        self.nodes = {}

    def add_node(self, target, title, type):
        self.links_to_target[target] += 1
        node = "%s.%04i" % (target, self.links_to_target[target])
        self.nodes[node] = {"id": node, "title": title, type: True}
        return node

    # pylint: disable=too-many-branches
    def link(self, cset, target, var, depth=0, subarray=False, named=False):
        "adds links of a given constraint set to self.links"
        total_sens = 0
        switchedtarget = False
        if not named and isnamedmodel(cset) and 0 < depth <= self.maxdepth:
            switchedtarget = target
            target = self.add_node(target, cset.lineage[-1][0], "namedmodel")
            depth += 1
        elif depth == 0:
            depth += 1  # to make top-level named models look right
        elif isinstance(cset, ArrayConstraint) and cset.constraints.size > 1:
            switchedtarget = target
            cstr = cset.str_without(["lineage", "units"])
            cstr = cstr.replace("[:]", "")  # implicitly vectors
            label = cstr if len(cstr) <= 30 else "%s ..." % cstr[:30]
            target = self.add_node(target, label, "constraint")
            subarray = True
        if getattr(cset, "idxlookup", None):
            cset = {k: cset[i] for k, i in cset.idxlookup.items()}
        if isinstance(cset, dict):
            for label, c in cset.items():
                if depth > self.maxdepth:
                    subtotal_sens = self.link(c, target, var, depth, subarray)
                else:
                    source = self.add_node(target, label, "constraintlabel")
                    subtotal_sens = self.link(c, source, var, depth+1,
                                              subarray, isnamedmodel(c))
                    self.links[source, target] += subtotal_sens
                total_sens += subtotal_sens
        elif isinstance(cset, Iterable):
            for c in cset:
                subtotal_sens = self.link(c, target, var, depth, subarray)
                total_sens += subtotal_sens
        else:
            if var is None and cset in self.csenss:
                total_sens = -abs(self.csenss[cset])
            elif var is not None and var.key in cset.v_ss:
                total_sens = cset.v_ss[var.key] or 1e-30  # just an epsilon
            if subarray:
                source = self.add_node(target, "", "subarray")
            else:
                cstr = cset.str_without(["lineage", "units"])
                label = cstr if len(cstr) <= 30 else "%s ..." % cstr[:30]
                source = self.add_node(target, label, "constraint")
            self.links[source, target] = total_sens
        if switchedtarget:
            self.links[target, switchedtarget] += total_sens
        return total_sens

    def filterlinks(self, label, function):
        "If over maxlinks, removes links that do not match criteria."
        if len(self.links) > self.maxlinks:
            if label:
                print("Links exceed link limit (%s > %s); skipping %s."
                      % (len(self.links), self.maxlinks, label))
            self.links = {(s, t): v for (s, t), v in self.links.items()
                          if function(s, t, v)}

    # pylint: disable=too-many-locals
    def diagram(self, variable=None, *, maxdepth=None,
                width=900, height=400, top=0, bottom=0, left=120, right=120):
        "creates links and an ipython widget to show them"
        margins = dict(top=top, bottom=bottom, left=left, right=right)
        if maxdepth is not None:
            self.maxdepth = maxdepth  # NOTE: side effects

        total_sens = self.link(self.cset, self.leftlabel, variable)
        if variable:
            self.links[self.leftlabel, str(variable)] = total_sens
            self.nodes[str(variable)] = {"id": str(variable),
                                         "title": str(variable)}
        self.nodes[self.leftlabel] = {"id": self.leftlabel,
                                      "title": self.leftlabel}

        self.filterlinks("", lambda s, t, v: v)
        self.filterlinks("constraints inside arrays",
                         lambda s, t, v: "subarray" not in self.nodes[s])
        self.filterlinks("all constraints",
                         lambda s, t, v: "constraint" not in self.nodes[s])
        self.filterlinks("constraint labels",
                         lambda s, t, v: "constraintlabel" not in self.nodes[s])

        links, nodes = [], []
        for (source, target), value in self.links.items():
            if source in self.nodes:
                nodes.append({"id": self.nodes[source]["id"],
                              "title": self.nodes[source]["title"]})
            links.append({"source": source, "target": target,
                          "value": abs(value),
                          "color": getcolor(value), "title": "%+.2g" % value})
        out = SankeyWidget(nodes=nodes, links=links,
                           margins=margins,
                           layout=Layout(width=str(width), height=str(height)))
        filename = self.leftlabel
        if variable:
            filename += "_" + variable.key.name
        out.auto_save_png(filename + ".png")
        # clean up side effects
        self.links = defaultdict(float)
        self.maxdepth = Sankey.maxdepth
        return out

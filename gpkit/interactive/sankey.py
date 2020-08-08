"implements Sankey"
from collections import defaultdict, Iterable
import numpy as np
from ipywidgets import Layout
from ipysankeywidget import SankeyWidget  # pylint: disable=import-error
from ..repr_conventions import lineagestr
from .. import Model, GPCOLORS


INSENSITIVE = 1e-2

def isnamedmodel(constraint):
    "Checks if a constraint is a named model"
    return isinstance(constraint, Model) and type(constraint) is not Model

def getcolor(value):
    "color scheme for sensitivities"
    if abs(value) < INSENSITIVE:
        return "#cfcfcf"
    return GPCOLORS[0 if value < 0 else 1]


class Sankey:
    "diagrams of sensitivity flow"
    maxdepth = np.inf
    constraintlabels = True

    def __init__(self, solution, constraintset, leftlabel=None):
        self.csenss = solution["sensitivities"]["constraints"]
        self.cset = constraintset
        if leftlabel is None:
            leftlabel = lineagestr(self.cset) or self.cset.__class__.__name__
        self.leftlabel = leftlabel
        self.links = defaultdict(float)
        self.nodes = []

    def constrlinks(self, constrset, target, i=0, depth=0, named=False):
        "adds links of a given constraint set to self.links"
        total_sens = 0
        switchedtarget = False
        if not named and isnamedmodel(constrset) and 0 < depth <= self.maxdepth:
            switchedtarget = target
            name, _ = constrset.lineage[-1]
            target = "%s.%03i.%s" % (switchedtarget, i, name)
            self.nodes.append({"id": target, "title": name})
            depth += 1
        elif depth == 0:
            depth += 1  # to make top-level named models look right
        if getattr(constrset, "idxlookup", None):
            constrset = {k: constrset[i]
                         for k, i in constrset.idxlookup.items()}
        if isinstance(constrset, dict):
            for i, (label, constr) in enumerate(constrset.items()):
                if depth > self.maxdepth or not self.constraintlabels:
                    subtotal_sens = self.constrlinks(constr, target, i, depth)
                else:
                    source = "%s.%03i.%s" % (target, i, label)
                    self.nodes.append({"id": source, "title": label})
                    subtotal_sens = self.constrlinks(constr, source, i, depth+1,
                                                     named=isnamedmodel(constr))
                    self.links[source, target] += subtotal_sens
                total_sens += subtotal_sens
        elif isinstance(constrset, Iterable):
            for i, constr in enumerate(constrset):
                total_sens += self.constrlinks(constr, target, i, depth)
        elif constrset in self.csenss:
            total_sens = -abs(self.csenss[constrset])
        if switchedtarget:
            self.links[target, switchedtarget] += total_sens
        return total_sens

    def varlinks(self, constrset, target, key, i=0, depth=1):
        "adds links of a given constraint set to self.links"
        total_sens = None
        switchedtarget = False
        if (isinstance(constrset, Model) and type(constrset) is not Model
                and depth <= self.maxdepth):
            switchedtarget = target
            name, _ = constrset.lineage[-1]
            target = "%s.%03i.%s" % (switchedtarget, i, name)
            self.nodes.append({"id": target, "title": name})
            depth += 1
        if getattr(constrset, "idxlookup", None):
            constrset = {k: constrset[i]
                         for k, i in constrset.idxlookup.items()}
        if isinstance(constrset, dict):
            for i, (label, constr) in enumerate(constrset.items()):
                if depth <= self.maxdepth:
                    source = "%s.%03i.%s" % (target, i, label)
                    self.nodes.append({"id": source, "title": label})
                    subtotal_sens = self.varlinks(constr, source, key, i, depth+1)
                    if subtotal_sens is not None:
                        self.links[source, target] += subtotal_sens
                else:
                    subtotal_sens = self.varlinks(constr, target, key, i, depth)
                if subtotal_sens is not None:
                    if total_sens is None:
                        total_sens = 0
                    total_sens += subtotal_sens
        elif isinstance(constrset, Iterable):
            for i, constr in enumerate(constrset):
                subtotal_sens = self.varlinks(constr, target, key, i, depth)
                if subtotal_sens is not None:
                    if total_sens is None:
                        total_sens = 0
                    total_sens += subtotal_sens
        elif key in constrset.v_ss:
            total_sens = constrset.v_ss[key]
        if switchedtarget and total_sens is not None:
                self.links[target, switchedtarget] += total_sens
        return total_sens

    def diagram(self, variable=None, *, flowright=False, width=900, height=400,
                top=0, bottom=0, left=120, right=55, maxdepth=None,
                constraintlabels=None):
        "creates links and an ipython widget to show them"
        margins = dict(top=top, bottom=bottom, left=left, right=right)
        if flowright:
            margins = dict(top=top, bottom=bottom, left=right, right=left)
        if maxdepth is not None:
            self.maxdepth = maxdepth  # NOTE: side effects
        if constraintlabels is not None:
            self.constraintlabels = constraintlabels  # NOTE: side effects
        if variable is None:
            self.constrlinks(self.cset, self.leftlabel)
        else:
            self.links[self.leftlabel, str(variable)] = \
                self.varlinks(self.cset, self.leftlabel, variable.key)
        links = []
        maxflow = np.abs(list(self.links.values())).max()
        for (source, target), value in self.links.items():
            if not flowright:  # reverse by default, sigh
                source, target = target, source
            links.append({"source": source, "target": target,
                          "value": max(abs(value), maxflow/1e5),
                          "title": "%+.2g" % value, "color": getcolor(value)})
        out = SankeyWidget(nodes=self.nodes, links=links,
                           layout=Layout(width=str(width),
                                         height=str(height)),
                           margins=margins)
        # clean up side effects
        self.links = defaultdict(float)
        self.nodes = []
        self.maxdepth = Sankey.maxdepth
        self.constraintlabels = Sankey.constraintlabels
        return out

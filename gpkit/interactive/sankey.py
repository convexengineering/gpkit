"implements Sankey"
from collections import defaultdict
from collections.abc import Iterable
import numpy as np
from ipywidgets import Layout
from ipysankeywidget import SankeyWidget  # pylint: disable=import-error
from ..repr_conventions import lineagestr
from .. import Model, GPCOLORS


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
    constraintlabels = True

    def __init__(self, solution, constraintset, leftlabel=None):
        self.csenss = solution["sensitivities"]["constraints"]
        self.cset = constraintset
        if leftlabel is None:
            leftlabel = lineagestr(self.cset) or self.cset.__class__.__name__
        self.leftlabel = leftlabel
        self.links = defaultdict(float)
        self.nodes = []

    # pylint: disable=too-many-branches
    def link(self, constrset, target, var=None, i=0, depth=0, named=False):
        "adds links of a given constraint set to self.links"
        total_sens = None
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
            for i, (label, c) in enumerate(constrset.items()):  # pylint: disable=redefined-argument-from-local
                if depth > self.maxdepth or not self.constraintlabels:
                    subtotal_sens = self.link(c, target, var, i, depth)
                else:
                    source = "%s.%03i.%s" % (target, i, label)
                    self.nodes.append({"id": source, "title": label})
                    subtotal_sens = self.link(c, source, var, i, depth+1,
                                              named=isnamedmodel(c))
                    if subtotal_sens is not None:
                        self.links[source, target] += subtotal_sens
                if subtotal_sens is not None:
                    if total_sens is None:
                        total_sens = 0
                    total_sens += subtotal_sens
        elif isinstance(constrset, Iterable):
            for i, c in enumerate(constrset):  # pylint: disable=redefined-argument-from-local
                subtotal_sens = self.link(c, target, var, i, depth)
                if subtotal_sens is not None:
                    if total_sens is None:
                        total_sens = 0
                    total_sens += subtotal_sens
        elif var is None and constrset in self.csenss:
            total_sens = -abs(self.csenss[constrset])
        elif var is not None and var.key in constrset.v_ss:
            total_sens = constrset.v_ss[var.key]
        if switchedtarget and total_sens is not None:
            self.links[target, switchedtarget] += total_sens
        return total_sens

    # pylint: disable=too-many-locals
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

        total_sens = self.link(self.cset, self.leftlabel, variable)
        if variable is not None:
            self.links[self.leftlabel, str(variable)] = total_sens

        maxflow = np.abs(list(self.links.values())).max()
        links = []
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

"implements Sankey"
import os
import re
from collections import defaultdict
from collections.abc import Iterable
import numpy as np
from ipywidgets import Layout
from ipysankeywidget import SankeyWidget
from ..repr_conventions import lineagestr, unitstr
from .. import Model, GPCOLORS
from ..constraints.array import ArrayConstraint


INSENSITIVE = 1e-2
EPS = 1e-10

def isnamedmodel(constraint):
    "Checks if a constraint is a named model"
    return (isinstance(constraint, Model)
            and constraint.__class__.__name__ != "Model")

def getcolor(value):
    "color scheme for sensitivities"
    if abs(value) < INSENSITIVE:
        return "#cfcfcf"
    return GPCOLORS[0 if value < 0 else 1]

def cleanfilename(string):
    "Parses string into valid filename"
    return re.sub(r"\\/?|\"><:\*", "_", string)  # Replace invalid with _


# pylint: disable=too-many-instance-attributes
class Sankey:
    "Return Jupyter diagrams of sensitivity flow"
    minsenss = 0
    maxlinks = 20
    showconstraints = True
    last_top_node = None

    def __init__(self, solution, constraintset, csetlabel=None):
        self.solution = solution
        self.csenss = solution["sensitivities"]["constraints"]
        self.cset = constraintset
        if csetlabel is None:
            csetlabel = lineagestr(self.cset) or self.cset.__class__.__name__
        self.csetlabel = csetlabel
        self.links = defaultdict(float)
        self.links_to_target = defaultdict(int)
        self.nodes = {}

    def add_node(self, target, title, tag=None):
        "adds nodes of a given target, title, and tag to self.nodes"
        self.links_to_target[target] += 1
        node = "%s.%04i" % (target, self.links_to_target[target])
        self.nodes[node] = {"id": node, "title": title, tag: True}
        return node

    def linkfixed(self, cset, target):
        "adds fixedvariable links as if they were (array)constraints"
        fixedvecs = {}
        total_sens = 0
        for vk in sorted(cset.unique_varkeys, key=str):
            if vk not in self.solution["constants"]:
                continue
            if vk.veckey and vk.veckey not in fixedvecs:
                vecval = self.solution["constants"][vk.veckey]
                firstval = vecval.flatten()[0]
                if vecval.shape and (firstval == vecval).all():
                    label = "%s = %.4g %s" % (vk.veckey.name, firstval,
                                              unitstr(vk.veckey))
                    fixedvecs[vk.veckey] = self.add_node(target, label,
                                                         "constraint")
            abs_var_sens = -abs(self.solution["sensitivities"] \
                                             ["constants"].get(vk, EPS))
            if np.isnan(abs_var_sens):
                abs_var_sens = EPS
            label = "%s = %.4g %s" % (vk.str_without(["lineage"]),
                                      self.solution["variables"][vk],
                                      unitstr(vk))
            if vk.veckey in fixedvecs:
                vectarget = fixedvecs[vk.veckey]
                source = self.add_node(vectarget, label, "subarray")
                self.links[source, vectarget] = abs_var_sens
                self.links[vectarget, target] += abs_var_sens
            else:
                source = self.add_node(target, label, "constraint")
                self.links[source, target] = abs_var_sens
            total_sens += abs_var_sens
        return total_sens

    # pylint: disable=too-many-branches
    def link(self, cset, target, vk, *, labeled=False, subarray=False):
        "adds links of a given constraint set to self.links"
        total_sens = 0
        switchedtarget = False
        if not labeled and isnamedmodel(cset):
            if cset is not self.cset:  # top-level, no need to switch targets
                switchedtarget = target
                target = self.add_node(target, cset.lineage[-1][0])
            if vk is None:
                total_sens += self.linkfixed(cset, target)
        elif isinstance(cset, ArrayConstraint) and cset.constraints.size > 1:
            switchedtarget = target
            cstr = cset.str_without(["lineage", "units"]).replace("[:]", "")
            label = cstr if len(cstr) <= 30 else "%s ..." % cstr[:30]
            target = self.add_node(target, label, "constraint")
            subarray = True
        if getattr(cset, "idxlookup", None):
            cset = {k: cset[i] for k, i in cset.idxlookup.items()}
        if isinstance(cset, dict):
            for label, c in cset.items():
                source = self.add_node(target, label)
                subtotal_sens = self.link(c, source, vk, labeled=True)
                self.links[source, target] += subtotal_sens
                total_sens += subtotal_sens
        elif isinstance(cset, Iterable):
            for c in cset:
                total_sens += self.link(c, target, vk, subarray=subarray)
        else:
            if vk is None and cset in self.csenss:
                total_sens = -abs(self.csenss[cset]) or -EPS
            elif vk is not None:
                if cset.v_ss is None:
                    if vk in cset.varkeys:
                        total_sens = EPS
                elif vk in cset.v_ss:
                    total_sens = cset.v_ss[vk] or EPS
            if not labeled:
                cstr = cset.str_without(["lineage", "units"])
                label = cstr if len(cstr) <= 30 else "%s ..." % cstr[:30]
                tag = "subarray" if subarray else "constraint"
                source = self.add_node(target, label, tag)
                self.links[source, target] = total_sens
        if switchedtarget:
            self.links[target, switchedtarget] += total_sens
        return total_sens

    def filter(self, links, function, forced=False):
        "If over maxlinks, removes links that do not match criteria."
        if len(links) > self.maxlinks or forced:
            for (s, t), v in list(links.items()):
                if not function(s, t, v):
                    del links[(s, t)]

    # pylint: disable=too-many-locals
    def diagram(self, variable=None, varlabel=None, *, minsenss=0, maxlinks=20,
                top=0, bottom=0, left=230, right=140, width=1000, height=400,
                showconstraints=True):
        "creates links and an ipython widget to show them"
        margins = dict(top=top, bottom=bottom, left=left, right=right)
        self.minsenss = minsenss
        self.maxlinks = maxlinks
        self.showconstraints = showconstraints

        self.solution.set_necessarylineage()

        if variable:
            variable = variable.key
            if not varlabel:
                varlabel = str(variable)
                if len(varlabel) > 20:
                    varlabel = variable.str_without(["lineage"])
            self.nodes[varlabel] = {"id": varlabel, "title": varlabel}
            csetnode = self.add_node(varlabel, self.csetlabel)
            if variable in self.solution["sensitivities"]["cost"]:
                costnode = self.add_node(varlabel, "[cost function]")
                self.links[costnode, varlabel] = \
                    self.solution["sensitivities"]["cost"][variable]
        else:
            csetnode = self.csetlabel
            self.nodes[self.csetlabel] = {"id": self.csetlabel,
                                          "title": self.csetlabel}
        total_sens = self.link(self.cset, csetnode, variable)
        if variable:
            self.links[csetnode, varlabel] = total_sens

        links, nodes = self._links_and_nodes()
        out = SankeyWidget(nodes=nodes, links=links, margins=margins,
                           layout=Layout(width=str(width), height=str(height)))

        filename = self.csetlabel
        if variable:
            filename += "_%s" % variable
        if not os.path.isdir("sankey_autosaves"):
            os.makedirs("sankey_autosaves")
        filename = "sankey_autosaves" + os.path.sep + cleanfilename(filename)
        out.auto_save_png(filename + ".png")
        out.auto_save_svg(filename + ".svg")
        out.on_node_clicked(self.onclick)
        out.on_link_clicked(self.onclick)

        self.solution.set_necessarylineage(clear=True)
        return out

    def _links_and_nodes(self, top_node=None):
        links = self.links.copy()
        # filter if...not below the chosen top node
        if top_node is not None:
            self.filter(links, lambda s, t, v: top_node in s or top_node in t,
                        forced=True)
        # ...below minimum sensitivity
        self.filter(links, lambda s, t, v: abs(v) > self.minsenss, forced=True)
        if not self.showconstraints:
            # ...is a constraint or subarray and we're not showing those
            self.filter(links, lambda s, t, v:
                        ("constraint" not in self.nodes[s]
                         and "subarray" not in self.nodes[s]), forced=True)
        # ...is a subarray and we still have too many links
        self.filter(links, lambda s, t, v: "subarray" not in self.nodes[s])
        # ...is an insensitive constraint and we still have too many links
        self.filter(links, lambda s, t, v: ("constraint" not in self.nodes[s]
                                            or abs(v) > INSENSITIVE))
        # ...is at culldepth, repeating up to a relative depth of 1 or 2
        culldepth = max(node.count(".") for node in self.nodes) - 1
        mindepth = 1 if not top_node else top_node.count(".") + 1
        while len(links) > self.maxlinks and culldepth > mindepth:
            self.filter(links, lambda s, t, v: culldepth > s.count("."))
            culldepth -= 1
        # ...is a constraint and we still have too many links
        self.filter(links, lambda s, t, v: "constraint" not in self.nodes[s])

        linkslist, nodes, nodeset = [], [], set()
        for (source, target), value in links.items():
            if source == top_node:
                nodes.append({"id": self.nodes[target]["id"],
                              "title": "⟶ %s" % self.nodes[target]["title"]})
                nodeset.add(target)
            for node in [source, target]:
                if node not in nodeset:
                    nodes.append({"id": self.nodes[node]["id"],
                                  "title": self.nodes[node]["title"]})
                    nodeset.add(node)
            linkslist.append({"source": source, "target": target,
                              "value": abs(value), "color": getcolor(value),
                              "title": "%+.2g" % value})
        return linkslist, nodes

    def onclick(self, sankey, node_or_link):
        "Callback function for when a node or link is clicked on."
        if node_or_link is not None:
            if "id" in node_or_link:  # it's a node
                top_node = node_or_link["id"]
            else:  # it's a link
                top_node = node_or_link["source"]
            if self.last_top_node != top_node:
                sankey.links, sankey.nodes = self._links_and_nodes(top_node)
                sankey.send_state()
                self.last_top_node = top_node

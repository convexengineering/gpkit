"implements Sankey"
from collections import defaultdict
from collections.abc import Iterable
from ipywidgets import Layout
from ipysankeywidget import SankeyWidget  # pylint: disable=import-error
from ..repr_conventions import lineagestr, unitstr
from .. import Model, GPCOLORS
from ..constraints.array import ArrayConstraint
from ..tests.from_paths import clean as clean_str


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


# pylint: disable=too-many-instance-attributes
class Sankey:
    "Return Jupyter diagrams of sensitivity flow"
    minsenss = 0
    maxlinks = 30

    def __init__(self, solution, constraintset, csetlabel=None):
        self.solution = solution
        self.csenss = solution["sensitivities"]["constraints"]
        self.vsenss = solution["sensitivities"]["variables"]
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

    # pylint: disable=too-many-branches, too-many-locals
    def link(self, cset, target, var, depth=0, labeled=False, subarray=False):
        "adds links of a given constraint set to self.links"
        total_sens = 0
        switchedtarget = False
        if not labeled and isnamedmodel(cset):
            if depth:
                switchedtarget = target
                target = self.add_node(target, cset.lineage[-1][0])
            depth += 1
            if var is None:
                for vk in sorted(cset.unique_varkeys, key=str):
                    if vk not in self.solution["variables"]:
                        continue
                    abs_var_sens = -abs(self.vsenss.get(vk, 0))
                    label = "%s = %.4g %s" % (
                        vk.str_without(["lineage"]),
                        self.solution["variables"][vk], unitstr(vk))
                    source = self.add_node(target, label, "constraint")
                    self.links[source, target] = abs_var_sens
                    total_sens += abs_var_sens
        elif isinstance(cset, ArrayConstraint) and cset.constraints.size > 1:
            switchedtarget = target
            cstr = cset.str_without(["lineage", "units"])
            cstr = cstr.replace("[:]", "")  # implicitly vectors
            label = cstr if len(cstr) <= 30 else "%s ..." % cstr[:30]
            target = self.add_node(target, label, "constraint")
            subarray = True
        if depth == 0:
            depth += 1
        if getattr(cset, "idxlookup", None):
            cset = {k: cset[i] for k, i in cset.idxlookup.items()}
        if isinstance(cset, dict):
            for label, c in cset.items():
                source = self.add_node(target, label)
                subtotal_sens = self.link(c, source, var, depth+1, True)
                self.links[source, target] += subtotal_sens
                total_sens += subtotal_sens
        elif isinstance(cset, Iterable):
            for c in cset:
                subtotal_sens = self.link(c, target, var, depth, subarray)
                total_sens += subtotal_sens
        else:
            if var is None and cset in self.csenss:
                total_sens = -abs(self.csenss[cset]) or -EPS
            elif var is not None and var.key in cset.v_ss:
                total_sens = cset.v_ss[var.key] or EPS
            if not labeled:
                cstr = cset.str_without(["lineage", "units"])
                label = cstr if len(cstr) <= 30 else "%s ..." % cstr[:30]
                source = self.add_node(target, label,
                                       "subarray" if subarray else "constraint")
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

    def diagram(self, variable=None, varlabel=None, *,
                top=0, bottom=0, left=200, right=120, width=900, height=400,
                minsenss=0, maxlinks=30, maxdepth=None):
        "creates links and an ipython widget to show them"
        margins = dict(top=top, bottom=bottom, left=left, right=right)
        self.minsenss = minsenss
        self.maxlinks = maxlinks

        if variable:
            if not varlabel:
                varlabel = str(variable)
                if len(varlabel) > 20:
                    varlabel = variable.str_without(["lineage"])
            self.nodes[varlabel] = {"id": varlabel, "title": varlabel}
            csetnode = self.add_node(varlabel, self.csetlabel)
            if variable.key in self.solution["sensitivities"]["cost"]:
                costnode = self.add_node(varlabel, "[cost function]")
                self.links[costnode, varlabel] = \
                    self.solution["sensitivities"]["cost"][variable.key]
        else:
            csetnode = self.csetlabel
            self.nodes[self.csetlabel] = {"id": self.csetlabel,
                                          "title": self.csetlabel}
        total_sens = self.link(self.cset, csetnode, variable)
        if variable:
            self.links[csetnode, varlabel] = total_sens

        links, nodes = self._links_and_nodes(maxdepth)
        out = SankeyWidget(nodes=nodes, links=links, margins=margins,
                           layout=Layout(width=str(width), height=str(height)))

        filename = self.csetlabel
        if variable:
            filename += "_" + varlabel
        out.auto_save_png(clean_str(filename) + ".png")
        out.on_node_clicked(self.onclick)
        out.on_link_clicked(self.onclick)
        return out

    def _links_and_nodes(self, maxdepth=None, top_node=None):
        links = self.links.copy()
        # filter if...not below the chosen top node
        if top_node is not None:
            self.filter(links, lambda s, t, v: top_node in s or top_node in t,
                        forced=True)
        # ...below the chosen maxdepth
        if maxdepth is not None:
            self.filter(links, lambda s, t, v: s.count(".") <= maxdepth,
                        forced=True)
        # ...below minimum sensitivity
        self.filter(links, lambda s, t, v: abs(v) > self.minsenss, forced=True)
        # ...is a subarray and we still have too many links
        self.filter(links, lambda s, t, v: "subarray" not in self.nodes[s])
        # ...is an insensitive constraint and we still have too many links
        self.filter(links, lambda s, t, v: ("constraint" not in self.nodes[s]
                                            or abs(v) > INSENSITIVE))
        # ...is a constraint and we still have too many links
        self.filter(links, lambda s, t, v: "constraint" not in self.nodes[s])
        # ...is at culldepth, repeating up to a relative depth of 1 or 2
        culldepth = max(node.count(".") for node in self.nodes) - 1
        mindepth = 2 if not top_node else top_node.count(".") + 1
        while len(links) > self.maxlinks and culldepth > mindepth:
            self.filter(links, lambda s, t, v: culldepth > s.count("."))
            culldepth -= 1

        linkslist, nodes, nodeset = [], [], set()
        for (source, target), value in links.items():
            if source == top_node:
                nodes.append({"id": self.nodes[target]["id"],
                              "title": "⟶ %s" % self.nodes[target]["title"]})
                nodeset.add(target)
            else:
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
        if node_or_link is None:
            return
        if "id" in node_or_link:  # it's a node
            top_node = node_or_link["id"]
        else:  # it's a link
            top_node = node_or_link["source"]
        sankey.links, sankey.nodes = \
            self._links_and_nodes(top_node=top_node)
        sankey.send_state()

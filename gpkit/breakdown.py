#TODO: cleanup weird conditionals
#      add conversions to plotly/sankey

# pylint: skip-file
import string
from collections import defaultdict, namedtuple, Counter
from gpkit.nomials import Monomial, Posynomial, Variable
from gpkit.nomials.map import NomialMap
from gpkit.small_scripts import mag
from gpkit.small_classes import FixedScalar, HashVector
from gpkit.exceptions import DimensionalityError
from gpkit.repr_conventions import unitstr as get_unitstr
from gpkit.varkey import VarKey
import numpy as np


Tree = namedtuple("Tree", ["key", "value", "branches"])
Transform = namedtuple("Transform", ["factor", "power", "origkey"])
def is_factor(key):
    return (isinstance(key, Transform) and key.power == 1)
def is_power(key):
    return (isinstance(key, Transform) and key.power != 1)

def get_free_vks(posy, solution):
    "Returns all free vks of a given posynomial for a given solution"
    return set(vk for vk in posy.vks if vk not in solution["constants"])

def get_model_breakdown(solution):
    breakdowns = {"|sensitivity|": 0}
    for modelname, senss in solution["sensitivities"]["models"].items():
        senss = abs(senss)  # for those monomial equalities
        *namespace, name = modelname.split(".")
        subbd = breakdowns
        subbd["|sensitivity|"] += senss
        for parent in namespace:
            if parent not in subbd:
                subbd[parent] = {parent: {}}
            subbd = subbd[parent]
            if "|sensitivity|" not in subbd:
                subbd["|sensitivity|"] = 0
            subbd["|sensitivity|"] += senss
        subbd[name] = {"|sensitivity|": senss}
    # print(breakdowns["HyperloopSystem"]["|sensitivity|"])
    breakdowns = {"|sensitivity|": 0}
    for constraint, senss in solution["sensitivities"]["constraints"].items():
        senss = abs(senss)  # for those monomial
        if senss <= 1e-5:
            continue
        subbd = breakdowns
        subbd["|sensitivity|"] += senss
        for parent in constraint.lineagestr().split("."):
            if parent == "":
                continue
            if parent not in subbd:
                subbd[parent] = {}
            subbd = subbd[parent]
            if "|sensitivity|" not in subbd:
                subbd["|sensitivity|"] = 0
            subbd["|sensitivity|"] += senss
        # treat vectors as namespace
        constraint = constraint.str_without({"unnecessary lineage", "units", ":MAGIC:"+constraint.lineagestr()})
        subbd[constraint] = {"|sensitivity|": senss}
    for vk in solution.vks:
        if vk not in solution["sensitivities"]["variables"]:
            continue
        senss = abs(solution["sensitivities"]["variables"][vk])
        if hasattr(senss, "shape"):
            senss = np.nansum(senss)
        if senss <= 1e-5:
            continue
        subbd = breakdowns
        subbd["|sensitivity|"] += senss
        for parent in vk.lineagestr().split("."):
            if parent == "":
                continue
            if parent not in subbd:
                subbd[parent] = {}
            subbd = subbd[parent]
            if "|sensitivity|" not in subbd:
                subbd["|sensitivity|"] = 0
            subbd["|sensitivity|"] += senss
        # treat vectors as namespace (indexing vectors above)
        vk = vk.str_without({"lineage"}) + get_valstr(vk, solution, " = %s").replace(", fixed", "")
        subbd[vk] = {"|sensitivity|": senss}
    # TODO: track down in a live-solve environment why this isn't the same
    # print(breakdowns["HyperloopSystem"]["|sensitivity|"])
    return breakdowns

def crawl_modelbd(bd, name="Model"):
    tree = Tree(name, bd.pop("|sensitivity|"), [])
    for subname, subtree in sorted(bd.items(),
                                   key=lambda kv: -kv[1]["|sensitivity|"]):
        tree.branches.append(crawl_modelbd(subtree, subname))
    return tree

def divide_out_vk(vk, pow, lt, gt):
    hmap = NomialMap({HashVector({vk: 1}): 1.0})
    hmap.units = vk.units
    var = Monomial(hmap)**pow
    lt, gt = lt/var, gt/var
    lt.ast = gt.ast = None
    return lt, gt

# @profile
def get_breakdowns(solution):
    """Returns {key: (lt, gt, constraint)} for breakdown constrain in solution.

    A breakdown constraint is any whose "gt" contains a single free variable.

    (At present, monomial constraints check both sides as "gt")
    """
    breakdowns = defaultdict(list)
    beatout = defaultdict(set)
    for constraint, senss in sorted(solution["sensitivities"]["constraints"].items(), key=lambda kv: (-abs(kv[1]), str(kv[0]))):
        if abs(senss) <= 1e-5:  # only tight-ish ones
            continue
        if constraint.oper == ">=":
            gt, lt = (constraint.left, constraint.right)
        elif constraint.oper == "<=":
            lt, gt = (constraint.left, constraint.right)
        elif constraint.oper == "=":
            if senss > 0:  # l_over_r is more sensitive - see nomials/math.py
                lt, gt = (constraint.left, constraint.right)
            else:  # r_over_l is more sensitive - see nomials/math.py
                gt, lt = (constraint.left, constraint.right)
        if lt.any_nonpositive_cs or len(gt.hmap) > 1:
            continue  # no signomials  # TODO: approximate signomials at sol
        pos_gtvks = {vk for vk, pow in gt.exp.items() if pow > 0}
        if len(pos_gtvks) > 1:
            pos_gtvks &= get_free_vks(gt, solution)  # remove constants
        if len(pos_gtvks) == 1:
            chosenvk, = pos_gtvks
            breakdowns[chosenvk].append((lt, gt, constraint))
    for constraint, senss in sorted(solution["sensitivities"]["constraints"].items(), key=lambda kv: (-abs(kv[1]), str(kv[0]))):
        if abs(senss) <= 1e-5:  # only tight-ish ones
            continue
        if constraint.oper == ">=":
            gt, lt = (constraint.left, constraint.right)
        elif constraint.oper == "<=":
            lt, gt = (constraint.left, constraint.right)
        elif constraint.oper == "=":
            if senss > 0:  # l_over_r is more sensitive - see nomials/math.py
                lt, gt = (constraint.left, constraint.right)
            else:  # r_over_l is more sensitive - see nomials/math.py
                gt, lt = (constraint.left, constraint.right)
        if lt.any_nonpositive_cs or len(gt.hmap) > 1:
            continue  # no signomials  # TODO: approximate signomials at sol
        pos_gtvks = {vk for vk, pow in gt.exp.items() if pow > 0}
        if len(pos_gtvks) > 1:
            pos_gtvks &= get_free_vks(gt, solution)  # remove constants
        if len(pos_gtvks) != 1:  # we'll choose our favorite vk
            for vk, pow in gt.exp.items():
                if pow < 0:  # remove all non-positive
                    lt, gt = divide_out_vk(vk, pow, lt, gt)
            # bring over common factors from lt
            lt_pows = defaultdict(set)
            for exp in lt.hmap:
                for vk, pow in exp.items():
                    lt_pows[vk].add(pow)
            for vk, pows in lt_pows.items():
                if len(pows) == 1:
                    pow, = pows
                    if pow < 0:  # ...but only if they're positive
                        lt, gt = divide_out_vk(vk, pow, lt, gt)
            # don't choose something that's already been broken down
            candidatevks = {vk for vk in gt.vks if vk not in breakdowns}
            if candidatevks:
                vrisk = solution["sensitivities"]["variablerisk"]
                chosenvk, *_ = sorted(
                    candidatevks,
                    key=lambda vk: (-gt.exp[vk]*vrisk.get(vk, 0), str(vk))
                )
                for vk in gt.vks:
                    if vk is not chosenvk:
                        lt, gt = divide_out_vk(vk, pow, lt, gt)
                breakdowns[chosenvk].append((lt, gt, constraint))
    breakdowns = dict(breakdowns)  # remove the defaultdict-ness

    prevlen = None
    while len(BASICALLY_FIXED_VARIABLES) != prevlen:
        prevlen = len(BASICALLY_FIXED_VARIABLES)
        for key in breakdowns:
            if key not in BASICALLY_FIXED_VARIABLES:
                get_fixity(key, breakdowns, solution, BASICALLY_FIXED_VARIABLES)
    return breakdowns

BASICALLY_FIXED_VARIABLES = set()


def get_fixity(key, bd, solution, basically_fixed=set(), visited=set()):
    lt, gt, _ = bd[key][0]
    free_vks = get_free_vks(lt, solution).union(get_free_vks(gt, solution))
    for vk in free_vks:
        if vk is key or vk in BASICALLY_FIXED_VARIABLES:
            continue  # currently checking or already checked
        if vk not in bd:
            return  # a very free variable, can't even be broken down
        if vk in visited:
            return  # tried it before, implicitly it didn't work out
        # maybe it's basically fixed?
        visited.add(key)
        get_fixity(vk, bd, solution, basically_fixed, visited)
        if vk not in BASICALLY_FIXED_VARIABLES:
            return  # ...well, we tried
    basically_fixed.add(key)

SOLCACHE = {}
def solcache(solution, key):  # replaces solution(key)
    if key not in SOLCACHE:
        SOLCACHE[key] = solution(key)
    return SOLCACHE[key]

# @profile  # ~84% of total last check # TODO: remove
def crawl(key, bd, solution, basescale=1, permissivity=2, verbosity=0,
          visited_bdkeys=None, gone_negative=False):
    "Returns the tree of breakdowns of key in bd, sorting by solution's values"
    if key in bd:
        # TODO: do multiple if sensitivities are quite close?
        composition, keymon, constraint = bd[key][0]
    elif isinstance(key, Posynomial):
        composition = key
        keymon = None
    else:
        raise TypeError("the `key` argument must be a VarKey or Posynomial.")

    if visited_bdkeys is None:
        visited_bdkeys = set()
    if verbosity == 1:
        solution.set_necessarylineage()
    if verbosity:
        indent = verbosity-1  # HACK: a bit of overloading, here
        kvstr = "%s (%s)" % (key.str_without(["unnecessary lineage", "units"]),
                             get_valstr(key, solution))
        print("  "*indent + kvstr + ", which breaks down further")
        indent += 1
    orig_subtree = subtree = []
    tree = Tree(key, basescale, subtree)
    visited_bdkeys.add(key)
    if keymon is None:
        scale = solution(key)/basescale
    else:
        interesting_vks = {key}
        subkey, = interesting_vks
        power = keymon.exp[subkey]
        boring_vks = set(keymon.vks) - interesting_vks
        scale = solution(key)**power/basescale
        # TODO: make method that can handle both kinds of transforms
        if (power != 1 or boring_vks or mag(keymon.c) != 1
                or keymon.units != key.units):  # some kind of transform here
            units = 1
            exp = HashVector()
            for vk in interesting_vks:
                exp[vk] = keymon.exp[vk]
                if vk.units:
                    units *= vk.units**keymon.exp[vk]
            subhmap = NomialMap({exp: 1})
            try:
                subhmap.units = None if units == 1 else units
            except DimensionalityError:
                # pints was unable to divide a unit by itself bc
                #   it has terrible floating-point errors.
                #   so let's assume it isn't dimensionless
                #   even though it probably is
                subhmap.units = units
            freemon = Monomial(subhmap)
            factor = Monomial(keymon/freemon)
            scale = scale * solution(factor)
            if factor != 1:
                factor = factor**(-1/power)  # invert the transform
                factor.ast = None
                if verbosity:
                    print("  "*indent + "(with a factor of %s (%s))" %
                          (factor.str_without(["unnecessary lineage", "units"]),
                           get_valstr(factor, solution)))
                subsubtree = []
                transform = Transform(factor, 1, keymon)
                orig_subtree.append(Tree(transform, basescale, subsubtree))
                orig_subtree = subsubtree
            if power != 1:
                if verbosity:
                    print("  "*indent + "(with a power of %.2g )" % power)
                subsubtree = []
                transform = Transform(1, 1/power, keymon)  # inverted bc it's on the gt side
                orig_subtree.append(Tree(transform, basescale, subsubtree))
                orig_subtree = subsubtree
    if verbosity:
        if keymon is not None:
            print("  "*indent + "in: "
                  + constraint.str_without(["units", "lineage"]))
        print("  "*indent + "by:")
        indent += 1

    # TODO: use ast_parsing instead of chop?
    monsols = [solcache(solution, mon) for mon in composition.chop()]  # ~20% of total last check # TODO: remove
    parsed_monsols = [getattr(mon, "value", mon) for mon in monsols]
    monvals = [float(mon/scale) for mon in parsed_monsols]   # ~10% of total last check # TODO: remove
    # sort by value, preserving order in case of value tie
    sortedmonvals = sorted(zip(monvals, range(len(monvals)),
                               composition.chop()), reverse=True)
    for scaledmonval, _, mon in sortedmonvals:
        if not scaledmonval:
            continue
        subtree = orig_subtree  # return to the original subtree

        # time for some filtering
        interesting_vks = mon.vks
        potential_filters = [
            {vk for vk in interesting_vks if vk not in bd},
            mon.vks - get_free_vks(mon, solution),
            {vk for vk in interesting_vks if vk in BASICALLY_FIXED_VARIABLES}
        ]
        if scaledmonval < 1 - permissivity:  # skip breakdown filter
            potential_filters = potential_filters[1:]
        for filter in potential_filters:
            if interesting_vks - filter:  # don't remove the last one
                interesting_vks = interesting_vks - filter
        # if filters weren't enough and permissivity is high enough, sort!
        if len(interesting_vks) > 1 and permissivity > 1:
            csenss = solution["sensitivities"]["constraints"]
            best_vks = sorted((vk for vk in interesting_vks if vk in bd),
                key=lambda vk: (-mon.exp[vk]*abs(csenss[bd[vk][0][2]]),
                                str(bd[vk][0][0])))   # ~5% of total last check # TODO: remove
                     # TODO: changing to str(vk) above does some odd stuff, why?
            if best_vks:
                interesting_vks = set([best_vks[0]])
        boring_vks = mon.vks - interesting_vks

        subkey = None
        if len(interesting_vks) == 1:
            subkey, = interesting_vks
            if subkey in visited_bdkeys and len(sortedmonvals) == 1:
                continue  # don't even go there
            if subkey not in bd:
                power = 1  # no need for a transform
            else:
                power = mon.exp[subkey]
                if power < 0 and gone_negative:
                    subkey = None  # don't breakdown another negative

        if subkey is None:
            power = 1
            if scaledmonval > 1 - permissivity and not boring_vks:
                boring_vks = interesting_vks
                interesting_vks = set()
            if not interesting_vks:
                # prioritize showing some boring_vks as if they were "free"
                if len(boring_vks) == 1:
                    interesting_vks = boring_vks
                    boring_vks = set()
                else:
                    for vk in list(boring_vks):
                        if vk.units and not vk.units.dimensionless:
                            interesting_vks.add(vk)
                            boring_vks.remove(vk)

        if interesting_vks and (boring_vks or mag(mon.c) != 1):
            units = 1
            exp = HashVector()
            for vk in interesting_vks:
                exp[vk] = mon.exp[vk]
                if vk.units:
                    units *= vk.units**mon.exp[vk]
            subhmap = NomialMap({exp: 1})
            subhmap.units = None if units is 1 else units
            freemon = Monomial(subhmap)
            factor = mon/freemon  # autoconvert...
            if (factor.units is None and isinstance(factor, FixedScalar)
                    and abs(factor.value - 1) <= 1e-4):
                factor = 1  # minor fudge to clear numerical inaccuracies
            if factor != 1 :
                factor.ast = None
                if verbosity:
                    keyvalstr = "%s (%s)" % (factor.str_without(["unnecessary lineage", "units"]),
                                             get_valstr(factor, solution))
                    print("  "*indent + "(with a factor of %s )" % keyvalstr)
                subsubtree = []
                transform = Transform(factor, 1, mon)
                subtree.append(Tree(transform, scaledmonval, subsubtree))
                subtree = subsubtree
            mon = freemon  # simplifies units
            if power != 1:
                if verbosity:
                    print("  "*indent + "(with a power of %.2g )" % power)
                subsubtree = []
                transform = Transform(1, power, mon)
                subtree.append(Tree(transform, scaledmonval, subsubtree))
                subtree = subsubtree
                mon = mon**(1/power)
                mon.ast = None
        # TODO: make minscale an argument - currently an arbitrary 0.01
        if (subkey is not None and subkey not in visited_bdkeys
                and subkey in bd and scaledmonval > 0.05):
            if verbosity:
                verbosity = indent + 1  # slight hack
            subsubtree = crawl(subkey, bd, solution, scaledmonval,
                               permissivity, verbosity, set(visited_bdkeys),
                               gone_negative)
            subtree.append(subsubtree)
        else:
            if verbosity:
                keyvalstr = "%s (%s)" % (mon.str_without(["unnecessary lineage", "units"]),
                                         get_valstr(mon, solution))
                print("  "*indent + keyvalstr)
            subtree.append(Tree(mon, scaledmonval, []))
    if verbosity == 1:
        solution.set_necessarylineage(clear=True)
    return tree

SYMBOLS = string.ascii_uppercase + string.ascii_lowercase
for ambiguous_symbol in "lILT":
    SYMBOLS = SYMBOLS.replace(ambiguous_symbol, "")

def get_spanstr(legend, length, label, leftwards, solution):
    "Returns span visualization, collapsing labels to symbols"
    if label is None:
        return " "*length
    spacer, lend, rend = "│", "┯", "┷"
    if isinstance(label, Transform):
        spacer, lend, rend = "╎", "╤", "╧"
        if label.power != 1:
            spacer = " "
            lend = rend  = "^" if label.power > 0 else "/"
        # remove origkeys so they collide in the legends dictionary
        label = Transform(label.factor, label.power, None)
        if label.power == 1 and len(str(label.factor)) == 1:
            legend[label] = str(label.factor)

    if label not in legend:
        legend[label] = SYMBOLS[len(legend)]
    shortname = legend[label]

    if length <= 1:
        return shortname
    shortside = int(max(0, length - 2)/2)
    longside  = int(max(0, length - 3)/2)
    if leftwards:
        if length == 2:
            return lend + shortname
        return lend + spacer*shortside + shortname + spacer*longside + rend
    else:
        if length == 2:
            return shortname + rend
        # HACK: no corners on long rightwards - only used for depth 0
        return "┃"*(longside+1) + shortname + "┃"*(shortside+1)

def discretize(tree, extent, solution, collapse, depth=0, justsplit=False):
    # TODO: add vertical simplification?
    key, val, branches = tree
    if collapse:  # collapse Transforms with power 1
        while any(isinstance(branch.key, Transform) and branch.key.power > 0 for branch in branches):
            newbranches = []
            for branch in branches:
                # isinstance(branch.key, Transform) and branch.key.power > 0
                if isinstance(branch.key, Transform) and branch.key.power > 0:
                    newbranches.extend(branch.branches)
                else:
                    newbranches.append(branch)
            branches = newbranches

    scale = extent/val
    values = [b.value for b in branches]
    bkey_indexs = {}
    for i, b in enumerate(branches):
        k = get_keystr(b.key, solution)
        if isinstance(b.key, Transform):
            if len(b.branches) == 1:
                k = get_keystr(b.branches[0].key, solution)
        if k in bkey_indexs:
            values[bkey_indexs[k]] += values[i]
            values[i] = None
        else:
            bkey_indexs[k] = i
    if any(v is None for v in values):
        branches, values = zip(*((b, v) for b, v in zip(branches, values) if v is not None))
        branches = list(branches)
        values = list(values)
    extents = [int(round(scale*v)) for v in values]
    surplus = extent - sum(extents)
    for i, b in enumerate(branches):
        if isinstance(b.key, Transform):
            subscale = extents[i]/b.value
            if not any(round(subscale*subv) for _, subv, _ in b.branches):
                extents[i] = 0  # transform with no worthy heirs: misc it
    if not any(extents):
        return Tree(key, extent, [])
    if not all(extents):  # create a catch-all
        branches = branches.copy()
        miscvkeys, miscval = [], 0
        for subextent in reversed(extents):
            if not subextent or (branches[-1].value < miscval and surplus < 0):
                extents.pop()
                k, v, _ = branches.pop()
                if isinstance(k, Transform):
                    k = k.origkey  # TODO: this is the only use of origkey - remove it
                    if isinstance(k, tuple):
                        vkeys = [(-kv[1], str(kv[0]), kv[0]) for kv in k]
                if not isinstance(k, tuple):
                    vkeys = [(-v, str(k), k)]
                miscvkeys += vkeys
                surplus -= (round(scale*(miscval + v))
                            - round(scale*miscval) - subextent)
                miscval += v
        misckeys = tuple(k for _, _, k in sorted(miscvkeys))
        branches.append(Tree(misckeys, miscval, []))
        extents.append(int(round(scale*miscval)))
    if surplus:
        sign = int(np.sign(surplus))
        bump_priority = sorted((ext, sign*b.value, i) for i, (b, ext)
                               in enumerate(zip(branches, extents)))
        while surplus:
            extents[bump_priority.pop()[-1]] += sign
            surplus -= sign

    tree = Tree(key, extent, [])
    # simplify based on how we're branching
    branchfactor = len([ext for ext in extents if ext]) - 1
    if depth and not isinstance(key, Transform):
        if extent == 1 or branchfactor >= max(extent-2, 1):
            # if we'd branch too much, stop
            return tree
        if collapse and not branchfactor and not justsplit:
            # if we didn't just split and aren't about to, skip through
            return discretize(branches[0], extent, solution, collapse,
                              depth=depth+1, justsplit=False)
    if branchfactor:
        justsplit = True
    elif not isinstance(key, Transform):  # justsplit passes through transforms
        justsplit = False

    for branch, subextent in zip(branches, extents):
        if subextent:
            branch = discretize(branch, subextent, solution, collapse,
                                depth=depth+1, justsplit=justsplit)
            if (collapse and is_power(branch.key)
                    and all(is_power(b.key) for b in branch.branches)):
                # combine stacked powers
                power = branch.key.power
                for b in branch.branches:
                    key = Transform(1, power*b.key.power, None)
                    if key.power == 1:  # powers canceled, collapse both
                        tree.branches.extend(b.branches)
                    else:  # collapse this level
                        tree.branches.append(Tree(key, b.value, b.branches))
            else:
                tree.branches.append(branch)
    return tree

def layer(map, tree, maxdepth, depth=0):
    "Turns the tree into a 2D-array"
    key, extent, branches = tree
    if depth <= maxdepth:
        if len(map) <= depth:
            map.append([])
        map[depth].append((key, extent))
        if not branches:
            branches = [Tree(None, extent, [])]  # pad it out
        for branch in branches:
            layer(map, branch, maxdepth, depth+1)
    return map

def plumb(tree, depth=0):
    "Finds maximum depth of a tree"
    maxdepth = depth
    for branch in tree.branches:
        maxdepth = max(maxdepth, plumb(branch, depth+1))
    return maxdepth

def prune(tree, solution, maxlength, length=4, prefix=""):
    "Prune branches that are longer than a certain number of characters"
    key, extent, branches = tree
    if length == 4 and isinstance(key, VarKey) and key.necessarylineage:
        prefix = key.lineagestr()
    keylength = max(len(get_valstr(key, solution, into="(%s)")),
                    len(get_keystr(key, solution, prefix)))
    length += keylength + 3
    stop_here = False
    for branch in branches:
        keylength = max(len(get_valstr(branch.key, solution, into="(%s)")),
                        len(get_keystr(branch.key, solution, prefix)))
        branchlength = length + keylength
        if branchlength > maxlength:
            return Tree(key, extent, [])
    return Tree(key, extent, [prune(b, solution, maxlength, length, prefix)
                              for b in branches])

def simplify(tree, solution, extent, maxdepth, maxlength, collapse):
    "Discretize, prune, and layer a tree to prepare for printing"
    subtree = discretize(tree, extent, solution, collapse)
    if collapse and maxlength:
        subtree = prune(subtree, solution, maxlength)
    return layer([], subtree, maxdepth)

# @profile  # ~16% of total last check # TODO: remove
def graph(tree, solution, height=None, maxdepth=None, maxwidth=110,
          showlegend=False):
    "Prints breakdown"
    solution.set_necessarylineage()
    collapse = True #(not showlegend)
    maxwidth = None
    if maxdepth is None:
        maxdepth = plumb(tree)
    if height is not None:
        mt = simplify(tree, solution, height, maxdepth, maxwidth, collapse)
    else:  # zoom in from a default height of 20 to a height of 4 per branch
        prev_height = None
        height = 20
        while prev_height != height:
            mt = simplify(tree, solution, height, maxdepth, maxwidth, collapse)
            prev_height = height
            height = min(height, max(*(4*len(at_depth) for at_depth in mt)))

    legend = {}
    chararray = np.full((len(mt), height), "", "object")
    for depth, elements_at_depth in enumerate(mt):
        row = ""
        for i, (element, length) in enumerate(elements_at_depth):
            leftwards = depth > 0 and length > 2
            row += get_spanstr(legend, length, element, leftwards, solution)
        chararray[depth, :] = list(row)

    # Format depth=0
    A_key, = [key for key, value in legend.items() if value == "A"]
    A_str = get_keystr(A_key, solution)
    prefix = ""
    if isinstance(A_key, VarKey) and A_key.necessarylineage:
        prefix = A_key.lineagestr()
    A_valstr = get_valstr(A_key, solution, into="(%s)")
    fmt = "{0:>%s}" % (max(len(A_str), len(A_valstr)) + 3)
    for j, entry in enumerate(chararray[0,:]):
        if entry == "A":
            chararray[0,j] = fmt.format(A_str + "╺┫")
            chararray[0,j+1] = fmt.format(A_valstr + " ┃")
        else:
            chararray[0,j] = fmt.format(entry)
    # Format depths 1+
    labeled = set()
    reverse_legend = {v: k for k, v in legend.items()}
    legend = {}
    for pos in range(height):
        for depth in reversed(range(1,len(mt))):
            char = chararray[depth, pos]
            if char not in reverse_legend:
                continue
            key = reverse_legend[char]
            if key not in legend and (isinstance(key, tuple) or (depth != len(mt) - 1 and chararray[depth+1, pos] != " ")):
                legend[key] = SYMBOLS[len(legend)]
            if collapse and is_power(key):
                chararray[depth, pos] = "^" if key.power > 0 else "/"
                del legend[key]
                continue
            if key in legend:
                chararray[depth, pos] = legend[key]
                if isinstance(key, tuple) and not isinstance(key, Transform):
                    chararray[depth, pos] =  "*" # + chararray[depth, pos]
                    del legend[key]
                if showlegend:
                    continue

            keystr = get_keystr(key, solution, prefix)
            if keystr in labeled:
                valuestr = ""
            else:
                valuestr = get_valstr(key, solution, into=" (%s)")
            if collapse:
                fmt = "{0:<%s}" % max(len(keystr) + 3, len(valuestr) + 2)
            else:
                fmt = "{0:<1}"
            span = 0
            tryup, trydn = True, True
            while tryup or trydn:
                span += 1
                if tryup:
                    if pos - span < 0:
                        tryup = False
                    else:
                        upchar = chararray[depth, pos-span]
                        if upchar == "│":
                            chararray[depth, pos-span] = fmt.format("┃")
                        elif upchar == "┯":
                            chararray[depth, pos-span] = fmt.format("┓")
                        else:
                            tryup = False
                if trydn:
                    if pos + span >= height:
                        trydn = False
                    else:
                        dnchar = chararray[depth, pos+span]
                        if dnchar == "│":
                            chararray[depth, pos+span] = fmt.format("┃")
                        elif dnchar == "┷":
                            chararray[depth, pos+span] = fmt.format("┛")
                        else:
                            trydn = False
            linkstr = "┣╸"
            if not isinstance(key, FixedScalar):
                labeled.add(keystr)
                if span > 1 and (collapse or pos + 2 >= height
                                 or chararray[depth, pos+1] == "┃"):
                    vallabel = chararray[depth, pos+1].rstrip() + valuestr
                    chararray[depth, pos+1] = fmt.format(vallabel)
                elif showlegend:
                    keystr += valuestr
            chararray[depth, pos] = fmt.format(linkstr + keystr)
    # Rotate and print
    rowstrs = ["    " + "".join(row).rstrip() for row in chararray.T.tolist()]
    print("\n" + "\n".join(rowstrs) + "\n")

    if showlegend:  # create and print legend
        legend_lines = []
        for key, shortname in sorted(legend.items(), key=lambda kv: kv[1]):
            legend_lines.append(legend_entry(key, shortname, solution))
        maxlens = [max(len(el) for el in col) for col in zip(*legend_lines)]
        fmts = ["{0:<%s}" % L for L in maxlens]
        for line in legend_lines:
            line = "".join(fmt.format(cell)
                           for fmt, cell in zip(fmts, line) if cell).rstrip()
            print("    " + line)

    solution.set_necessarylineage(clear=True)

def legend_entry(key, shortname, solution):
    "Returns list of legend elements"
    operator = note = ""
    keystr = valuestr = " "
    operator = "= " if shortname else "  + "
    if is_factor(key):
        operator = " ×"
        key = key.factor
        free, quasifixed = False, False
        if any(vk not in BASICALLY_FIXED_VARIABLES
               for vk in get_free_vks(key, solution)):
            note = "  [free factor]"
    if is_power(key):
        valuestr = "   ^%.3g" % key.power
    else:
        valuestr = get_valstr(key, solution, into="  "+operator+"%s")
        if not isinstance(key, FixedScalar):
            keystr = get_keystr(key, solution)
    return ["%-4s" % shortname, keystr, valuestr, note]

def get_keystr(key, solution, prefix=""):
    "Returns formatted string of the key in solution."
    if key is solution.costposy:
        out = "Cost"
    elif hasattr(key, "str_without"):
        out = key.str_without({"unnecessary lineage",
                               "units", ":MAGIC:"+prefix})
    elif isinstance(key, tuple):
        out = "[%i terms]" % len(key)
    else:
        out = str(key)
    return out if len(out) <= 67 else out[:66]+"…"

def get_valstr(key, solution, into="%s"):
    "Returns formatted string of the value of key in solution."
    # get valuestr
    try:
        value = solution(key)
    except (ValueError, TypeError):
        try:
            value = sum(solution(subkey) for subkey in key)
        except (ValueError, TypeError):
            return " "
    if isinstance(value, FixedScalar):
        value = value.value
    if 1e3 <= mag(value) < 1e6:
        valuestr = "{:,.0f}".format(mag(value))
    else:
        valuestr = "%-.3g" % mag(value)
    # get unitstr
    if hasattr(key, "unitstr"):
        unitstr = key.unitstr()
    else:
        try:
            if hasattr(value, "units"):
                value.ito_reduced_units()
        except DimensionalityError:
            pass
        unitstr = get_unitstr(value)
    if unitstr[:2] == "1/":
        unitstr = "/" + unitstr[2:]
    if key in solution["constants"] or (
            hasattr(key, "vks") and key.vks
            and all(vk in solution["constants"] for vk in key.vks)):
        unitstr += ", fixed"
    return into % (valuestr + unitstr)


import pickle
from gpkit import ureg
ureg.define("pax = 1")
ureg.define("paxkm = km")
ureg.define("trip = 1")

print("STARTING...")
from gpkit.tests.helpers import StdoutCaptured

import difflib

permissivity = 2

# sol = pickle.load(open("solar.p", "rb"))
# bd = get_breakdowns(sol)
# mbd = get_model_breakdown(sol)
# mtree = crawl_modelbd(mbd)
# # graph(mtree, sol, showlegend=True, height=20)
# # graph(mtree.branches[0].branches[0].branches[0], sol, showlegend=False, height=20)
#
# tree = crawl(sol.costposy, bd, sol, permissivity=2, verbosity=0)
# graph(tree, sol)
#
#
# # key, = [vk for vk in bd if "Wing.Planform.b" in str(vk)]
# # tree = crawl(key, bd, sol, permissivity=2, verbosity=1)
# # graph(tree, sol)
# # key, = [vk for vk in bd if "Aircraft.Fuselage.R[0,0]" in str(vk)]
# # tree = crawl(key, bd, sol, permissivity=2, verbosity=1)
# # graph(tree, sol)
# # key, = [vk for vk in bd if "Mission.Climb.AircraftDrag.CD[0]" in str(vk)]
# # tree = crawl(key, bd, sol, permissivity=2, verbosity=1)
# # graph(tree, sol)
#
#
# keys = sorted(bd.keys(), key=str)
#
# # with StdoutCaptured("solarbreakdowns.log"):
# #     graph(mtree, sol, showlegend=False)
# #     graph(mtree, sol, showlegend=True)
# #     tree = crawl(sol.costposy, bd, sol, permissivity=permissivity)
# #     graph(tree, sol)
# #     for key in keys:
# #         tree = crawl(key, bd, sol, permissivity=permissivity)
# #         graph(tree, sol)
#
# with StdoutCaptured("solarbreakdowns.log.new"):
#     graph(mtree, sol, showlegend=False)
#     graph(mtree, sol, showlegend=True)
#     tree = crawl(sol.costposy, bd, sol, permissivity=permissivity)
#     graph(tree, sol)
#     for key in keys:
#         tree = crawl(key, bd, sol, permissivity=permissivity)
#         try:
#             graph(tree, sol)
#         except:
#             raise ValueError(key)
#
# with open("solarbreakdowns.log", "r") as original:
#     with open("solarbreakdowns.log.new", "r") as new:
#         diff = difflib.unified_diff(
#             original.readlines(),
#             new.readlines(),
#             fromfile="original",
#             tofile="new",
#         )
#         for line in diff:
#             print(line[:-1])
#
# print("SOLAR DONE")


sol = pickle.load(open("bd.p", "rb"))
bd = get_breakdowns(sol)
mbd = get_model_breakdown(sol)
mtree = crawl_modelbd(mbd)

# graph(mtree, sol, showlegend=False)
#
# key, = [vk for vk in bd if "podtripenergy[1,0]" in str(vk)]
# tree = crawl(key, bd, sol, permissivity=2, verbosity=1)
# graph(tree, sol, height=40)
tree = crawl(sol.costposy, bd, sol, permissivity=2, verbosity=1)
graph(tree, sol, showlegend=True)
# key, = [vk for vk in bd if "statorlaminationmassperpolepair" in str(vk)]
# tree = crawl(key, bd, sol, permissivity=2, verbosity=1)
# graph(tree, sol)
# key, = [vk for vk in bd if "numberofcellsperstring" in str(vk)]
# tree = crawl(key, bd, sol, permissivity=2, verbosity=1)
# graph(tree, sol)
#
# keys = sorted((key for key in bd.keys() if not key.idx or len(key.shape) == 1),
#               key=lambda k: k.str_without(excluded={}))

# with StdoutCaptured("breakdowns.log"):
#     graph(mtree, sol, showlegend=False)
#     graph(mtree.branches[0].branches[1], sol, showlegend=False)
#     graph(mtree, sol, showlegend=True)
#     tree = crawl(sol.costposy, bd, sol, permissivity=permissivity)
#     graph(tree, sol)
#     for key in keys:
#         tree = crawl(key, bd, sol, permissivity=permissivity)
#         graph(tree, sol)

# with StdoutCaptured("breakdowns.log.new"):
#     graph(mtree, sol, showlegend=False)
#     graph(mtree.branches[0].branches[1], sol, showlegend=False)
#     graph(mtree, sol, showlegend=True)
#     tree = crawl(sol.costposy, bd, sol, permissivity=permissivity)
#     graph(tree, sol)
#     for key in keys:
#         tree = crawl(key, bd, sol, permissivity=permissivity)
#         try:
#             graph(tree, sol)
#         except:
#             raise ValueError(key)
#
# with open("breakdowns.log", "r") as original:
#     with open("breakdowns.log.new", "r") as new:
#         diff = difflib.unified_diff(
#             original.readlines(),
#             new.readlines(),
#             fromfile="original",
#             tofile="new",
#         )
#         for line in diff:
#             print(line[:-1])

print("DONE")

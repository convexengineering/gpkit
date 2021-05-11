# pylint: skip-file
import string
from collections import defaultdict, namedtuple
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
        vk = vk.str_without({"lineage"}) + get_valstr(vk, solution, " = %s")
        subbd[vk] = {"|sensitivity|": senss}
    # print(breakdowns["|sensitivity|"])
    return breakdowns

def crawl_modelbd(bd, tree=None, name="Model"):
    if tree is None:
        tree = []
    subtree = []
    tree.append(Tree(name, bd.pop("|sensitivity|"), subtree))
    for key, _ in sorted(bd.items(), key=lambda kv: kv[1]["|sensitivity|"], reverse=True):
        crawl_modelbd(bd[key], subtree, key)
    return tree[0]

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
                lt, gt = [constraint.left, constraint.right]
            else:  # r_over_l is more sensitive - see nomials/math.py
                gt, lt = [constraint.left, constraint.right]
        if lt.any_nonpositive_cs or len(gt.hmap) > 1:
            continue  # no signomials
        freegt_vks = get_free_vks(gt, solution)
        if len(freegt_vks) < 1:
            freegt_vks = gt.vks
        if len(freegt_vks) > 1:  # TODO: "strict_breakdowns" flag prevents this, maybe negative exps too
            consistent_lt_pows = defaultdict(set)
            for exp in lt.hmap:
                for vk, pow in exp.items():
                    consistent_lt_pows[vk].add(pow)
            for vk, pows in consistent_lt_pows.items():
                if len(pows) == 1:
                    pow, = pows
                    hmap = NomialMap({HashVector({vk: 1}): 1.0})
                    hmap.units = vk.units
                    var = Monomial(hmap)**pow
                    lt, gt = lt/var, gt/var
                    lt.ast = gt.ast = None
            sortedgtvks = sorted(gt.vks, key=lambda vk: (-np.sign(gt.exp[vk]), -round(solution["sensitivities"]["variablerisk"].get(vk, 0), 2), str(vk)))
            skip = set(breakdowns)
            freegt_vks = []
            for vk in sortedgtvks:
                if vk in skip:
                    skip.update(beatout[vk])
                freegt_vks.append(vk)
                break
            if not freegt_vks:
                continue
            else:
                beatout[freegt_vks[0]].update(gt.vks)
                freegt_vks = {freegt_vks[0]}
            for vk in gt.vks:
                if vk not in freegt_vks:
                    hmap = NomialMap({HashVector({vk: 1}): 1.0})
                    hmap.units = vk.units
                    var = Monomial(hmap)**gt.exp[vk]
                    lt, gt = lt/var, gt/var
                    lt.ast = gt.ast = None
        if len(freegt_vks) == 1:
            brokendownvk, = freegt_vks
            breakdowns[brokendownvk].append((lt, gt, constraint))
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
        if vk not in bd:
            return
        if vk in BASICALLY_FIXED_VARIABLES:
            continue
        if vk is key:
            continue
        if vk in visited:  # been here before and it's not me
            return
        visited.add(key)
        get_fixity(vk, bd, solution, basically_fixed, visited)
        if vk not in BASICALLY_FIXED_VARIABLES:
            return
    basically_fixed.add(key)

# @profile  # ~84% of total last check # TODO: remove
def crawl(key, bd, solution, basescale=1, permissivity=2, verbosity=0,
          visited_bdkeys=None):
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
        keyvalstr = "%s (%s)" % (key.str_without(["unnecessary lineage", "units"]),
                                 get_valstr(key, solution))
        print("  "*indent + keyvalstr + ", which breaks down further")
        indent += 1
    orig_subtree = subtree = []
    tree = Tree(key, basescale, subtree)
    visited_bdkeys.add(key)
    if keymon is None:
        scale = solution(key)/basescale
    else:
        if len(keymon.vks) == 1:  # constant
            free_vks = keymon.vks
        else:
            free_vks = get_free_vks(keymon, solution)
        # if len(free_vks) != 1:
        #     free_vks = {sorted(keymon.vks, key=lambda vk: (-round(solution["sensitivities"]["variablerisk"].get(vk, 0), 2), str(vk)))[0]}
        subkey, = free_vks
        power = keymon.exp[subkey]
        fixed_vks = set(keymon.vks) - free_vks
        scale = solution(key)**power/basescale
        # TODO: make method that can handle both kinds of transforms
        if power != 1 or fixed_vks or mag(keymon.c) != 1 or keymon.units != key.units:
            units = 1
            exp = HashVector()
            for vk in free_vks:
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
                    keyvalstr = "%s (%s)" % (factor.str_without(["unnecessary lineage", "units"]),
                                             get_valstr(factor, solution))
                    print("  "*indent + "(with a factor of " + keyvalstr + " )")
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
            print("  "*indent + "in: " + constraint.str_without(["units", "lineage"]))
        print("  "*indent + "by:")
        indent += 1

    # TODO: use ast_parsing instead of chop?
    monsols = [solution(mon) for mon in composition.chop()]  # ~20% of total last check # TODO: remove
    parsed_monsols = [getattr(mon, "value", mon) for mon in monsols]
    monvals = [float(mon/scale) for mon in parsed_monsols]   # ~10% of total last check # TODO: remove
    # sort by value, preserving order in case of value tie
    sortedmonvals = sorted(zip(monvals, range(len(monvals)),
                               composition.chop()), reverse=True)
    for scaledmonval, _, mon in sortedmonvals:
        if not scaledmonval:
            continue
        scaledmonval = min(1, scaledmonval)  # clip it
        subtree = orig_subtree  # revert back to the original subtree

        # time for some filtering
        free_vks = mon.vks

        if scaledmonval > 1 - permissivity:
            unbreakdownable_vks = {vk for vk in free_vks if vk not in bd}
            if free_vks - unbreakdownable_vks:  # don't remove the last one
                free_vks = free_vks - unbreakdownable_vks

        fixed_vks = mon.vks - get_free_vks(mon, solution)
        if free_vks - fixed_vks:  # don't remove the last one
            free_vks = free_vks - fixed_vks

        basically_fixed_vks = {vk for vk in free_vks
                               if vk in BASICALLY_FIXED_VARIABLES}
        if free_vks - basically_fixed_vks:  # don't remove the last one
            free_vks = free_vks - basically_fixed_vks

        # if scaledmonval > 1 - permissivity:
        #     unbreakdownable_vks = {vk for vk in free_vks if vk not in bd}
        #     if free_vks - unbreakdownable_vks:  # don't remove the last one
        #         free_vks = free_vks - unbreakdownable_vks

        if len(free_vks) > 1 and permissivity > 1:
            best_vks = sorted((vk for vk in free_vks if vk in bd),
                key=lambda vk:
                    # TODO: without exp: "most strongly broken-down component"
                    #       but it could use nus (or v_ss) to say
                    #       "breakdown which the solution is most sensitive to"
                    #  ...right now it's in-between
                    (abs(mon.exp[vk]*solution["sensitivities"]["constraints"][bd[vk][0][2]]),
                     str(bd[vk][0][0])), reverse=True)   # ~5% of total last check # TODO: remove
            if best_vks:
                free_vks = set([best_vks[0]])

        fixed_vks = mon.vks - free_vks

        if len(free_vks) == 1:
            subkey, = free_vks
            if subkey in visited_bdkeys and len(sortedmonvals) == 1:
                continue  # don't continue
            power = mon.exp[subkey]
            if power != 1 and subkey not in bd:
                power = 1  # no need for a transform
        else:
            subkey = None
            power = 1
            if scaledmonval > 1 - permissivity and not fixed_vks:
                fixed_vks = free_vks
                free_vks = set()
            if not free_vks:
                # prioritize showing some fixed_vks as if they were "free"
                if len(fixed_vks) == 1:
                    free_vks = fixed_vks
                    fixed_vks = set()
                else:
                    for vk in list(fixed_vks):
                        if vk.units and not vk.units.dimensionless:
                            free_vks.add(vk)
                            fixed_vks.remove(vk)

        if free_vks and (fixed_vks or mag(mon.c) != 1):
            if subkey:
                kindafree_vks = set(vk for vk in fixed_vks
                                    if vk not in solution["constants"])
                if kindafree_vks == fixed_vks:
                    kindafree_vks = set()  # don't remove ALL of them
                else:
                    free_vks.update(kindafree_vks)
            units = 1
            exp = HashVector()
            for vk in free_vks:
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
            if subkey and fixed_vks and kindafree_vks:
                units = 1
                exp = HashVector()
                for vk in kindafree_vks:
                    exp[vk] = mon.exp[vk]
                    if vk.units:
                        units *= vk.units**mon.exp[vk]
                subhmap = NomialMap({exp: 1})
                try:
                    subhmap.units = None if units == 1 else units
                except DimensionalityError:
                    # pints was unable to divide a unit by itself bc
                    #   it has terrible floating-point errors.
                    #   so let's assume it isn't dimensionless
                    #   even though it probably is
                    subhmap.units = units
                factor = Monomial(subhmap)
                if factor != 1:
                    factor.ast = None
                    if verbosity:
                        keyvalstr = "%s (%s)" % (factor.str_without(["unnecessary lineage", "units"]),
                                                 get_valstr(factor, solution))
                        print("  "*indent + "(with a factor of " + keyvalstr + " )")
                    subsubtree = []
                    transform = Transform(factor, 1, mon)
                    subtree.append(Tree(transform, scaledmonval, subsubtree))
                    subtree = subsubtree
                    mon = mon/factor
                    mon.ast = None
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
                and subkey in bd and scaledmonval > 0.01):
            if verbosity:
                verbosity = indent + 1  # slight hack
            try:
                subsubtree = crawl(subkey, bd, solution, scaledmonval,
                                   permissivity, verbosity, set(visited_bdkeys))
                subtree.append(subsubtree)
                continue
            except Exception as e:
                print(subkey, e)
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
        # TODO: catch PI (or again could that come from AST parsing?)
        # if label.power == 1 and len(str(label.factor)) == 1:
        #     legend[label] = str(label.factor)

    if label not in legend:
        shortname = SYMBOLS[len(legend)]
        legend[label] = shortname
    else:
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

def simplify(tree, extent, collapse, depth=0, justsplit=False):
    # TODO: add vertical simplification?
    key, val, branches = tree
    if collapse:  # collapse Transforms with power 1
        while any(is_factor(branch.key) for branch in branches):
            newbranches = []
            for branch in branches:
                if is_factor(branch.key):
                    newbranches.extend(branch.branches)
                else:
                    newbranches.append(branch)
            branches = newbranches

    scale = extent/val
    extents = [round(scale*b.value) for b in branches]
    for i, b in enumerate(branches):
        if isinstance(b.key, Transform):
            subscale = extents[i]/b.value
            if not any(round(subscale*subv) for _, subv, _ in b.branches):
                extents[i] = 0  # transform with no worthy heirs: misc it
    if not any(extents):
        return Tree(key, extent, [])
    if not all(extents):  # create a catch-all
        branches = branches.copy()
        surplus = extent - sum(extents)
        miscvkeys, miscval = [], 0
        for subextent in reversed(extents):
            if not subextent or (branches[-1].value < miscval and surplus < 0):
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

    extents = [int(round(scale*b.value)) for b in branches]
    surplus = extent - sum(extents)
    if surplus:
        sign = int(np.sign(surplus))
        bump_priority = sorted((ext, sign*b.value, i) for i, (b, ext)
                               in enumerate(zip(branches, extents)))
        while surplus:
            extents[bump_priority.pop()[-1]] += sign
            surplus -= sign

    branchfactor = len([ext for ext in extents if ext]) - 1

    if depth and not isinstance(key, Transform):
        if extent == 1 or branchfactor >= max(extent-2, 1):
            # if we'd branch too much, stop
            return Tree(key, extent, [])
        if collapse and not branchfactor and not justsplit:
            # if we didn't just split and aren't about to, collapse
            return simplify(branches[0], extent, collapse,
                            depth=depth+1, justsplit=False)

    if branchfactor:
        justsplit = True
    elif not isinstance(key, Transform):  # pass through transforms
        justsplit = False

    tree = Tree(key, extent, [])
    for branch, subextent in zip(branches, extents):
        if subextent:
            branch = simplify(branch, subextent, collapse,
                              depth=depth+1, justsplit=justsplit)
            if (collapse and is_power(branch.key)
                    and all(is_power(b.key) for b in branch.branches)):
                # collapse stacked powers
                power = branch.key.power
                for b in branch.branches:
                    key = Transform(1, power*b.key.power, None)
                    if key.power == 1:
                        tree.branches.extend(b.branches)
                    else:
                        tree.branches.append(Tree(key, b.value, b.branches))
            else:
                tree.branches.append(branch)
    return tree

def layer(map, tree, maxdepth, depth=0):
    "Turns the tree into a 2D-array"
    if depth > maxdepth:
        return map
    if len(map) <= depth:
        map.append([])
    key, extent, branches = tree
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

# @profile  # ~16% of total last check # TODO: remove
def graph(tree, solution, extent=None, maxdepth=None, showlegend=False, maxwidth=110):
    "Prints breakdown"
    if maxdepth is None:
        maxdepth = plumb(tree)
    if extent is None:  # auto-zoom-in from 20
        prev_extent = None
        extent = 20
        while prev_extent != extent:
            subtree = simplify(tree, extent, collapse=not showlegend)
            mt = layer([], subtree, maxdepth)
            prev_extent = extent
            extent = min(extent, max(*(4*len(at_depth) for at_depth in mt)))
    else:
        subtree = simplify(tree, extent, collapse=not showlegend)
        mt = layer([], subtree, maxdepth)
    legend = {}
    chararray = np.full((len(mt), extent), "", "object")
    for depth, elements_at_depth in enumerate(mt):
        row = ""
        for i, (element, length) in enumerate(elements_at_depth):
            leftwards = depth > 0 and length > 2
            row += get_spanstr(legend, length, element, leftwards, solution)
        chararray[depth, :] = list(row)

    solution.set_necessarylineage()

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
    new_legend = {}
    for pos in range(extent):
        for depth in reversed(range(1,len(mt))):
            value = chararray[depth, pos]
            if value not in SYMBOLS:
                continue
            key, = [k for k, val in legend.items() if val == value]
            if getattr(key, "vks", None) and len(key.vks) == 1 and all(vk in new_legend for vk in key.vks):
                key, = key.vks
            if key not in new_legend and (isinstance(key, tuple) or (depth != len(mt) - 1 and chararray[depth+1, pos] != " ")):
                new_legend[key] = SYMBOLS[len(new_legend)]
            if key in new_legend:
                chararray[depth, pos] = new_legend[key]
                if isinstance(key, tuple) and not isinstance(key, Transform):
                    chararray[depth, pos] =  "*" + chararray[depth, pos]
                if showlegend:
                    continue
            tryup, trydn = True, True
            span = 0
            if not showlegend and is_power(key):
                chararray[depth, pos] = "^" if key.power > 0 else "/"
                continue

            keystr = get_keystr(key, solution, prefix)
            valuestr = get_valstr(key, solution, into=" (%s)")
            if keystr in labeled:
                valuestr = ""
            if not showlegend:
                fmt = "{0:<%s}" % max(len(keystr) + 3, len(valuestr) + 2)
            else:
                fmt = "{0:<1}"
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
                    if pos + span >= extent:
                        trydn = False
                    else:
                        dnchar = chararray[depth, pos+span]
                        if dnchar == "│":
                            chararray[depth, pos+span] = fmt.format("┃")
                        elif dnchar == "┷":
                            chararray[depth, pos+span] = fmt.format("┛")
                        else:
                            trydn = False
            #TODO: make submodels show up with this; bd should be an argument
            if showlegend and (key in bd or (hasattr(key, "vks") and key.vks and any(vk in bd for vk in key.vks))):
                linkstr = "┣┉"
            else:
                linkstr = "┣╸"
            if not isinstance(key, FixedScalar):
                labeled.add(keystr)
                if span > 1 and (not showlegend or pos + 2 >= extent or chararray[depth, pos+1] == "┃"):
                    chararray[depth, pos+1] = fmt.format(chararray[depth, pos+1].rstrip() + valuestr)
                elif showlegend:
                    keystr += valuestr
            chararray[depth, pos] = fmt.format(linkstr + keystr)

    # Rotate and print
    toowiderows = []
    rows = chararray.T.tolist()
    if not showlegend:  # remove according to maxwidth
        for i, orig_row in enumerate(rows):
            depth_occluded = -1
            width = None
            row = orig_row.copy()
            row.append("")
            while width is None or width > maxwidth:
                row = row[:-1]
                rowstr = "    " + "".join(row).rstrip()
                width = len(rowstr)
                depth_occluded += 1
            if depth_occluded:
                previous_is_pow = orig_row[-depth_occluded-1] in "^/"
                if abs(depth_occluded) + 1 + previous_is_pow < len(orig_row):
                    strdepth = len("    " + "".join(orig_row[:-depth_occluded]))
                    toowiderows.append((strdepth, i))
    rowstrs = ["    " + "".join(row).rstrip() for row in rows]
    for depth_occluded, i in sorted(toowiderows, reverse=True):
        if len(rowstrs[i]) <= depth_occluded:
            continue  # already occluded
        if "┣" == rowstrs[i][depth_occluded]:
            pow = 0
            while rowstrs[i][depth_occluded-pow-1] in "^/":
                pow += 1
            rowstrs[i] = rowstrs[i][:depth_occluded-pow]
            connected = "^┃┓┛┣╸"
            for dir in [-1, 1]:
                idx = i + dir
                while (0 <= idx < len(rowstrs)
                       and len(rowstrs[idx]) > depth_occluded
                       and rowstrs[idx][depth_occluded]
                       and rowstrs[idx][depth_occluded] in connected):
                    while rowstrs[idx][depth_occluded-pow-1] in "^/":
                        pow += 1
                    rowstrs[idx] = rowstrs[idx][:depth_occluded-pow]
                    idx += dir
    vertstr = "\n".join(rowstr.rstrip() for rowstr in rowstrs)
    print()
    print(vertstr)
    print()
    legend = new_legend

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
    if key is None:
        out = " "
    elif key is solution.costposy:
        out = "Cost"
    elif hasattr(key, "str_without"):
        out = key.str_without({"unnecessary lineage", "units", ":MAGIC:"+prefix})
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
    if key in solution["constants"] or (hasattr(key, "vks") and key.vks and all(vk in solution["constants"] for vk in key.vks)):
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

sol = pickle.load(open("solar.p", "rb"))

bd = get_breakdowns(sol)

sol.set_necessarylineage()
mbd = get_model_breakdown(sol)
sol.set_necessarylineage(clear=True)
mtree = crawl_modelbd(mbd)
graph(mtree, sol, showlegend=True, extent=20)
graph(mtree.branches[0].branches[0].branches[0], sol, showlegend=False, extent=20)


# tree = crawl(sol.costposy, bd, sol, permissivity=2, verbosity=0)
# graph(tree, sol)
#
#
# key, = [vk for vk in bd if "Aircraft.Empennage.HorizontalTail.BoxSpar.W" in str(vk)]
# tree = crawl(key, bd, sol, permissivity=2, verbosity=1)
# graph(tree, sol)
# key, = [vk for vk in bd if "Aircraft.Fuselage.R[0,0]" in str(vk)]
# tree = crawl(key, bd, sol, permissivity=2, verbosity=1)
# graph(tree, sol)
# key, = [vk for vk in bd if "Mission.Climb.AircraftDrag.CD[0]" in str(vk)]
# tree = crawl(key, bd, sol, permissivity=2, verbosity=1)
# graph(tree, sol)


keys = sorted(bd.keys(), key=lambda k: str(k))

# with StdoutCaptured("solarbreakdowns.log"):
#     graph(mtree, sol, showlegend=False)
#     graph(mtree, sol, showlegend=True)
#     tree = crawl(sol.costposy, bd, sol, permissivity=permissivity)
#     graph(tree, sol)
#     for key in keys:
#         tree = crawl(key, bd, sol, permissivity=permissivity)
#         graph(tree, sol)

with StdoutCaptured("solarbreakdowns.log.new"):
    graph(mtree, sol, showlegend=False)
    graph(mtree, sol, showlegend=True)
    tree = crawl(sol.costposy, bd, sol, permissivity=permissivity)
    graph(tree, sol)
    for key in keys:
        tree = crawl(key, bd, sol, permissivity=permissivity)
        try:
            graph(tree, sol)
        except:
            raise ValueError(key)

with open("solarbreakdowns.log", "r") as original:
    with open("solarbreakdowns.log.new", "r") as new:
        diff = difflib.unified_diff(
            original.readlines(),
            new.readlines(),
            fromfile="original",
            tofile="new",
        )
        for line in diff:
            print(line[:-1])

print("SOLAR DONE")


sol = pickle.load(open("bd.p", "rb"))
bd = get_breakdowns(sol)
mbd = get_model_breakdown(sol)
mtree = crawl_modelbd(mbd)

key, = [vk for vk in bd if "ccorechannelheight[0]" in str(vk)]
tree = crawl(key, bd, sol, permissivity=2, verbosity=0)
# sol.set_necessarylineage()
graph(tree, sol)
# sol.set_necessarylineage(clear=True)

keys = sorted((key for key in bd.keys() if not key.idx or len(key.shape) == 1),
              key=lambda k: k.str_without(excluded={}))

# with StdoutCaptured("breakdowns.log"):
#     graph(mtree, sol, showlegend=False)
#     graph(mtree.branches[0].branches[1], sol, showlegend=False)
#     graph(mtree, sol, showlegend=True)
#     tree = crawl(sol.costposy, bd, sol, permissivity=permissivity)
#     graph(tree, sol)
#     for key in keys:
#         tree = crawl(key, bd, sol, permissivity=permissivity)
#         graph(tree, sol)

with StdoutCaptured("breakdowns.log.new"):
    graph(mtree, sol, showlegend=False)
    graph(mtree.branches[0].branches[1], sol, showlegend=False)
    graph(mtree, sol, showlegend=True)
    tree = crawl(sol.costposy, bd, sol, permissivity=permissivity)
    graph(tree, sol)
    for key in keys:
        tree = crawl(key, bd, sol, permissivity=permissivity)
        try:
            graph(tree, sol)
        except:
            raise ValueError(key)

with open("breakdowns.log", "r") as original:
    with open("breakdowns.log.new", "r") as new:
        diff = difflib.unified_diff(
            original.readlines(),
            new.readlines(),
            fromfile="original",
            tofile="new",
        )
        for line in diff:
            print(line[:-1])

print("DONE")

#TODO: cleanup weird conditionals
#      add conversions to plotly/sankey

# pylint: skip-file
import string
from collections import defaultdict, namedtuple, Counter
from gpkit.nomials import Monomial, Posynomial, Variable
from gpkit.nomials.map import NomialMap
from gpkit.small_scripts import mag, try_str_without
from gpkit.small_classes import FixedScalar, HashVector
from gpkit.exceptions import DimensionalityError
from gpkit.repr_conventions import unitstr as get_unitstr
from gpkit.repr_conventions import lineagestr
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
        for parent in lineagestr(constraint).split("."):
            if parent == "":
                continue
            if parent not in subbd:
                subbd[parent] = {}
            subbd = subbd[parent]
            if "|sensitivity|" not in subbd:
                subbd["|sensitivity|"] = 0
            subbd["|sensitivity|"] += senss
        # treat vectors as namespace
        constrstr = try_str_without(constraint, {"units", ":MAGIC:"+lineagestr(constraint)})
        if " at 0x" in constrstr:  # don't print memory addresses
            constrstr = constrstr[:constrstr.find(" at 0x")] + ">"
        subbd[constrstr] = {"|sensitivity|": senss}
    for vk in solution["sensitivities"]["variables"].keymap:  # could this be done away with for backwards compatibility?
        if not isinstance(vk, VarKey) or (vk.shape and not vk.index):
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

def crawl_modelbd(bd, lookup, name="Model"):
    tree = Tree(name, bd.pop("|sensitivity|"), [])
    if bd:
        lookup[name] = tree
    for subname, subtree in sorted(bd.items(),
                                   key=lambda kv: (-float("%.2g" % kv[1]["|sensitivity|"]), kv[0])):
        tree.branches.append(crawl_modelbd(subtree, lookup, subname))
    return tree

def divide_out_vk(vk, pow, lt, gt):
    hmap = NomialMap({HashVector({vk: 1}): 1.0})
    hmap.units = vk.units
    var = Monomial(hmap)**pow
    lt, gt = lt/var, gt/var
    lt.ast = gt.ast = None
    return lt, gt

# @profile
def get_breakdowns(basically_fixed_variables, solution):
    """Returns {key: (lt, gt, constraint)} for breakdown constrain in solution.

    A breakdown constraint is any whose "gt" contains a single free variable.

    (At present, monomial constraints check both sides as "gt")
    """
    breakdowns = defaultdict(list)
    beatout = defaultdict(set)
    for constraint, senss in sorted(solution["sensitivities"]["constraints"].items(), key=lambda kv: (-abs(float("%.2g" % kv[1])), str(kv[0]))):
        while getattr(constraint, "child", None):
            constraint = constraint.child
        while getattr(constraint, "generated", None):
            constraint = constraint.generated
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
        for gtvk in gt.vks:  # remove RelaxPCCP.C
            if (gtvk.name == "C" and gtvk.lineage[0][0] == "RelaxPCCP"
                    and gtvk not in solution["freevariables"]):
                lt, gt = lt.sub({gtvk: 1}), gt.sub({gtvk: 1})
        if len(gt.hmap) > 1:
            continue
        pos_gtvks = {vk for vk, pow in gt.exp.items() if pow > 0}
        if len(pos_gtvks) > 1:
            pos_gtvks &= get_free_vks(gt, solution)  # remove constants
        if len(pos_gtvks) == 1:
            chosenvk, = pos_gtvks
            while getattr(constraint, "parent", None):
                constraint = constraint.parent
            while getattr(constraint, "generated_by", None):
                constraint = constraint.generated_by
            breakdowns[chosenvk].append((lt, gt, constraint))
    for constraint, senss in sorted(solution["sensitivities"]["constraints"].items(), key=lambda kv: (-abs(float("%.2g" % kv[1])), str(kv[0]))):
        if abs(senss) <= 1e-5:  # only tight-ish ones
            continue
        while getattr(constraint, "child", None):
            constraint = constraint.child
        while getattr(constraint, "generated", None):
            constraint = constraint.generated
        if constraint.oper == ">=":
            gt, lt = (constraint.left, constraint.right)
        elif constraint.oper == "<=":
            lt, gt = (constraint.left, constraint.right)
        elif constraint.oper == "=":
            if senss > 0:  # l_over_r is more sensitive - see nomials/math.py
                lt, gt = (constraint.left, constraint.right)
            else:  # r_over_l is more sensitive - see nomials/math.py
                gt, lt = (constraint.left, constraint.right)
        for gtvk in gt.vks:
            if (gtvk.name == "C" and gtvk.lineage[0][0] == "RelaxPCCP"
                    and gtvk not in solution["freevariables"]):
                lt, gt = lt.sub({gtvk: 1}), gt.sub({gtvk: 1})
        if len(gt.hmap) > 1:
            continue
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
                    key=lambda vk: (-float("%.2g" % (gt.exp[vk]*vrisk.get(vk, 0))), str(vk))
                )
                for vk, pow in gt.exp.items():
                    if vk is not chosenvk:
                        lt, gt = divide_out_vk(vk, pow, lt, gt)
                while getattr(constraint, "parent", None):
                    constraint = constraint.parent
                while getattr(constraint, "generated_by", None):
                    constraint = constraint.generated_by
                breakdowns[chosenvk].append((lt, gt, constraint))
    breakdowns = dict(breakdowns)  # remove the defaultdict-ness

    prevlen = None
    while len(basically_fixed_variables) != prevlen:
        prevlen = len(basically_fixed_variables)
        for key in breakdowns:
            if key not in basically_fixed_variables:
                get_fixity(basically_fixed_variables, key, breakdowns, solution)
    return breakdowns


def get_fixity(basically_fixed, key, bd, solution, visited=set()):
    lt, gt, _ = bd[key][0]
    free_vks = get_free_vks(lt, solution).union(get_free_vks(gt, solution))
    for vk in free_vks:
        if vk is key or vk in basically_fixed:
            continue  # currently checking or already checked
        if vk not in bd:
            return  # a very free variable, can't even be broken down
        if vk in visited:
            return  # tried it before, implicitly it didn't work out
        # maybe it's basically fixed?
        visited.add(key)
        get_fixity(basically_fixed, vk, bd, solution, visited)
        if vk not in basically_fixed:
            return  # ...well, we tried
    basically_fixed.add(key)

# @profile  # ~84% of total last check # TODO: remove
def crawl(basically_fixed_variables, key, bd, solution, basescale=1, permissivity=2, verbosity=0,
          visited_bdkeys=None, gone_negative=False, all_visited_bdkeys=None):
    "Returns the tree of breakdowns of key in bd, sorting by solution's values"
    if key != solution["cost function"] and hasattr(key, "key"):
        key = key.key  # clear up Variables
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
        all_visited_bdkeys = set()
    if verbosity == 1:
        already_set = False #not solution._lineageset TODO
        if not already_set:
            solution.set_necessarylineage()
    if verbosity:
        indent = verbosity-1  # HACK: a bit of overloading, here
        kvstr = "%s (%s)" % (key, get_valstr(key, solution))
        if key in all_visited_bdkeys:
            print("  "*indent + kvstr + " [as broken down above]")
            verbosity = 0
        else:
            print("  "*indent + kvstr)
            indent += 1
    orig_subtree = subtree = []
    tree = Tree(key, basescale, subtree)
    visited_bdkeys.add(key)
    all_visited_bdkeys.add(key)
    if keymon is None:
        scale = solution(key)/basescale
    else:
        if verbosity:
            print("  "*indent + "which in: "
                  + constraint.str_without(["units", "lineage"])
                  + " (sensitivity %+.2g)" % solution["sensitivities"]["constraints"][constraint])
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
                    print("  "*indent + "{ through a factor of %s (%s) }" %
                          (factor.str_without(["units"]),
                           get_valstr(factor, solution)))
                subsubtree = []
                transform = Transform(factor, 1, keymon)
                orig_subtree.append(Tree(transform, basescale, subsubtree))
                orig_subtree = subsubtree
            if power != 1:
                if verbosity:
                    print("  "*indent + "{ through a power of %.2g }" % power)
                subsubtree = []
                transform = Transform(1, 1/power, keymon)  # inverted bc it's on the gt side
                orig_subtree.append(Tree(transform, basescale, subsubtree))
                orig_subtree = subsubtree

    # TODO: use ast_parsing instead of chop?
    mons = composition.chop()
    monsols = [solution(mon) for mon in mons]  # ~20% of total last check # TODO: remove
    parsed_monsols = [getattr(mon, "value", mon) for mon in monsols]
    monvals = [float(mon/scale) for mon in parsed_monsols]   # ~10% of total last check # TODO: remove
    # sort by value, preserving order in case of value tie
    sortedmonvals = sorted(zip([-float("%.2g" % mv) for mv in monvals], range(len(mons)), monvals, mons))
    # print([m.str_without({"units", "lineage"}) for m in mons])
    if verbosity:
        if len(monsols) == 1:
            print("  "*indent + "breaks down into:")
        else:
            print("  "*indent + "breaks down into %i monomials:" % len(monsols))
            indent += 1
        indent += 1
    for i, (_, _, scaledmonval, mon) in enumerate(sortedmonvals):
        if not scaledmonval:
            continue
        subtree = orig_subtree  # return to the original subtree
        # time for some filtering
        interesting_vks = mon.vks
        potential_filters = [
            {vk for vk in interesting_vks if vk not in bd},
            mon.vks - get_free_vks(mon, solution),
            {vk for vk in interesting_vks if vk in basically_fixed_variables}
        ]
        if scaledmonval < 1 - permissivity:  # skip breakdown filter
            potential_filters = potential_filters[1:]
        potential_filters.insert(0, visited_bdkeys)
        for filter in potential_filters:
            if interesting_vks - filter:  # don't remove the last one
                interesting_vks = interesting_vks - filter
        # if filters weren't enough and permissivity is high enough, sort!
        if len(interesting_vks) > 1 and permissivity > 1:
            csenss = solution["sensitivities"]["constraints"]
            best_vks = sorted((vk for vk in interesting_vks if vk in bd),
                key=lambda vk: (-abs(float("%.2g" % (mon.exp[vk]*csenss[bd[vk][0][2]]))),
                                -float("%.2g" % solution["variables"][vk]),
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

        if len(monsols) > 1 and verbosity:
            indent -= 1
            print("  "*indent + "%s) forming %i%% of the RHS and %i%% of the total:" % (i+1, scaledmonval/basescale*100, scaledmonval*100))
            indent += 1

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
                    keyvalstr = "%s (%s)" % (factor.str_without(["units"]),
                                             get_valstr(factor, solution))
                    print("  "*indent + "{ through a factor of %s }" % keyvalstr)
                subsubtree = []
                transform = Transform(factor, 1, mon)
                subtree.append(Tree(transform, scaledmonval, subsubtree))
                subtree = subsubtree
            mon = freemon  # simplifies units
            if power != 1:
                if verbosity:
                    print("  "*indent + "{ through a power of %.2g }" % power)
                subsubtree = []
                transform = Transform(1, power, mon)
                subtree.append(Tree(transform, scaledmonval, subsubtree))
                subtree = subsubtree
                mon = mon**(1/power)
                mon.ast = None
        # TODO: make minscale an argument - currently an arbitrary 0.01
        if (subkey is not None and subkey not in visited_bdkeys
                and subkey in bd and scaledmonval > 0.05):
            subverbosity = indent + 1 if verbosity else 0  # slight hack
            subsubtree = crawl(basically_fixed_variables, subkey, bd, solution, scaledmonval,
                               permissivity, subverbosity, set(visited_bdkeys),
                               gone_negative, all_visited_bdkeys)
            subtree.append(subsubtree)
        else:
            if verbosity:
                keyvalstr = "%s (%s)" % (mon.str_without(["units"]),
                                         get_valstr(mon, solution))
                print("  "*indent + keyvalstr)
            subtree.append(Tree(mon, scaledmonval, []))
    if verbosity == 1:
        if not already_set:
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
        bvs = zip(*sorted(((-float("%.2g" % v), i, b, v) for i, (b, v) in enumerate(zip(branches, values)) if v is not None)))
        _, _, branches, values = bvs
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
                    vkeys = [(-float("%.2g" % v), str(k), k)]
                miscvkeys += vkeys
                surplus -= (round(scale*(miscval + v))
                            - round(scale*miscval) - subextent)
                miscval += v
        misckeys = tuple(k for _, _, k in sorted(miscvkeys))
        branches.append(Tree(misckeys, miscval, []))
        extents.append(int(round(scale*miscval)))
    if surplus:
        sign = int(np.sign(surplus))
        bump_priority = sorted((ext, sign*float("%.2g" % b.value), i) for i, (b, ext)
                               in enumerate(zip(branches, extents)))
        # print(key, surplus, bump_priority)
        while surplus:
            try:
                extents[bump_priority.pop()[-1]] += sign
                surplus -= sign
            except IndexError:
                raise ValueError(val, [b.value for b in branches])

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

def prune(tree, solution, maxlength, length=-1, prefix=""):
    "Prune branches that are longer than a certain number of characters"
    key, extent, branches = tree
    keylength = max(len(get_valstr(key, solution, into="(%s)")),
                    len(get_keystr(key, solution, prefix)))
    if length == -1 and isinstance(key, VarKey) and key.necessarylineage:
        prefix = key.lineagestr()
    length += keylength + 3
    for branch in branches:
        keylength = max(len(get_valstr(branch.key, solution, into="(%s)")),
                        len(get_keystr(branch.key, solution, prefix)))
        branchlength = length + keylength + 3
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
def graph(tree, breakdowns, solution, basically_fixed_variables, *,
          height=None, maxdepth=None, maxwidth=81, showlegend=False):
    "Prints breakdown"
    already_set = solution._lineageset
    if not already_set:
        solution.set_necessarylineage()
    collapse = (not showlegend)  # TODO: set to True while showlegend is True for first approx of receipts; autoinclude with trace?
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
    prefix = ""
    if A_key is solution["cost function"]:
        A_str = "Cost"
    else:
        A_str = get_keystr(A_key, solution)
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
                    chararray[depth, pos] =  "*" + chararray[depth, pos]
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
            if key in breakdowns and not chararray[depth+1, pos].strip():
                keystr = keystr + "╶⎨"
            chararray[depth, pos] = fmt.format(linkstr + keystr)
    # Rotate and print
    rowstrs = ["".join(row).rstrip() for row in chararray.T.tolist()]
    print("\n" + "\n".join(rowstrs) + "\n")

    if showlegend:  # create and print legend
        legend_lines = []
        for key, shortname in sorted(legend.items(), key=lambda kv: kv[1]):
            legend_lines.append(legend_entry(key, shortname, solution, prefix,
                                             basically_fixed_variables))
        maxlens = [max(len(el) for el in col) for col in zip(*legend_lines)]
        fmts = ["{0:<%s}" % L for L in maxlens]
        for line in legend_lines:
            line = "".join(fmt.format(cell)
                           for fmt, cell in zip(fmts, line) if cell).rstrip()
            print(" " + line)

    if not already_set:
        solution.set_necessarylineage(clear=True)

def legend_entry(key, shortname, solution, prefix, basically_fixed_variables):
    "Returns list of legend elements"
    operator = note = ""
    keystr = valuestr = " "
    operator = "= " if shortname else "  + "
    if is_factor(key):
        operator = " ×"
        key = key.factor
        free, quasifixed = False, False
        if any(vk not in basically_fixed_variables
               for vk in get_free_vks(key, solution)):
            note = "  [free factor]"
    if is_power(key):
        valuestr = "   ^%.3g" % key.power
    else:
        valuestr = get_valstr(key, solution, into="  "+operator+"%s")
        if not isinstance(key, FixedScalar):
            keystr = get_keystr(key, solution, prefix)
    return ["%-4s" % shortname, keystr, valuestr, note]

def get_keystr(key, solution, prefix=""):
    "Returns formatted string of the key in solution."
    if hasattr(key, "str_without"):
        out = key.str_without({"units", ":MAGIC:"+prefix})
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


import plotly.graph_objects as go
def plotlyify(tree, solution, minval=None):
    """Plots model structure as Plotly TreeMap

    Arguments
    ---------
    model: Model
        GPkit model object

    itemize (optional): string, either "variables" or "constraints"
        Specify whether to iterate over the model varkeys or constraints

    sizebycount (optional): bool
        Whether to size blocks by number of variables/constraints or use
        default sizing

    Returns
    -------
    plotly.graph_objects.Figure
        Plot of model hierarchy

    """
    ids = []
    labels = []
    parents = []
    values = []

    key, value, branches = tree
    if isinstance(key, VarKey) and key.necessarylineage:
        prefix = key.lineagestr()
    else:
        prefix = ""

    if minval is None:
        minval = value/1000

    parent_budgets = {}

    def crawl(tree, parent_id=None):
        key, value, branches = tree
        if value > minval:
            if isinstance(key, Transform):
                id = parent_id
            else:
                id = len(ids)+1
                ids.append(id)
                labels.append(get_keystr(key, solution, prefix))
                if not isinstance(key, str):
                    labels[-1] = labels[-1] + "<br>" + get_valstr(key, solution)
                parents.append(parent_id)
                parent_budgets[id] = value
                if parent_id is not None:  # make sure there's no overflow
                    if parent_budgets[parent_id] < value:
                        value = parent_budgets[parent_id]  # take remained
                    parent_budgets[parent_id] -= value
                values.append(value)
            for branch in branches:
                crawl(branch, id)

    crawl(tree)
    return ids, labels, parents, values

def treemap(ids, labels, parents, values):
    return go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total"
    ))

def icicle(ids, labels, parents, values):
    return go.Figure(go.Icicle(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total"
    ))


import functools

class Breakdowns(object):
    def __init__(self, sol):
        self.sol = sol
        self.mlookup = {}
        self.mtree = crawl_modelbd(get_model_breakdown(sol), self.mlookup)
        self.basically_fixed_variables = set()
        self.bd = get_breakdowns(self.basically_fixed_variables, self.sol)

    def trace(self, key, *, permissivity=2):
        print("")  # a little padding to start
        self.get_tree(key, permissivity=permissivity, verbosity=1)

    def get_tree(self, key, *, permissivity=2, verbosity=0):
        tree = None
        kind = "variable"
        if isinstance(key, str):
            if key == "model sensitivities":
                tree = self.mtree
                kind = "constraint"
            elif key == "cost":
                key = self.sol["cost function"]
            elif key in self.mlookup:
                tree = self.mlookup[key]
                kind = "constraint"
            else:
                # TODO: support submodels
                keys = [vk for vk in self.bd if key in str(vk)]
                if not keys:
                    raise KeyError(key)
                elif len(keys) > 1:
                    raise KeyError("There are %i keys containing '%s'." % (len(keys), key))
                key, = keys
        if tree is None:
            tree = crawl(self.basically_fixed_variables, key, self.bd, self.sol,
                         permissivity=permissivity, verbosity=verbosity)
        return tree, kind

    def plot(self, key, *, height=None, permissivity=2, showlegend=False,
             maxwidth=85):
        tree, kind = self.get_tree(key, permissivity=permissivity)
        lookup = self.bd if kind == "variable" else self.mlookup
        graph(tree, lookup, self.sol, self.basically_fixed_variables,
              height=height, showlegend=showlegend, maxwidth=maxwidth)

    def treemap(self, key, *, permissivity=2, returnfig=False, filename=None):
        tree, _ = self.get_tree(key)
        fig = treemap(*plotlyify(tree, self.sol))
        if returnfig:
            return fig
        if filename is None:
            filename = str(key)+"_treemap.html"
            keepcharacters = (".","_")
            filename = "".join(c for c in filename if c.isalnum()
                               or c in keepcharacters).rstrip()
        import plotly
        plotly.offline.plot(fig, filename=filename)


    def icicle(self, key, *, permissivity=2, returnfig=False, filename=None):
        tree, _ = self.get_tree(key, permissivity=permissivity)
        fig = icicle(*plotlyify(tree, self.sol))
        if returnfig:
            return fig
        if filename is None:
            filename = str(key)+"_icicle.html"
            keepcharacters = (".","_")
            filename = "".join(c for c in filename if c.isalnum()
                               or c in keepcharacters).rstrip()
        import plotly
        plotly.offline.plot(fig, filename=filename)

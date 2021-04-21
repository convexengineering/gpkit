import string
from collections import defaultdict, namedtuple
from gpkit.nomials import Monomial, Posynomial
from gpkit.nomials.map import NomialMap
from gpkit.small_scripts import mag
from gpkit.small_classes import FixedScalar, HashVector
from gpkit.exceptions import DimensionalityError
from gpkit.repr_conventions import unitstr as get_unitstr
import numpy as np


Transform = namedtuple("Transform", ["factor", "power", "origkey"])

def get_freevks(posy, solution):
    "Returns all free vks of a given posynomial for a given solution"
    return set(vk for vk in posy.vks if vk not in solution["constants"])

def get_breakdowns(solution):
    """Returns {key: (lt, gt, constraint)} for breakdown constrain in solution.

    A breakdown constraint is any whose "gt" contains a single free variable.

    (At present, monomial constraints check both sides as "gt")
    """
    if solution is None:
        return
    breakdowns = defaultdict(list)
    for constraint, senss in solution["sensitivities"]["constraints"].items():
        # TODO: should also check tightness by value, or?
        if senss <= 1e-5:  # only tight-ish ones
            continue
        if constraint.oper == ">=":
            gt_lts = [(constraint.left, constraint.right)]
        elif constraint.oper == "<=":
            gt_lts = [(constraint.right, constraint.left)]
        elif constraint.oper == "=":
            gt_lts = [(constraint.right, constraint.left),
                      (constraint.left, constraint.right)]
        for gt, lt in gt_lts:
            if lt.any_nonpositive_cs:
                continue  # no signomials
            freegt_vks = get_freevks(gt, solution)
            if len(freegt_vks) != 1:
                continue  # not a breakdown constraint
            brokendownvk, = freegt_vks
            if gt.exp[brokendownvk] < 0:
                if constraint.oper == "=" or len(lt.hmap) != 1:
                    continue
                # We can try flipping gt/lt to make a breakdown.
                freelt_vks = get_freevks(lt, solution)
                if len(lt.hmap) != 1 or len(freelt_vks) != 1:
                    continue
                brokendownvk, = freelt_vks
                if lt.exp[brokendownvk] > 0:
                    continue  # not a breakdown constraint after transformation
                gt, lt = 1/lt, 1/gt
            breakdowns[brokendownvk].append((lt, gt, constraint))
    for key, bds in breakdowns.items():
        # TODO: do multiple if sensitivities are quite close? right now we have to break ties!
        if len(bds) > 1:
            bds.sort(key=lambda lgc: (solution["sensitivities"]["constraints"][lgc[2]], str(lgc[0])), reverse=True)
    return dict(breakdowns)  # remove the defaultdict-ness

def crawl(key, bd, solution, basescale=1, verbosity=0, visited_bdkeys=None):
    "Returns the tree of breakdowns of key in bd, sorting by solution's values"
    if key in bd:
        # TODO: do multiple if sensitivities are quite close?
        composition, keymon, _ = bd[key][0]
    elif isinstance(key, Posynomial):
        composition = key
        keymon = None
    else:
        raise TypeError("the `key` argument must be a VarKey or Posynomial.")

    if visited_bdkeys is None:
        visited_bdkeys = set()
    if verbosity:
        indent = verbosity-1  # HACK: a bit of overloading, here
        keyvalstr = "%s (%s)" % (key.str_without(["lineage", "units"]),
                                 get_valstr(key, solution))
        print("  "*indent + keyvalstr + ", which breaks down further")
        indent += 1
    orig_subtree = subtree = []
    tree = {(key, basescale): subtree}
    visited_bdkeys.add(key)
    if keymon is None:
        scale = solution(key)/basescale
    else:
        mon_freevks = set(get_freevks(keymon, solution))
        subkey, = mon_freevks
        power = 1/keymon.exp[subkey]  # inverted bc it's on the gt side
        fixed_vks = set(keymon.vks) - mon_freevks
        scale = solution(key)**(1/power)/basescale
        if power != 1 or fixed_vks or mag(keymon.c) != 1:
            units = 1
            exp = HashVector()
            for vk in mon_freevks:
                exp[vk] = keymon.exp[vk]
                if vk.units:
                    units *= vk.units**keymon.exp[vk]
            subhmap = NomialMap({exp: 1})
            subhmap.units = None if units == 1 else units
            freemon = Monomial(subhmap)
            factor = Monomial(freemon/keymon)  # inverted bc it's on the gt side
            factor.ast = None
            if factor != 1:
                factor = factor**power  # HACK: done here to prevent odd units issue
                factor.ast = None
                if verbosity:
                    keyvalstr = "%s (%s)" % (factor.str_without(["lineage", "units"]),
                                             get_valstr(factor, solution))
                    print("  "*indent + "(with a factor of " + keyvalst + " )")
                subsubtree = []
                transform = Transform(factor, 1, keymon)
                orig_subtree.append({(transform, basescale): subsubtree})
                scale = scale/solution(factor)
                orig_subtree = subsubtree
            if power != 1:
                if verbosity:
                    print("  "*indent + "(with a power of %.2g )" % power)
                subsubtree = []
                transform = Transform(1, power, keymon)
                orig_subtree.append({(transform, basescale): subsubtree})
                orig_subtree = subsubtree
    if verbosity:
        if constraint is not None:
            print("  "*indent + "in: " + constraint.str_without(["units", "lineage"]))
            indent += 1
        print("  "*indent + "by\n")
        indent += 1

    try:
        # TODO: use ast_parsing instead of chop?
        monsols = [solution(mon) for mon in composition.chop()]
        parsed_monsols = [getattr(mon, "value", mon) for mon in monsols]
        monvals = [float(mon/scale) for mon in parsed_monsols]
        # sort by value, preserving order in case of value tie
        sortedmonvals = sorted(zip(monvals, range(len(monvals)),
                                   composition.chop()), reverse=True)
    except DimensionalityError:
        # fails in numerical edge-cases for fits...
        return tree  # TODO: a more graceful failure mode?

    for scaledmonval, _, mon in sortedmonvals:
        subtree = orig_subtree  # revert back to the original subtree
        mon_freevks = get_freevks(mon, solution)
        further_recursion_allowed = True
        for vk in list(mon_freevks):
            # free variables are allowed as factors...
            if vk not in bd or mon.exp[vk] < 0: # or not mon_units_same_as_vk:
                mon_freevks.remove(vk)
                # but recursion ends here if it's not a relatively huge mon
                further_recursion_allowed = (scaledmonval > 0.4)
        if len(mon_freevks) > 1 and scaledmonval > 0.4:
            fixed_breakdown_vks = set()
            for vk in mon_freevks:
                subcomposition, _, _ = bd[vk][0]
                if not any(vk in bd for vk in subcomposition.vks):
                    fixed_breakdown_vks.add(vk)
            if len(fixed_breakdown_vks) == len(mon_freevks) - 1:
                mon_freevks = mon_freevks - fixed_breakdown_vks
        fixed_vks = set(mon.vks) - mon_freevks

        if len(mon_freevks) == 1 and further_recursion_allowed:
            subkey, = mon_freevks
            power = mon.exp[subkey]
            if power < 0: # != 1:
                if subkey in bd:
                    posy, _, _ = bd[subkey][0]
                    further_recursion_allowed = (len(posy.hmap) == 1)
                if not further_recursion_allowed and subkey not in visited_bdkeys:
                    power = 1  # so that it's not made into a transform
        else:
            subkey = None
            power = 1
            if not mon_freevks:
                # prioritize showing some fixed_vks as if they were "free"
                if len(fixed_vks) == 1:
                    mon_freevks = fixed_vks
                    fixed_vks = set()
                else:
                    for vk in list(fixed_vks):
                        if vk.units and not vk.units.dimensionless:
                            mon_freevks.add(vk)
                            fixed_vks.remove(vk)

        if mon_freevks and (fixed_vks or mag(mon.c) != 1):
            if subkey:
                kindafree_vks = set(vk for vk in fixed_vks
                                    if vk not in solution["constants"])
                if kindafree_vks == fixed_vks:
                    kindafree_vks = set()  # don't remove ALL of them
                else:
                    mon_freevks.update(kindafree_vks)
            units = 1
            exp = HashVector()
            for vk in mon_freevks:
                exp[vk] = mon.exp[vk]
                if vk.units:
                    units *= vk.units**mon.exp[vk]
            subhmap = NomialMap({exp: 1})
            subhmap.units = None if units is 1 else units
            freemon = Monomial(subhmap)
            factor = mon/freemon  # autoconvert...
            factor.ast = None
            if (factor.units is None and isinstance(factor, FixedScalar)
                    and abs(factor.value - 1) <= 1e-4):
                factor = 1  # minor fudge to clear numerical inaccuracies
            if factor != 1 :
                if verbosity:
                    keyvalstr = "%s (%s)" % (factor.str_without(["lineage", "units"]),
                                             get_valstr(factor, solution))
                    print("  "*indent + "(with a factor of %s )" % keyvalstr)
                subsubtree = []
                transform = Transform(factor, 1, mon)
                subtree.append({(transform, scaledmonval): subsubtree})
                subtree = subsubtree
            if power != 1:
                if verbosity:
                    print("  "*indent + "(with a power of %.2g )" % power)
                subsubtree = []
                transform = Transform(1, power, mon)
                subtree.append({(transform, scaledmonval): subsubtree})
                subtree = subsubtree
            mon = freemon**(1/power)
            mon.ast = None
            if subkey and fixed_vks and kindafree_vks:
                units = 1
                exp = HashVector()
                for vk in kindafree_vks:
                    exp[vk] = mon.exp[vk]
                    if vk.units:
                        units *= vk.units**mon.exp[vk]
                subhmap = NomialMap({exp: 1})
                subhmap.units = None if units == 1 else units
                factor = Monomial(subhmap)
                factor.ast = None
                if factor != 1:
                    if verbosity:
                        keyvalstr = "%s (%s)" % (factor.str_without(["lineage", "units"]),
                                                 get_valstr(factor, solution))
                        print("  "*indent + "(with a factor of " + keyvalst + " )")
                    subsubtree = []
                    transform = Transform(factor, 1, mon)
                    subtree.append({(transform, scaledmonval): subsubtree})
                    subtree = subsubtree
                mon = mon/factor
                mon.ast = None
        # TODO: make minscale an argument - currently an arbitrary 0.01
        # power > 0 prevents inversion during recursion
        if (subkey is not None and subkey not in visited_bdkeys
                and subkey in bd and further_recursion_allowed and scaledmonval > 0.01):
            if verbosity:
                verbosity = indent  # slight HACK
            subsubtree = crawl(subkey, bd, solution, scaledmonval,
                               verbosity, visited_bdkeys)
            subtree.append(subsubtree)
        else:
            if verbosity:
                keyvalstr = "%s (%s)" % (mon.str_without(["lineage", "units"]),
                                         get_valstr(mon, solution))
                print("  "*indent + keyvalstr)
            subtree.append({(mon, scaledmonval): []})
    # TODO: or, instead of AST parsing, remove common factors at the composition level?
    ftidxs = defaultdict(list)
    for i, ((key, _),) in enumerate(orig_subtree):
        if isinstance(key, Transform):
            ftidxs[(key.factor, key.power)].append(i)
    to_delete = []
    to_insert = []
    for idxs in ftidxs.values():
        if len(idxs) > 1:
            valsum = 0
            newbranches = []
            new_origkeys = []
            for idx in idxs:
                ((key, val), subbranches), = orig_subtree[idx].items()
                valsum += val
                new_origkeys.append(key.origkey)
                newbranches.extend(subbranches)
            to_delete.extend(idxs)
            newkey = Transform(key.factor, key.power, tuple(new_origkeys))
            to_insert.append({(newkey, valsum): newbranches})
    for idx in sorted(to_delete, reverse=True): # high to low
        orig_subtree.pop(idx)
    if to_insert:
        orig_subtree.extend(to_insert)
        orig_subtree.sort(reverse=True,
                          key=lambda branch: list(branch.keys())[0][1])
    return tree

SYMBOLS = string.ascii_uppercase + string.ascii_lowercase
for ambiguous_symbol in "lILT":
    SYMBOLS = SYMBOLS.replace(ambiguous_symbol, "")
SYMBOLS += "⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵"

def get_spanstr(legend, length, label, leftwards, solution):
    "Returns span visualization, collapsing labels to symbols"
    spacer, lend, rend = "│", "┯", "┷"
    if isinstance(label, Transform):
        spacer, lend, rend = "╎", "╤", "╧"
        # if label.power != 1:
        #     spacer = "^"
        # remove origkey so they collide in the legends dictionary
        label = Transform(label.factor, label.power, None)
        # TODO: catch PI (or again could that come from AST parsing?)
        if label.power == 1 and len(str(label.factor)) == 1:
            legend[label] = str(label.factor)

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
          # HACK: no corners on rightwards
        return "┃"*(longside+1) + shortname + "┃"*(shortside+1)

def layer(map, tree, extent, depth=0, maxdepth=20):
    "Turns the tree into a 2D-array"
    ((key, val), branches), = tree.items()
    if not val:
        return map
    if len(map) <= depth:
        map.append([])
    scale = extent/val
    if extent == 1 and not isinstance(key, Transform):
        branches = []
    map[depth].append((key, extent))
    if not any(round(scale*v) for (_, v), in branches):
        branches = []
    if depth <= maxdepth:
        subvalsum = 0
        for branch in branches:
            (_, subval), = branch.keys()
            subvalsum += subval
        if round(scale*(val-subvalsum)):
            branches = [{(None, val - subvalsum): []}] + branches
    if depth > maxdepth or not branches:
        return map
    extents = [round(scale*v) for (_, v), in branches]
    # TODO: make the below optional?
    if not all(extents):
        if not round(sum(scale*v for (_, v), in branches if not round(scale*v))):
            extents = [e for e in extents if e]
            branches = branches[:len(extents)]
    surplus = extent - sum(extents)
    scaled = np.array([scale*v for (_, v), in branches]) % 1
    gain_targets = sorted([(s, i) for (i, s) in enumerate(scaled) if s > 0.5])
    while surplus < (1 - all(extents)):
        extents[gain_targets.pop(0)[1]] -= 1  # smallest & closest to 0.5
        surplus += 1
    loss_targets = sorted([(s, i) for (i, s) in enumerate(scaled) if s < 0.5])
    while surplus > (1 - all(extents)):
        extents[loss_targets.pop()[1]] += 1  # largest & closest to 0.5
        surplus -= 1
    if not all(extents):
        grouped_keys = ()
        for i, branch in enumerate(branches):
            if not extents[i]:
                (k, _), = branch
                if isinstance(k, Transform):
                    k = k.origkey  # TODO: this is the only use of origkey - remove it
                if not isinstance(k, tuple):  # TODO: if it is, may be out of sort-order
                    k = (k,)
                grouped_keys = grouped_keys + k
        if len(grouped_keys) == 1:
            grouped_keys, = grouped_keys
        branches = branches + [{(grouped_keys, 1/scale): []}]
        extents.append(1)
    for branch, subextent in zip(branches, extents):
        if subextent:
            # TODO: decide on this and the linked section above -- worth the hiding of leaves?
            ((k, v), bs), = branch.items()
            if (isinstance(k, Transform)  # transform with no worthy heirs
                    and not any(round(subextent/v*subv) for (_, subv), in bs)):
                branch = {(None, v): []}  # don't even show it
            layer(map, branch, subextent, depth+1, maxdepth)
    return map

def plumb(tree, depth=0):
    "Finds maximum depth of a tree"
    ((key, val), branches), = tree.items()
    maxdepth = depth
    for branch in branches:
        maxdepth = max(maxdepth, plumb(branch, depth+1))
    return maxdepth

def graph(tree, solution, extent=20):
    "Displays text plot of breakdown(s)"
    graph_printer(tree, solution, extent)
    # ((key, val), branches), = tree.items()
    # for i, branch in enumerate(branches):
    #     if len(branches) > 1:
    #         print("%i/%i:" % (i+1, len(branches)))
    #     graph_printer({(key, val): [branch]}, extent)
    #     print("\n")

def graph_printer(tree, solution, extent):
    "Prints breakdown"
    mt = layer([], tree, extent, maxdepth=plumb(tree)-1)
    max_width = max(len(at_depth) for at_depth in mt)
    if max_width*4 <= extent:
        extent = 4*max_width
    mt = layer([], tree, extent, maxdepth=plumb(tree)-1)
    scale = 1/extent
    legend = {}
    chararray = np.full((len(mt), extent), "", "object")
    for depth, elements_at_depth in enumerate(mt):
        row = ""
        for i, (element, length) in enumerate(elements_at_depth):
            if element is None:
                row += " "*length
                continue
            leftwards = depth > 0 and length > 2
            row += get_spanstr(legend, length, element, leftwards, solution)
        if row.strip():
            chararray[depth, :] = list(row)

    # Format depth=0
    A_key, = [key for key, value in legend.items() if value == "A"]
    if hasattr(A_key, "legendlabel") and A_key.legendlabel:
        A_str = A_key.legendlabel
    else:
        A_str = A_key.str_without(["lineage", "units"])
    A_valstr = "(%s)" % get_valstr(A_key, solution)
    fmt = "{0:>%s}" % (max(len(A_str), len(A_valstr)) + 3)
    for j, entry in enumerate(chararray[0,:]):
        if entry == "A":
            chararray[0,j] = fmt.format(A_str + "╺┫")
            chararray[0,j+1] = fmt.format(A_valstr + " ┃")
        else:
            chararray[0,j] = fmt.format(entry)

    # Format depths 1+
    new_legend = {}
    for pos in range(extent):
        for depth in reversed(range(1,len(mt))):
            value = chararray[depth, pos]
            if value == " ":  # spacer character
                chararray[depth, pos] = ""
                continue
            elif not value or value not in SYMBOLS:
                continue
            key, = [k for k, val in legend.items() if val == value]
            if getattr(key, "vks", None) and len(key.vks) == 1 and all(vk in new_legend for vk in key.vks):
                key, = key.vks
            if key not in new_legend and (isinstance(key, tuple) or (depth != len(mt) - 1 and chararray[depth+1, pos] != "")):
                new_legend[key] = SYMBOLS[len(new_legend)]
            if key in new_legend:
                chararray[depth, pos] = new_legend[key]
                if isinstance(key, tuple) and not isinstance(key, Transform):
                    chararray[depth, pos] =  "*" + chararray[depth, pos]
                continue
            tryup, trydn = True, True
            span = 0
            while tryup or trydn:
                span += 1
                if tryup:
                    if pos - span < 0:
                        tryup = False
                    else:
                        upchar = chararray[depth, pos-span]
                        if upchar == "│":
                            chararray[depth, pos-span] = "┃"
                        elif upchar == "┯":
                            chararray[depth, pos-span] = "┓"
                        else:
                            tryup = False
                if trydn:
                    if pos + span >= extent:
                        trydn = False
                    else:
                        dnchar = chararray[depth, pos+span]
                        if dnchar == "│":
                            chararray[depth, pos+span] = "┃"
                        elif dnchar == "┷":
                            chararray[depth, pos+span] = "┛"
                        else:
                            trydn = False
            keystr = key.str_without(["lineage", "units"])
            if key in bd or (key.vks and any(vk in bd for vk in key.vks)):
                linkstr = "┣┉"
            else:
                linkstr = "┣╸"
            if not isinstance(key, FixedScalar):
                valuestr = " (%s)" % get_valstr(key, solution)
                if span > 1 and (pos + 2 >= extent or chararray[depth, pos+1] == "┃"):
                    chararray[depth, pos+1] += valuestr
                else:
                    keystr += valuestr
            chararray[depth, pos] = linkstr + keystr

    # Rotate and print
    vertstr = "\n".join("    " + "".join(row) for row in chararray.T.tolist())
    print()
    print(vertstr)
    print()
    legend = new_legend

    # Create and print legend
    legend_lines = []
    for key, shortname in sorted(legend.items(), key=lambda kv: kv[1]):
        if isinstance(key, tuple) and not isinstance(key, Transform):
            asterisk, *others = key
            legend_lines.append(legend_entry(asterisk, shortname, solution))
            for k in others:
                legend_lines.append(legend_entry(k, "", solution))
        else:
            legend_lines.append(legend_entry(key, shortname, solution))
    maxlens = [max(len(el) for el in col) for col in zip(*legend_lines)]
    fmts = ["{0:<%s}" % L for L in maxlens]
    for line in legend_lines:
        line = "".join(fmt.format(cell)
                       for fmt, cell in zip(fmts, line) if cell).rstrip()
        print("    " + line)

def legend_entry(key, shortname, solution):
    "Returns list of legend elements"
    operator = note = ""
    keystr = valuestr = " "
    operator = "= " if shortname else "  + "
    if isinstance(key, Transform):
        if key.power == 1:
            operator = " ×"
            key = key.factor
            if get_freevks(key, solution):
                note = "  [free factor]"
        else:
            valuestr = "   ^%.3g" % key.power
            key = None
    if key is not None:
        if isinstance(key, FixedScalar):
            keystr = " "
        elif hasattr(key, "legendlabel") and key.legendlabel:
            keystr = key.legendlabel
        else:
            keystr = key.str_without(["lineage", "units"])
        valuestr = "  "+operator + get_valstr(key, solution)
    return ["%-4s" % shortname, keystr, valuestr, note]

def get_valstr(key, solution):
    "Returns formatted string of the value of key in solution."
    value = solution(key)
    if isinstance(value, FixedScalar):
        value = value.value
    value = mag(value)
    unitstr = key.unitstr()
    if unitstr[:2] == "1/":
        unitstr = "/" + unitstr[2:]
    if 1e3 <= value < 1e6:
        valuestr = "{:,.0f}".format(value)
    else:
        valuestr = "%-.3g" % value
    return valuestr + unitstr


import pickle
from gpkit import ureg
ureg.define("pax = 1")
ureg.define("paxkm = km")
ureg.define("trip = 1")

print("STARTING...")

sol = pickle.load(open("bd.p", "rb"))

bd = get_breakdowns(sol)

from gpkit.tests.helpers import StdoutCaptured

import difflib

keys = sorted((key for key in bd.keys() if not key.idx or len(key.shape) == 1),
              key=lambda k: str(k))

# with StdoutCaptured("breakdowns.log"):
#     for key in keys:
#         tree = crawl(key, bd, sol)
#         graph(tree, sol)

with StdoutCaptured("breakdowns.log.new"):
    for key in keys:
        tree = crawl(key, bd, sol)
        graph(tree, sol)

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

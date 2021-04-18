import string
from collections import defaultdict, namedtuple
from gpkit.nomials import Monomial
from gpkit.nomials.map import NomialMap
from gpkit.small_scripts import mag
from gpkit.small_classes import FixedScalar, HashVector
from gpkit.exceptions import DimensionalityError
import numpy as np


Transform = namedtuple("Transform", ["factor", "power", "origkey"])

def free_vks(self, obj):
    return [vk for vk in obj.vks if vk not in self.substitutions]

def parse_breakdowns(self):
    if self.solution is None:
        return
    breakdowns = defaultdict(set)
    for constraint in self.flat():
        if constraint.oper == ">=":
            gt, lt = constraint.left, constraint.right
        elif constraint.oper == "<=":
            lt, gt = constraint.left, constraint.right
        else:
            continue
        freegt_vks = free_vks(self, gt)
        if len(freegt_vks) != 1:
            continue
        var, = freegt_vks
        if gt.exp[var] < 0:  # flip gt/lt in this case TODO: probably remove
            freelt_vks = free_vks(self, lt)
            if len(lt.hmap) != 1 or len(freelt_vks) != 1:
                continue
            var, = freelt_vks
            if lt.exp[var] > 0:
                continue
            gt, lt = 1/lt, 1/gt
        if lt.any_nonpositive_cs:  # no signomials
            continue
        # if var.idx:  # spam filter
        #     continue
        # TODO: just iterate through the sol lol
        if self.solution["sensitivities"]["constraints"].get(constraint, 0) <= 1e-5:  # only tight ones
            continue  # if it's not in the dict assume it got optimized out
        breakdowns[var].add((lt, gt, constraint))
    return dict(breakdowns)
# bd = parse_breakdowns(m)
# key, = [k for k in bd if k.name == "podfrontalarea"]
def monstr(mon):
    freevks = free_vks(m, mon)
    if freevks:
        type = "%i free variables" % len(freevks)
    elif mon.vks:
        type = "fully fixed"
    else:  # it's a number
        return mon.str_without(["lineage", "units"]) + mon.unitstr()
    return "%s (%s, %s)" % (mon.str_without(["lineage", "units"]), type, valstr(mon))
def keystr(key):
    if key in m.solution["constants"]:
        value = m.solution["constants"][key]
        type = "fixed"
    elif key in m.solution["freevariables"]:
        value = m.solution["freevariables"][key]
        type = "free"
    return "%s (%s, %s)" % (key.str_without("lineage"), type, valstr(key))


# for key, value in bd.items():
#     if len(value) > 1:
#         sc = sorted(((m.solution["sensitivities"]["constraints"][constraint], constraint)
#                       for _, _, constraint in value), reverse=True)
#         print(key)
#         for s, c in sc:
#             print("%.4f" % s, c)
#         print()


def crawl(key, bd, basescale=1, indent=0, visited_bdkeys=None):
    if key in bd:
        # sort by tightness; only do multiple if they're quite close!
        composition, keymon, constraint = sorted(bd[key], key=lambda pmc: m.solution["sensitivities"]["constraints"][pmc[2]], reverse=True)[0]
    else:
        composition = key
        constraint = keymon = None

    if visited_bdkeys is None:
        visited_bdkeys = set()
    if keymon is None:
        out = "  "*indent + monstr(key) + ", which breaks down further in:\n"
    else:
        out = "  "*indent + keystr(key) + ", which breaks down further in:\n"
    indent += 1
    orig_subtree = subtree = []
    tree = {(key, basescale): subtree}
    visited_bdkeys.add(key)
    # for composition, keymon, constraint in bd[key]:
    # TODO: remove code duplication below
    if keymon is None:
        power = 1
        factor = False
    else:
        free_monvks = set(free_vks(m, keymon))
        subkey, = free_monvks
        power = 1/keymon.exp[subkey]  # inverted bc it's on the gt side
        fixed_vks = set(keymon.vks) - free_monvks
        factor = (power != 1 or fixed_vks or mag(keymon.c) != 1)
    scale = m.solution(key)**(1/power)/basescale

    if factor:
        units = 1
        exp = HashVector()
        for vk in free_monvks:
            exp[vk] = keymon.exp[vk]
            if vk.units:
                units *= vk.units**keymon.exp[vk]
        subhmap = NomialMap({exp: 1})
        subhmap.units = None if units == 1 else units
        freemon = Monomial(subhmap)
        factor = Monomial(freemon/keymon)  # inverted bc it's on the gt side
        factor.ast = None
        if factor != 1:
            out += "  "*indent + "(with a factor of " + monstr(factor) + ")\n"
            subsubtree = []
            transform = Transform(factor**power, 1, keymon)
            orig_subtree.append({(transform, basescale): subsubtree})
            scale = scale/m.solution(factor)
            orig_subtree = subsubtree
        if power != 1:
            out += "  "*indent + "(and a power of %.2g )\n" %power
            subsubtree = []
            transform = Transform(1, power, keymon)
            orig_subtree.append({(transform, basescale): subsubtree})
            orig_subtree = subsubtree
    if constraint is not None:
        out += "  "*indent + constraint.str_without(["units", "lineage"]) + "\n"
        indent += 1
    else:
        out += "  "*indent + "from input %s" % key.str_without(["units", "lineage"])
        indent += 1
    out += "  "*indent + "by\n"
    indent += 1
    # TODO: mag should be float but it fails in numerical edge-cases for fits...
    try:
        monvals = [float(getattr(m.solution(mon), "value", m.solution(mon))/scale)
                   for mon in composition.chop()]
    except DimensionalityError:
        out += "UGHH NUMERICAL ERROR IN DIMENSIONS\n"
        return out, tree
    # TODO: use ast_parsing instead of chop?
    for scaledmonval, _, mon in sorted(zip(monvals, range(len(monvals)), composition.chop()), reverse=True):
        subtree = orig_subtree
        free_monvks = set(free_vks(m, mon))
        altered_vks = False
        for vk in list(free_monvks):
            # NOTE: very strong restriction - no unit conversion
            # try:
            #     float((vk.units or 1)/(mon.units or 1))
            #     mon_units_same_as_vk = True
            # except DimensionalityError:
            #     mon_units_same_as_vk = False
            # NOTE: permissive, allows free variables as factors
            if vk not in bd or mon.exp[vk] < 0: # or not mon_units_same_as_vk:
                free_monvks.remove(vk)
                altered_vks = True
            # NOTE: in-between, mostly restrictive; do this, but don't allow recursion
        # NOTE: VERY permissive, always chooses an arbitrary breakdown to follow
        # if len(free_monvks) > 1:
        #     free_monvks = set([list(free_monvks)[0]])
        fixed_vks = set(mon.vks) - free_monvks
        subkey = None
        power = 1
        # NOTE: more permissive, descends past a free factor if it's a big factor of the original key
        if len(free_monvks) == 1 and (not altered_vks or scaledmonval > 0.4):
            subkey, = free_monvks
            power = mon.exp[subkey]
        elif not free_monvks:
            if len(fixed_vks) == 1:
                free_monvks = fixed_vks
                fixed_vks = set()
            else:
                for vk in list(fixed_vks):
                    if vk.units and not vk.units.dimensionless:
                        free_monvks.add(vk)
                        fixed_vks.remove(vk)

        if (free_monvks and fixed_vks) or mag(mon.c) != 1:
            units = 1
            exp = HashVector()
            for vk in free_monvks:
                exp[vk] = mon.exp[vk]
                if vk.units:
                    units *= vk.units**mon.exp[vk]
            subhmap = NomialMap({exp: 1})
            subhmap.units = None if units == 1 else units
            freemon = Monomial(subhmap)
            factor = mon/freemon  # autoconvert...
            factor.ast = None
            if factor.units is None and isinstance(factor, FixedScalar) and abs(factor.value - 1) <= 1e-4:
                factor = 1  # minor fudge to clear up numerical inaccuracies
            if power > 0 and freemon != 1:
                if factor != 1 :
                    out += "  "*indent + "(with a factor of " + monstr(factor) + ")\n"
                    subsubtree = []
                    transform = Transform(factor, 1, mon)
                    subtree.append({(transform, scaledmonval): subsubtree})
                    subtree = subsubtree
                if power != 1:
                    out += "  "*indent + "(and a power of %.2g )\n" %power
                    subsubtree = []
                    transform = Transform(1, power, mon)
                    subtree.append({(transform, scaledmonval): subsubtree})
                    subtree = subsubtree
                mon = freemon**(1/power)
                mon.ast = None
        # TODO: make minscale an argument - currently an arbitrary 0.01
        if subkey is not None and subkey not in visited_bdkeys and subkey in bd and power > 0 and scaledmonval > 0.01:  # to prevent inversion!!
            subout, subsubtree = crawl(subkey, bd, scaledmonval, indent, visited_bdkeys)
            subtree.append(subsubtree)
            out += subout
        else:
            out += "  "*indent + monstr(mon) + "\n"
            subtree.append({(mon, scaledmonval): []})
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
    indent -= 2
    # break  # for now skip other trees
    return out, tree

SYMBOLS = string.ascii_uppercase + string.ascii_lowercase
SYMBOLS = SYMBOLS.replace("l", "").replace("I", "").replace("L", "").replace("T", "")
SYMBOLS += "⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵"
SYMBOLS += "ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ"

def widthstr(legend, length, label, *, leftwards=True):
    if isinstance(label, Transform):
        label = Transform(label.factor, label.power, None)  # remove origkey so they collide
        # TODO: catch PI
        if label.power == 1 and len(str(label.factor)) == 1:
            legend[label] = str(label.factor)
    if label not in legend:
        shortname = SYMBOLS[len(legend)]
        legend[label] = shortname
    else:
        shortname = legend[label]
    if length <= 1:
        return shortname
    spacer, lend, rend = "│", "┯", "┷"
    if isinstance(label, Transform):
        spacer, lend, rend = "╎", "╤", "╧"
    if leftwards:
        if length == 2:
            return lend + shortname
        return lend + spacer*int(max(0, length - 2)/2) + shortname + spacer*int(max(0, length - 3)/2) + rend
        # return lend + spacer*int(max(0, length - 2)) + shortname
    else:
        if length == 2:
            return shortname + rend
        # return shortname + spacer*int(max(0, length - 2)) + rend
        return "┃" + "┃"*int(max(0, length - 3)/2) + shortname + "┃"*int(max(0, length - 2)/2) + "┃"

def layer(map, tree, extent, depth=0, maxdepth=20):
    # TODO: hard-enforce maxdepth so that can be an input parameter
    ((key, val), branches), = tree.items()
    if not val:
        return map
    if len(map) <= depth:
        map.append([])
    scale = extent/val
    if extent == 1 and not isinstance(key, Transform):
        branches = []
        # if isinstance(key, Transform):
        #     key = key.origkey
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
    if not branches:
        return map
    extents = [round(scale*v) for (_, v), in branches]
    # TODO: make the below optional
    if not all(extents):
        if not round(sum(scale*v for (_, v), in branches if not round(scale*v))):
            extents = [e for e in extents if e]
            branches = branches[:len(extents)]
    surplus = extent - sum(extents)
    scaled = np.array([scale*v for (_, v), in branches]) % 1
    gain_targets = sorted([(s, i) for (i, s) in enumerate(scaled) if s > 0.5])
    while surplus < 0:
        extents[gain_targets.pop(0)[1]] -= 1  # the smallest & closest to 0.5
        surplus += 1
    loss_targets = sorted([(s, i) for (i, s) in enumerate(scaled) if s < 0.5])
    while surplus > (1 - all(extents)):
        extents[loss_targets.pop()[1]] += 1  # the largest & closest to 0.5
        surplus -= 1
    if not all(extents):
        if not surplus:
            extents[gain_targets.pop(0)[1]] -= 1  # the smallest & closest to 0.5
        grouped_keys = ()
        for i, branch in enumerate(branches):
            if not extents[i]:
                (k, _), = branch
                if isinstance(k, Transform):
                    k = k.origkey
                if not isinstance(k, tuple):  # TODO: if it is, may be out of sort-order
                    k = (k,)
                grouped_keys = grouped_keys + k
        if len(grouped_keys) == 1:
            grouped_keys, = grouped_keys
        branches = branches + [{(grouped_keys, 1): []}]
        extents.append(1)
    for branch, subextent in zip(branches, extents):
        if subextent:
            # TODO: decide on this and the linked section above -- worth the hiding of leaves?
            ((k, v), bs), = branch.items()
            if isinstance(k, Transform):  # ft with no worthy heirs
                subscale = subextent/v
                if not any(round(subscale*v) for (_, v), in bs):
                    branch = {(None, v): []}
            layer(map, branch, subextent, depth+1, maxdepth)
    return map

def plumb(tree, depth=0):
    ((key, val), branches), = tree.items()
    subdepth = depth
    for branch in branches:
        subdepth = max(subdepth, plumb(branch, depth+1))
    return subdepth

def graph(tree, extent=20):
    graph_printer(tree, extent)  # breaks if there are multiple breakdowns; heatexchangersmass, peakmotorsystemforce
    # ((key, val), branches), = tree.items()
    # for i, branch in enumerate(branches):
    #     if len(branches) > 1:
    #         print("%i/%i:" % (i+1, len(branches)))
    #     graph_printer({(key, val): [branch]}, extent)
    #     print("\n")

def graph_printer(tree, extent):
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
            row += widthstr(legend, length, element, leftwards=leftwards)
        if row.strip():
            chararray[depth, :] = list(row)

    A_key, = [key for key, value in legend.items() if value == "A"]
    A_str = A_key.legendlabel if hasattr(A_key, "legendlabel") and A_key.legendlabel else A_key.str_without(["lineage", "units"])
    valuestr = "(%s)" % valstr(A_key)
    fmt = "{0:>%s}" % (max(len(A_str), len(valuestr)) + 3)
    for j, entry in enumerate(chararray[0,:]):
        if entry == "A":
            chararray[0,j] = fmt.format(A_str + "╺┫")
            chararray[0,j+1] = fmt.format(valuestr + " ┃")
        elif valuestr not in entry:
            chararray[0,j] = fmt.format(entry)
    new_legend = {}
    for pos in range(extent):
        for depth in reversed(range(1,len(mt))):
            value = chararray[depth, pos]
            if value == " ":
                chararray[depth, pos] = ""
            elif value and value in SYMBOLS:
                key, = [k for k, val in legend.items() if val == value]
                if getattr(key, "vks", None) and len(key.vks) == 1 and all(vk in new_legend for vk in key.vks):
                    key, = key.vks
                if key not in new_legend and (isinstance(key, tuple) or (depth != len(mt) - 1 and chararray[depth+1, pos] != "")):
                    new_legend[key] = SYMBOLS[len(new_legend)]
                if key in new_legend:
                    chararray[depth, pos] = new_legend[key] #  + "=%s" % key.str_without(["lineage", "units"])
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
                valuestr = " (%s)" % valstr(key)
                if key in bd or (key.vks and any(vk in bd for vk in key.vks)):
                    linkstr = "┣┉"
                else:
                    linkstr = "┣╸"
                if not isinstance(key, FixedScalar):
                    if span > 1 and (pos + 2 >= extent or chararray[depth, pos+1] == "┃"):
                        chararray[depth, pos+1] += valuestr
                    else:
                        keystr += valuestr
                chararray[depth, pos] = linkstr + keystr
    vertstr = "\n".join(["    " + "".join(row) for row in chararray.T.tolist()])
    print()
    print(vertstr)
    print()
    legend = new_legend

    legend_lines = []
    for key, shortname in sorted(legend.items(), key=lambda kv: kv[1]):
        if isinstance(key, tuple) and not isinstance(key, Transform):
            asterisk, *others = key
            legend_lines.append(legend_entry(asterisk, shortname))
            for k in others:
                legend_lines.append(legend_entry(k, ""))
        else:
            legend_lines.append(legend_entry(key, shortname))
    dirs = ["<", "<", "<","<"]
    maxlens = [max(len(el) for el in col) for col in zip(*legend_lines)]
    fmts = ["{0:%s%s}" % (direc, L) for direc, L in zip(dirs, maxlens)]
    for line in legend_lines:
        print("    " + "".join(fmt.format(cell) for fmt, cell in zip(fmts, line) if cell))

def legend_entry(key, shortname):
    operator = note = ""
    keystr = valuestr = " "
    operator = "= " if shortname else "  + "
    if isinstance(key, Transform):
        operator = " ×"
        if key.power != 1:
            valuestr = "   ^%.3g" % key.power
            key = None
        else:
            key = key.factor
    if hasattr(key, "vks") and key.vks and free_vks(m, key):
        note = "  [free]"
    if key:
        if isinstance(key, FixedScalar):
             keystr = " "
        elif hasattr(key, "legendlabel") and key.legendlabel:
            keystr = key.legendlabel
        else:
            keystr = key.str_without(["lineage", "units"])
        valuestr = "  "+operator + valstr(key)
    return ["%-4s" % shortname, keystr, valuestr, note]

from gpkit.repr_conventions import unitstr as get_unitstr

def valstr(key):
    value = m.solution(key)
    if isinstance(value, FixedScalar):
        value = value.value
    value = mag(value)
    if isinstance(key, Monomial) and key.hmap.units:
        try:
            reduced = key.hmap.units.to_reduced_units()
            value *= reduced.magnitude
            unitstr = get_unitstr(reduced)
        except DimensionalityError:
            unitstr = key.unitstr()
    else:
        unitstr = key.unitstr()
    if unitstr[:2] == "1/":
        unitstr = "/" + unitstr[2:]
    if 1e3 <= value < 1e6:
        valuestr = "{:,.0f}".format(value)
    else:
        valuestr = "%-.3g" % value
    return valuestr + unitstr

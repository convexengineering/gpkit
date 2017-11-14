"Docstring-parsing methods"
import numpy as np


def expected_unbounded(instance, doc):
    "Gets expected-unbounded variables from a string"
    exp_unbounded = set()
    for direction in ["upper", "lower"]:
        flag = direction[0].upper()+direction[1:]+" Unbounded\n"
        count = doc.count(flag)
        if count == 0:
            continue
        elif count > 1:
            raise ValueError("multiple instances of %s" % flag)
        idx = doc.index(flag) + len(flag)
        idx2 = doc[idx:].index("\n")
        idx3 = doc[idx:][idx2+1:].index("\n")
        varstrs = doc[idx:][idx2+1:][:idx3].strip()
        if varstrs:
            for var in varstrs.split(", "):
                if " (if " in var:  # it's a conditional!
                    var, condition = var.split(" (if ")
                    assert condition[-1] == ")"
                    condition = condition[:-1]
                    invert = condition[:4] == "not "
                    if invert:
                        condition = condition[4:]
                        if getattr(instance, condition):
                            continue
                    elif not getattr(instance, condition):
                        continue
                # TODO: catch err if var not found?
                variables = getattr(instance, var)
                if not hasattr(variables, "shape"):
                    variables = np.array([variables])
                it = np.nditer(variables, flags=['multi_index', 'refs_ok'])
                while not it.finished:
                    i = it.multi_index
                    it.iternext()
                    exp_unbounded.add((variables[i].key, direction))
    return exp_unbounded


def parse_variables(string):
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-nested-blocks
    "Parses a string to determine what variables to create from it"
    outstr = ""
    flag = "Variables\n"
    count = string.count(flag)
    ostring = string
    if count:
        outstr += "from gpkit import Variable\n"
        for _ in range(count):
            idx = string.index(flag)
            string = string[idx:]
            if idx == -1:
                idx = 0
                skiplines = 0
            else:
                skiplines = 2
            for line in string.split("\n")[skiplines:]:
                try:
                    unitstart, unitend = line.index("["), line.index("]")
                except ValueError:
                    break
                units = line[unitstart+1:unitend]
                labelstart = unitend + 1
                if labelstart < len(line):
                    while line[labelstart] == " ":
                        labelstart += 1
                    label = line[labelstart:].replace("'", "\\'")
                    nameval = line[:unitstart].split()
                    if len(nameval) == 2:
                        out = ("{0} = self.{0}"
                               " = Variable('{0}', {1}, '{2}', '{3}')\n")
                        outstr += out.format(nameval[0], nameval[1], units,
                                             label)
                    elif len(nameval) == 1:
                        out = ("{0} = self.{0}"
                               " = Variable('{0}', '{1}', '{2}')\n")
                        outstr += out.format(nameval[0], units, label)
            string = string[len(flag):]
        string = ostring
        flag = "Variables of length"
        count = string.count(flag)
        if count:
            outstr += "\nfrom gpkit import VectorVariable\n"
            for _ in range(count):
                idx = string.index(flag)
                string = string[idx:]
                idx2 = string.index("\n")
                length = string[len(flag):idx2].strip()
                string = string[idx2:]
                if idx == -1:
                    idx = 0
                    skiplines = 0
                else:
                    skiplines = 2
                for line in string.split("\n")[skiplines:]:
                    try:
                        unitstart, unitend = line.index("["), line.index("]")
                    except ValueError:
                        break
                    units = line[unitstart+1:unitend]
                    labelstart = unitend + 1
                    if labelstart < len(line):
                        while line[labelstart] == " ":
                            labelstart += 1
                        label = line[labelstart:].replace("'", "\\'")
                        nameval = line[:unitstart].split()
                        if len(nameval) == 2:
                            out = ("{0} = self.{0} = VectorVariable({4},"
                                   " '{0}', {1}, '{2}', '{3}')\n")
                            outstr += out.format(nameval[0], nameval[1],
                                                 units, label, length)
                        elif len(nameval) == 1:
                            out = ("{0} = self.{0} = VectorVariable({3},"
                                   " '{0}', '{1}', '{2}')\n")
                            outstr += out.format(nameval[0], units, label,
                                                 length)
                string = string[len(flag):]
    return outstr

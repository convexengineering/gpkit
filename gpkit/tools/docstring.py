"Docstring-parsing methods"
from collections import defaultdict
from ..constraints.bounded import Bounded
from ..constraints.model import Model


def verify_model(cls):
    # pylint: disable=too-many-locals
    "Creates an instance of a model and verifies its docstring"
    instance = cls()
    errmessage = "while verifying %s:\n" % cls.__name__
    err = False
    expected = defaultdict(set)
    for direction in ["upper", "lower"]:
        flag = direction[0].upper()+direction[1:]+" Unbounded\n"
        count = cls.__doc__.count(flag)
        if count == 0:
            continue
        elif count > 1:
            raise ValueError("multiple instances of %s" % flag)
        idx = cls.__doc__.index(flag) + len(flag)
        idx2 = cls.__doc__[idx:].index("\n")
        idx3 = cls.__doc__[idx:][idx2+1:].index("\n")
        varstrs = cls.__doc__[idx:][idx2+1:][:idx3].strip()
        if varstrs:
            for var in varstrs.split(", "):
                # TODO: catch err if var not found
                expected[direction].add(getattr(instance, var).key)
    easy_bounds = instance.gp(allow_missingbounds=True).missingbounds
    for (vk, direction) in easy_bounds:
        expected[direction].discard(vk)
    if expected["upper"] or expected["lower"]:
        model = Model(None, Bounded(instance, verbosity=0))
        boundedness = model.solve(verbosity=0)["boundedness"]
        actual = defaultdict(set)
        for direction in ["upper", "lower"]:
            act, exp = actual[direction], expected[direction]
            for key in ["value near %s bound" % direction,
                        "sensitive to %s bound" % direction]:
                if key in boundedness:
                    act.update(boundedness[key])
            if act-exp:
                badvks = ", ".join(map(str, act-exp))
                badvks += " were" if len(act-exp) > 1 else " was"
                errmessage += ("  %s %s-unbounded; expected"
                               " bounded\n" % (badvks, direction))
            if exp-act:
                badvks = ", ".join(map(str, exp-act))
                badvks += " were" if len(exp-act) > 1 else " was"
                errmessage += ("  %s %s-bounded; expected"
                               " unbounded\n" % (badvks, direction))
                if exp-act or act-exp:
                    err = True
    if err:
        raise ValueError(errmessage)


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
                    label = line[labelstart:]
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
                        label = line[labelstart:]
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

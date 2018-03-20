"Docstring-parsing methods"
import numpy as np
import re


def expected_unbounded(instance, doc):
    "Gets expected-unbounded variables from a string"
    # pylint: disable=too-many-locals,too-many-nested-blocks
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
        try:
            idx3 = doc[idx:][idx2+1:].index("\n\n")
        except ValueError:
            idx3 = doc[idx:][idx2+1:].index("\n")
        varstrs = doc[idx:][idx2+1:][:idx3].strip()
        varstrs = varstrs.replace("\n", ", ")  # cross newlines
        varstrs = re.sub(" +", " ", varstrs)   # multiple-whitespace removal
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
                try:
                    obj = instance
                    for subdot in var.split("."):
                        obj = getattr(obj, subdot)
                    variables = obj
                except AttributeError:
                    raise AttributeError("`%s` is noted in %s as "
                                         "unbounded, but is not "
                                         "an attribute of that model."
                                         % (var, instance.__class__.__name__))
                if not hasattr(variables, "shape"):
                    variables = np.array([variables])
                it = np.nditer(variables, flags=['multi_index', 'refs_ok'])
                while not it.finished:
                    i = it.multi_index
                    it.iternext()
                    exp_unbounded.add((variables[i].key, direction))
    return exp_unbounded


PARSETIP = ("Is this line following the format `Name (optional Value) [Units]"
            " (Optional Description)` without any whitespace in the Name or"
            " Value fields?")


def variable_declaration(nameval, units, label, line):
    if len(nameval) > 2:
        raise ValueError("while parsing the line '%s', additional fields"
                         " (separated by whitespace) were found between Value"
                         " '%s' and the Units `%s`. %s"
                         % (line, nameval[1], units, PARSETIP))
    elif len(nameval) == 2:
        out = ("{0} = self.{0} = Variable('{0}', {1}, '{2}', '{3}')")
        out = out.format(nameval[0], nameval[1], units, label)
    elif len(nameval) == 1:
        out = ("{0} = self.{0} = Variable('{0}', '{1}', '{2}')")
        out = out.format(nameval[0], units, label)
    out = """
try:
    {0}
except Exception, e:
    raise ValueError("`"+e.__class__.__name__+": "+str(e)+"` was raised while executing the parsed line `{0}`. {1}")
""".format(out, PARSETIP)
    return out

def parse_variables(string):
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-nested-blocks
    "Parses a string to determine what variables to create from it"
    outstr = "from gpkit import Variable, VectorVariable\n"
    ostring = string
    outstr += check_and_parse_flag(ostring, "Constants\n", constant_declare)
    outstr += check_and_parse_flag(ostring, "Variables\n")
    outstr += check_and_parse_flag(ostring, "Variables of length", vv_declare)
    return outstr


def vv_declare(string, flag, idx2, countstr):
    length = string[len(flag):idx2].strip()
    return countstr.replace("Variable(", "VectorVariable(%s, " % length)


def constant_declare(string, flag, idx2, countstr):
    return countstr.replace("')", "', constant=True)")


def check_and_parse_flag(string, flag, declaration_func=None):
    overallstr = ""
    count = string.count(flag)
    for _ in range(count):
        countstr = ""
        idx = string.index(flag)
        string = string[idx:]
        if "\n" not in flag:
            idx2 = string.index("\n")
        else:
            idx2 = 0
        if idx == -1:
            idx = 0
            skiplines = 0
        else:
            skiplines = 2
        for line in string[idx2:].split("\n")[skiplines:]:
            if not line:
                break
            try:
                unitstart, unitend = line.index("["), line.index("]")
            except ValueError:
                raise ValueError("A unit declaration bracketed by [] was"
                                 " not found on the line reading:\n"
                                 "    %s" % line)
            units = line[unitstart+1:unitend]
            labelstart = unitend + 1
            if labelstart < len(line):
                while line[labelstart] == " ":
                    labelstart += 1
                label = line[labelstart:].replace("'", "\\'")
            else:
                label = ""
            nameval = line[:unitstart].split()
            countstr += variable_declaration(nameval, units, label, line)
        if declaration_func is None:
            overallstr += countstr
        else:
            overallstr += declaration_func(string, flag, idx2, countstr)
        string = string[idx2+len(flag):]
    return overallstr

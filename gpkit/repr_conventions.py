# -*- coding: utf-8 -*-
"Repository for representation standards"
from __future__ import unicode_literals, print_function
import sys
import re
import numpy as np
from .small_classes import Quantity, Numbers
from .small_scripts import try_str_without


if sys.version_info >= (3, 0):
    unichr = chr  # pylint: disable=redefined-builtin,invalid-name
    PI_STR = "π"  # fails on some external models if it's "π"
    UNICODE_EXPONENTS = True
    UNIT_FORMATTING = ":P~"  # ":P~" for unicode exponents in units
else:
    PI_STR = "PI"  # fails on some external models if it's "π"
    UNICODE_EXPONENTS = False
    UNIT_FORMATTING = ":~"  # ":P~" for unicode exponents in units


def lineagestr(lineage, modelnums=True):
    "Returns properly formatted lineage string"
    if not isinstance(lineage, tuple):
        lineage = getattr(lineage, "lineage", None)
    return ".".join(["%s%i" % (name, num) if (num and modelnums) else name
                     for name, num in lineage]) if lineage else ""


def unitstr(units, into="%s", options=UNIT_FORMATTING, dimless=""):
    "Returns the string corresponding to an object's units."
    if hasattr(units, "units") and isinstance(units.units, Quantity):
        units = units.units
    if not isinstance(units, Quantity):
        return dimless
    if options == ":~" and "ohm" in str(units.units):
        rawstr = str(units.units)  # otherwise it'll be a capital Omega
    else:
        rawstr = ("{%s}" % options).format(units.units)
    units = rawstr.replace(" ", "").replace("dimensionless", dimless)
    return into % units or dimless


def strify(val, excluded):
    "Turns a value into as pretty a string as possible."
    if isinstance(val, Numbers):
        isqty = hasattr(val, "magnitude")
        if isqty:
            units = val
            val = val.magnitude
        if (val > np.pi/12 and val < 100*np.pi       # within bounds?
                and abs(12*val/np.pi % 1) <= 1e-2):  # nice multiple of PI?
            if val > 3.1:                            # product of PI
                val = "%.3g%s" % (val/np.pi, PI_STR)
                if val == "1%s" % PI_STR:
                    val = PI_STR
            else:                                   # division of PI
                val = "(%s/%.3g)" % (PI_STR, np.pi/val)
        else:
            val = "%.3g" % val
        if isqty:
            val += unitstr(units, " [%s]")
    else:
        val = try_str_without(val, excluded)
    return val


INSIDE_PARENS = re.compile(r"\(.*\)")


def parenthesize(string, addi=True, mult=True):
    "Parenthesizes a string if it needs it and isn't already."
    parensless = string if "(" not in string else INSIDE_PARENS.sub("", string)
    bare_addi = (" + " in parensless or " - " in parensless)
    bare_mult = ("*" in parensless or "/" in parensless)
    if parensless and (addi and bare_addi) or (mult and bare_mult):
        return "(%s)" % string
    return string


class GPkitObject(object):
    "This class combines various printing methods for easier adoption."
    lineagestr = lineagestr
    unitstr = unitstr
    cached_strs = None
    ast = None

    # pylint: disable=too-many-branches, too-many-statements
    def parse_ast(self, excluded=("units")):
        "Turns the AST of this object's construction into a faithful string"
        if self.cached_strs is None:
            self.cached_strs = {}
        elif frozenset(excluded) in self.cached_strs:
            return self.cached_strs[frozenset(excluded)]
        aststr = None
        oper, values = self.ast  # pylint: disable=unpacking-non-sequence
        excluded = set(excluded)
        excluded.add("units")
        if oper == "add":
            left = strify(values[0], excluded)
            right = strify(values[1], excluded)
            if right[0] == "-":
                aststr = "%s - %s" % (left, right[1:])
            else:
                aststr = "%s + %s" % (left, right)
        elif oper == "mul":
            left = parenthesize(strify(values[0], excluded), mult=False)
            right = parenthesize(strify(values[1], excluded), mult=False)
            if left == "1":
                aststr = right
            elif right == "1":
                aststr = left
            else:
                aststr = "%s*%s" % (left, right)
        elif oper == "div":
            left = parenthesize(strify(values[0], excluded), mult=False)
            right = parenthesize(strify(values[1], excluded))
            if right == "1":
                aststr = left
            else:
                aststr = "%s/%s" % (left, right)
        elif oper == "neg":
            aststr = "-%s" % parenthesize(strify(values, excluded), mult=False)
        elif oper == "pow":
            left = parenthesize(strify(values[0], excluded))
            x = values[1]
            if left == "1":
                aststr = "1"
            elif UNICODE_EXPONENTS and int(x) == x and x >= 2 and x <= 9:
                if int(x) in (2, 3):
                    aststr = "%s%s" % (left, unichr(176+int(x)))
                elif int(x) in (4, 5, 6, 7, 8, 9):
                    aststr = "%s%s" % (left, unichr(8304+int(x)))
            else:
                aststr = "%s^%s" % (left, x)
        elif oper == "prod":  # TODO: only do if it makes a shorter string
            aststr = "%s.prod()" % parenthesize(strify(values[0], excluded))
        elif oper == "sum":  # TODO: only do if it makes a shorter string
            aststr = "%s.sum()" % parenthesize(strify(values[0], excluded))
        elif oper == "index":  # TODO: label vectorization idxs
            left = parenthesize(strify(values[0], excluded))
            idx = values[1]
            if left[-3:] == "[:]":  # pure variable access
                left = left[:-3]
            if isinstance(idx, tuple):
                elstrs = []
                for el in idx:
                    if isinstance(el, slice):
                        start = el.start or ""
                        stop = (el.stop if el.stop and el.stop != sys.maxint
                                else "")
                        step = ":%s" % el.step if el.step is not None else ""
                        elstrs.append("%s:%s%s" % (start, stop, step))
                    elif isinstance(el, Numbers):
                        elstrs.append("%s" % el)
                idx = ",".join(elstrs)
            elif isinstance(idx, slice):
                start = idx.start or ""
                stop = idx.stop if idx.stop and idx.stop < 1e6 else ""
                step = ":%s" % idx.step if idx.step is not None else ""
                idx = "%s:%s%s" % (start, stop, step)
            elif isinstance(idx, Numbers):
                idx = "%s" % idx
            else:
                raise ValueError(repr(idx))
            aststr = "%s[%s]" % (left, idx)
        else:
            raise ValueError(oper)
        self.cached_strs[frozenset(excluded)] = aststr
        return aststr

    def __repr__(self):
        "Returns namespaced string."
        return "gpkit.%s(%s)" % (self.__class__.__name__, self)

    def __str__(self):
        "Returns default string."
        return self.str_without()  # pylint: disable=no-member

    def _repr_latex_(self):
        "Returns default latex for automatic iPython Notebook rendering."
        return "$$"+self.latex()+"$$"  # pylint: disable=no-member

    def latex_unitstr(self):
        "Returns latex unitstr"
        us = self.unitstr(r"~\mathrm{%s}", ":L~")
        utf = us.replace("frac", "tfrac").replace(r"\cdot", r"\cdot ")
        return utf if utf != r"~\mathrm{-}" else ""

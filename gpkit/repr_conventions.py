# -*- coding: utf-8 -*-
"Repository for representation standards"
from __future__ import print_function
import sys
import re
import numpy as np
from .small_classes import Quantity, Numbers
from .small_scripts import try_str_without


try:
    print("​", end="")  # zero-width space
    DEFAULT_UNIT_PRINTING = [":P~"]
    pi_str = "PI"  # fails on some external models if it's "π"
except UnicodeEncodeError:
    DEFAULT_UNIT_PRINTING = [":~"]
    pi_str = "PI"


def lineagestr(lineage, modelnums=True):
    "Returns properly formatted lineage string"
    lineage = getattr(lineage, "lineage", None) or lineage
    return ".".join(["%s%i" % (name, num) if (num and modelnums) else name
                     for name, num in lineage]) if lineage else ""


def unitstr(units, into="%s", options=None, dimless=""):
    "Returns the string corresponding to an object's units."
    options = options or DEFAULT_UNIT_PRINTING[0]
    if hasattr(units, "units") and isinstance(units.units, Quantity):
        units = units.units
    if not isinstance(units, Quantity):
        return dimless
    rawstr = (u"{%s}" % options).format(units.units)
    units = rawstr.replace(" ", "").replace("dimensionless", dimless)
    return into % units or dimless


def strify(val, excluded):
    "Turns a value into as pretty a string as possible."
    if isinstance(val, Numbers):
        if (val > np.pi/12 and val < 100*np.pi       # within bounds?
                and abs(12*val/np.pi % 1) <= 1e-2):  # nice multiple of PI?
            if val > 3.1:                            # product of PI
                val = "%.3g%s" % (val/np.pi, pi_str)
                if val == "1%s" % pi_str:
                    val = pi_str
            else:                                   # division of PI
                val = "(%s/%.3g)" % (pi_str, np.pi/val)
        else:
            val = "%.3g" % val
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
            if left == "1":
                aststr = "1"
            else:
                aststr = "%s^%s" % (left, values[1])
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
                stop = idx.stop if idx.stop != sys.maxint else ""
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

# -*- coding: utf-8 -*-
"Repository for representation standards"
from __future__ import print_function
import sys
import numpy as np
from .small_classes import Quantity, Numbers
from .small_scripts import try_str_without


try:
    print("​", end="")  # zero-width space
    DEFAULT_UNIT_PRINTING = [":P~"]
except UnicodeEncodeError:
    DEFAULT_UNIT_PRINTING = [":~"]
unicode_pi = True  # fails on some external models if True


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
    if isinstance(val, Numbers):
        if val > np.pi/12 and val < 100*np.pi and abs(12*val/np.pi % 1) <= 1e-2:
            pi_str = "π" if unicode_pi else "PI"
            if val > 3.1:
                val = "%.3g%s" % (val/np.pi, pi_str)
                if val == "1%s" % pi_str:
                    val = pi_str
            else:
                val = "(%s/%.3g)" % (pi_str, np.pi/val)
        else:
            val = "%.3g" % val
    else:
        val = try_str_without(val, excluded)
    return val

def parenthesize(string):
    if string[0] == "(" and string[-1] == ")":
        return string
    else:
        return "(%s)" % string

class GPkitObject(object):
    "This class combines various printing methods for easier adoption."
    lineagestr = lineagestr
    unitstr = unitstr
    cached_strs = None

    def parse_ast(self, excluded=("units")):
        if self.cached_strs is None:
            self.cached_strs = {}
        elif frozenset(excluded) in self.cached_strs:
            return self.cached_strs[frozenset(excluded)]
        aststr = None
        oper, values = self.ast
        excluded = set(excluded)
        excluded.add("units")
        left, right = values
        if oper == "add":
            left = strify(left, excluded)
            right = strify(right, excluded)
            if right[0] == "-":
                aststr = "%s - %s" % (left, right[1:])
            else:
                aststr = "%s + %s" % (left, right)
        elif oper == "mul":
            maybe_left = strify(left, excluded)
            maybe_right = strify(right, excluded)
            if maybe_left == "1":
                aststr = maybe_right
            elif maybe_right == "1":
                aststr = maybe_left
            else:
                if len(getattr(left, "hmap", [])) > 1:
                    left = parenthesize(strify(left, excluded))
                if len(getattr(right, "hmap", [])) > 1:
                    right = parenthesize(strify(right, excluded))
                left = strify(left, excluded)
                right = strify(right, excluded)
                aststr = "%s*%s" % (left, right)
        elif oper == "div":
            left = strify(left, excluded)
            right = strify(right, excluded)
            if right == "1":
                aststr = left
            else:
                if "*" in right or "/" in right:
                    right = parenthesize(right)
                if " + " in left or " - " in left:
                    left = parenthesize(left)
                aststr = "%s/%s" % (left, right)
        elif oper == "neg":
            aststr = "-%s" % strify(left, excluded)
        elif oper == "pow":
            maybe_left = strify(left, excluded)
            if "*" in maybe_left or " + " in maybe_left or "/" in maybe_left or " - " in maybe_left:
                maybe_left = parenthesize(maybe_left)
            aststr = "%s^%s" % (maybe_left, right)
            if maybe_left == "1":
                aststr = "1"
        elif oper == "prod":  # TODO: really you only want to do these if it makes the overall thing shorter
            aststr = "%s.prod()" % strify(left, excluded)
        elif oper == "sum":
            left = strify(left, excluded)
            if "*" in left or " - " in left or "/" in left or " - " in left:
                left = parenthesize(left)
            aststr = "%s.sum()" % left
        elif oper == "index":  # TODO: label vectorization idxs
            left = strify(left, excluded)
            if "*" in left or " - " in left or "/" in left or " - " in left:
                left = parenthesize(left)
            else:
                left = left.replace("[:]", "")
            if isinstance(right, tuple):
                elstrs = []
                for el in right:
                    if isinstance(el, slice):
                        start = el.start or ""
                        stop = el.stop if el.stop and el.stop != sys.maxint else ""
                        step = ":%s" % el.step if el.step is not None else ""
                        elstrs.append("%s:%s%s" % (start, stop, step))
                    elif isinstance(el, Numbers):
                        elstrs.append("%s" % el)
                right = ",".join(elstrs)
            elif isinstance(right, slice):
                start = right.start or ""
                stop = right.stop if right.stop != sys.maxint else ""
                step = ":%s" % right.step if right.step is not None else ""
                right = "%s:%s%s" % (start, stop, step)
            elif isinstance(right, Numbers):
                right = "%s" % right
            else:
                raise ValueError(repr(right))
            aststr = "%s[%s]" % (left, right)
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

"Implements ConstraintSet"
from warnings import warn
from collections import defaultdict, OrderedDict
from itertools import chain
import numpy as np
from ..keydict import KeySet, KeyDict
from ..small_scripts import try_str_without
from ..repr_conventions import ReprMixin
from .single_equation import SingleEquationConstraint


def add_meq_bounds(bounded, meq_bounded):  #TODO: collapse with GP version?
    "Iterates through meq_bounds until convergence"
    still_alive = True
    while still_alive:
        still_alive = False  # if no changes are made, the loop exits
        for bound, conditions in list(meq_bounded.items()):
            if bound in bounded:  # bound exists in an inequality
                del meq_bounded[bound]
                continue
            for condition in conditions:
                if condition.issubset(bounded):  # bound's condition is met
                    del meq_bounded[bound]
                    bounded.add(bound)
                    still_alive = True
                    break

def _sort_by_name_and_idx(var):
    "return tuple for Variable sorting"
    return (var.key.str_without(["units", "idx"]), var.key.idx or ())

def _sort_constrs(item):
    "return tuple for Constraint sorting"
    label, constraint = item
    return (not isinstance(constraint, SingleEquationConstraint),
            hasattr(constraint, "lineage") and bool(constraint.lineage), label)

def flatiter(iterable, yield_if_hasattr=None):
    "Yields contained constraints, optionally including constraintsets."
    if isinstance(iterable, dict):
        iterable = iterable.values()
    for constraint in iterable:
        if yield_if_hasattr and hasattr(constraint, yield_if_hasattr):
            yield constraint
        elif hasattr(constraint, "flat"):
            if hasattr(constraint.flat, "__call__"):
                yield from constraint.flat(yield_if_hasattr)
            else:
                yield from constraint.flat
        elif hasattr(constraint, "__iter__"):
            yield from flatiter(constraint, yield_if_hasattr)
        else:
            yield constraint


class ConstraintSet(list, ReprMixin):
    "Recursive container for ConstraintSets and Inequalities"
    unique_varkeys, idxlookup = frozenset(), {}
    _name_collision_varkeys = None

    def __init__(self, constraints, substitutions=None):  # pylint: disable=too-many-branches,too-many-statements
        if isinstance(constraints, dict):
            if isinstance(constraints, OrderedDict):
                items = constraints.items()
            else:
                items = sorted(list(constraints.items()), key=_sort_constrs)
            self.idxlookup = {k: i for i, (k, _) in enumerate(items)}
            constraints = (item[1] for item in items)  # just the values
        elif isinstance(constraints, ConstraintSet):
            constraints = [constraints]  # put it one level down
        else:
            warn("please label your constraints!", PendingDeprecationWarning)
        list.__init__(self, constraints)
        # get substitutions and convert all members to ConstraintSets
        self.varkeys = KeySet(self.unique_varkeys)
        self.substitutions = KeyDict({k: k.value for k in self.unique_varkeys
                                      if "value" in k.descr})
        self.substitutions.varkeys = self.varkeys
        self.bounded, self.meq_bounded = set(), defaultdict(set)
        for i, constraint in enumerate(self):
            if hasattr(constraint, "varkeys"):
                self._update(constraint)
            elif not hasattr(constraint, "as_hmapslt1"):
                if not hasattr(constraint, "__iter__"):
                    raise_badelement(self, i, constraint)
                elif isinstance(constraint, ConstraintSet):
                    raise ValueError("found ConstraintSet of type %s in"
                                     " another ConstraintSet before it had"
                                     " been initialized." % type(constraint))
                for subconstraint in flatiter(constraint,
                                              yield_if_hasattr="varkeys"):
                    if isinstance(subconstraint, np.bool_):
                        raise_badelement(self, i, constraint)
                    self._update(subconstraint)
        if substitutions:
            self.substitutions.update(substitutions)
        updated_veckeys = False  # vector subs need to find each indexed varkey
        for subkey in self.substitutions:
            if not updated_veckeys and subkey.shape and not subkey.idx:
                for key in self.varkeys:
                    if key.veckey:
                        self.varkeys.keymap[key.veckey].add(key)
                updated_veckeys = True
            for key in self.varkeys[subkey]:
                self.bounded.add((key, "upper"))
                self.bounded.add((key, "lower"))
                if key.value is not None and not key.constant:
                    del key.descr["value"]
                    if key.veckey and key.veckey.value is not None:
                        del key.veckey.descr["value"]
        add_meq_bounds(self.bounded, self.meq_bounded)

    def _update(self, constraint):
        "Update parameters with a given constraint"
        self.varkeys.update(constraint.varkeys)
        if hasattr(constraint, "substitutions"):
            self.substitutions.update(constraint.substitutions)
        else:
            self.substitutions.update({k: k.value \
                for k in constraint.varkeys if "value" in k.descr})
        self.bounded.update(constraint.bounded)
        for bound, solutionset in constraint.meq_bounded.items():
            self.meq_bounded[bound].update(solutionset)

    def __getitem__(self, key):
        if key in self.idxlookup:
            key = self.idxlookup[key]
        if isinstance(key, int):
            return list.__getitem__(self, key)
        return self._choosevar(key, self.variables_byname(key))

    def _choosevar(self, key, variables):
        if not variables:
            raise KeyError(key)
        firstvar, *othervars = variables
        veckey = firstvar.key.veckey
        if veckey is None or any(v.key.veckey != veckey for v in othervars):
            if not othervars:
                return firstvar
            raise ValueError("multiple variables are called '%s'; show them"
                             " with `.variables_byname('%s')`" % (key, key))
        from ..nomials import NomialArray  # all one vector!
        arr = NomialArray(np.full(veckey.shape, np.nan, dtype="object"))
        for v in variables:
            arr[v.key.idx] = v
        arr.key = veckey
        return arr

    def variables_byname(self, key):
        "Get all variables with a given name"
        from ..nomials import Variable
        return sorted([Variable(k) for k in self.varkeys[key]],
                      key=_sort_by_name_and_idx)

    def constrained_varkeys(self):
        "Return all varkeys in non-ConstraintSet constraints"
        constrained_varkeys = set()
        for constraint in self.flat():
            constrained_varkeys.update(constraint.varkeys)
        return constrained_varkeys

    flat = flatiter

    def flathmaps(self, subs):
        "Yields hmaps<=1 from self.flat()"
        yield from chain(*(l.as_hmapslt1(subs) for l in self.flat()))

    def process_result(self, result):
        """Does arbitrary computation / manipulation of a program's result

        There's no guarantee what order different constraints will process
        results in, so any changes made to the program's result should be
        careful not to step on other constraint's toes.

        Potential Uses
        --------------
          - check that an inequality was tight
          - add values computed from solved variables

        """
        for constraint in self.flat(yield_if_hasattr="process_result"):
            if hasattr(constraint, "process_result"):
                constraint.process_result(result)
        for v in self.unique_varkeys:
            if not v.evalfn or v in result["variables"]:
                continue
            if v.veckey:
                v = v.veckey
            val = v.evalfn(result["variables"])
            result["variables"][v] = result["freevariables"][v] = val

    def __repr__(self):
        "Returns namespaced string."
        if not self:
            return "<gpkit.%s object>" % self.__class__.__name__
        return ("<gpkit.%s object containing %i top-level constraint(s)"
                " and %i variable(s)>" % (self.__class__.__name__,
                                          len(self), len(self.varkeys)))

    def name_collision_varkeys(self):
        "Returns the set of contained varkeys whose names are not unique"
        if self._name_collision_varkeys is None:
            self._name_collision_varkeys = set()
            for key in self.varkeys:
                if len(self.varkeys[key.str_without(["lineage", "vec"])]) > 1:
                    self._name_collision_varkeys.add(key)
        return self._name_collision_varkeys

    def lines_without(self, excluded):
        "Lines representation of a ConstraintSet."
        root = "root" not in excluded
        rootlines, lines = [], []
        indent = " " if getattr(self, "lineage", None) else ""
        if root:
            excluded += ("root",)
            if "unnecessary lineage" in excluded:
                for key in self.name_collision_varkeys():
                    key.descr["necessarylineage"] = True
            if hasattr(self, "_rootlines"):
                rootlines = self._rootlines(excluded)  # pylint: disable=no-member
        lines = recursively_line(self, excluded)
        if root:
            if "unnecessary lineage" in excluded:
                for key in self.name_collision_varkeys():
                    del key.descr["necessarylineage"]
        return rootlines + [(indent+line).rstrip() for line in lines]

    def str_without(self, excluded=("unnecessary lineage", "units")):
        "String representation of a ConstraintSet."
        return "\n".join(self.lines_without(excluded))

    def latex(self, excluded=("units",)):
        "LaTeX representation of a ConstraintSet."
        lines = []
        root = "root" not in excluded
        if root:
            excluded += ("root",)
            lines.append("\\begin{array}{ll} \\text{}")
            if hasattr(self, "_rootlatex"):
                lines.append(self._rootlatex(excluded))  # pylint: disable=no-member
        for constraint in self:
            cstr = try_str_without(constraint, excluded, latex=True)
            if cstr[:6] != "    & ":  # require indentation
                cstr = "    & " + cstr + " \\\\"
            lines.append(cstr)
        if root:
            lines.append("\\end{array}")
        return "\n".join(lines)

    def as_view(self):
        "Return a ConstraintSetView of this ConstraintSet."
        return ConstraintSetView(self)

def recursively_line(iterable, excluded):
    "Generates lines in a recursive tree-like fashion, the better to indent."
    named_constraints = {}
    if isinstance(iterable, dict):
        items = sorted(list(iterable.items()), key=_sort_constrs)
        named_constraints = {i: k for i, (k, _) in enumerate(items)}
        iterable = (item[1] for item in items)  # just the values
    elif hasattr(iterable, "idxlookup"):
        named_constraints = {i: k for k, i in iterable.idxlookup.items()}
    lines = []
    for i, constraint in enumerate(iterable):
        if hasattr(constraint, "lines_without"):
            clines = constraint.lines_without(excluded)
        elif not hasattr(constraint, "__iter__"):
            clines = try_str_without(constraint, excluded).split("\n")
        elif iterable is constraint:
            clines = ["(constraint contained itself)"]
        else:
            clines = recursively_line(constraint, excluded)
        if (getattr(constraint, "lineage", None)
                and isinstance(constraint, ConstraintSet)):
            name, num = constraint.lineage[-1]
            if not any(clines):
                clines = [" " + "(no constraints)"]  # named model indent
            if lines:
                lines.append("")
            lines.append(name if not num else name + str(num))
        elif "constraint names" not in excluded and i in named_constraints:
            lines.append("\"%s\":" % named_constraints[i])
            clines = ["  " + line for line in clines]  # named constraint indent
        lines.extend(clines)
    return lines


class ConstraintSetView:
    "Class to access particular views on a set's variables"

    def __init__(self, constraintset, index=()):
        self.constraintset = constraintset
        try:
            self.index = tuple(index)
        except TypeError:  # probably not iterable
            self.index = (index,)

    def __getitem__(self, index):
        "Appends the index to its own and returns a new view."
        if not isinstance(index, tuple):
            index = (index,)
        # indexes are preprended to match Vectorize convention
        return ConstraintSetView(self.constraintset, index + self.index)

    def __getattr__(self, attr):
        """Returns attribute from the base ConstraintSets

        If it's a another ConstraintSet, return the matching View;
        if it's an array, return it at the specified index;
        otherwise, raise an error.
        """
        if not hasattr(self.constraintset, attr):
            raise AttributeError("the underlying object lacks `.%s`." % attr)

        value = getattr(self.constraintset, attr)
        if isinstance(value, ConstraintSet):
            return ConstraintSetView(value, self.index)
        if not hasattr(value, "shape"):
            raise ValueError("attribute %s with value %s did not have"
                             " a shape, so ConstraintSetView cannot"
                             " return an indexed view." % (attr, value))
        index = self.index
        newdims = len(value.shape) - len(self.index)
        if newdims > 0:  # indexes are put last to match Vectorize
            index = (slice(None),)*newdims + index
        return value[index]



def raise_badelement(cns, i, constraint):
    "Identify the bad element and raise a ValueError"
    cause = "" if not isinstance(constraint, bool) else (
        " Did the constraint list contain an accidental equality?")
    if len(cns) == 1:
        loc = "as the only constraint"
    elif i == 0:
        loc = "at the start, before %s" % cns[i+1]
    elif i == len(cns) - 1:
        loc = "at the end, after %s" % cns[i-1]
    else:
        loc = "between %s and %s" % (cns[i-1], cns[i+1])
    raise ValueError("%s was found %s.%s"
                     % (type(constraint), loc, cause))

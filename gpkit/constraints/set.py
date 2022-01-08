"Implements ConstraintSet"
import sys
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
        for bound in list(meq_bounded):
            if bound in bounded:  # bound already exists
                del meq_bounded[bound]
                continue
            for condition in meq_bounded[bound]:
                if condition.issubset(bounded):  # bound's condition is met
                    del meq_bounded[bound]
                    bounded.add(bound)
                    still_alive = True
                    break

def _sort_by_name_and_idx(var):
    "return tuple for Variable sorting"
    return (var.key.str_without(["units", "idx"]), var.key.idx or ())

def _sort_constraints(item):
    "return tuple for Constraint sorting"
    label, constraint = item
    return (not isinstance(constraint, SingleEquationConstraint),
            bool(getattr(constraint, "lineage", None)), label)

def sort_constraints_dict(iterable):
    "Sort a dictionary of {k: constraint} and return its keys and values"
    if sys.version_info >= (3, 7) or isinstance(iterable, OrderedDict):
        return iterable.keys(), iterable.values()
    items = sorted(list(iterable.items()), key=_sort_constraints)
    return (item[0] for item in items), (item[1] for item in items)

def flatiter(iterable, yield_if_hasattr=None):
    "Yields contained constraints, optionally including constraintsets."
    if isinstance(iterable, dict):
        _, iterable = sort_constraints_dict(iterable)
    for constraint in iterable:
        if (not hasattr(constraint, "__iter__")
                or (yield_if_hasattr
                    and hasattr(constraint, yield_if_hasattr))):
            yield constraint
        else:
            try:  # numpy array
                yield from constraint.flat
            except TypeError:  # ConstrainSet
                yield from constraint.flat(yield_if_hasattr)
            except AttributeError:  # probably a list or dict
                yield from flatiter(constraint, yield_if_hasattr)


class ConstraintSet(list, ReprMixin):  # pylint: disable=too-many-instance-attributes
    "Recursive container for ConstraintSets and Inequalities"
    unique_varkeys, idxlookup = frozenset(), {}
    _name_collision_varkeys = None
    _varkeys = None
    _lineageset = False

    def __init__(self, constraints, substitutions=None, *, bonusvks=None):  # pylint: disable=too-many-branches,too-many-statements
        if isinstance(constraints, dict):
            keys, constraints = sort_constraints_dict(constraints)
            self.idxlookup = {k: i for i, k in enumerate(keys)}
        elif isinstance(constraints, ConstraintSet):
            constraints = [constraints]  # put it one level down
        list.__init__(self, constraints)
        self.vks = set(self.unique_varkeys)
        self.substitutions = KeyDict({k: k.value for k in self.unique_varkeys
                                      if "value" in k.descr})
        self.substitutions.vks = self.vks
        self.bounded, self.meq_bounded = set(), defaultdict(set)
        for i, constraint in enumerate(self):
            if hasattr(constraint, "vks"):
                self._update(constraint)
            elif not (hasattr(constraint, "as_hmapslt1")
                      or hasattr(constraint, "as_gpconstr")):
                try:
                    for subconstraint in flatiter(constraint, "vks"):
                        self._update(subconstraint)
                except Exception as e:
                    raise badelement(self, i, constraint) from e
            elif isinstance(constraint, ConstraintSet):
                raise badelement(self, i, constraint,
                                 " It had not yet been initialized!")
        if bonusvks:
            self.vks.update(bonusvks)
        if substitutions:
            self.substitutions.update(substitutions)
        for key in self.vks:
            if key not in self.substitutions:
                if key.veckey is None or key.veckey not in self.substitutions:
                    continue
                if np.isnan(self.substitutions[key.veckey][key.idx]):
                    continue
            self.bounded.add((key, "upper"))
            self.bounded.add((key, "lower"))
            if key.value is not None and not key.constant:
                del key.descr["value"]
                if key.veckey and key.veckey.value is not None:
                    del key.veckey.descr["value"]
        add_meq_bounds(self.bounded, self.meq_bounded)

    def _update(self, constraint):
        "Update parameters with a given constraint"
        self.vks.update(constraint.vks)
        if hasattr(constraint, "substitutions"):
            self.substitutions.update(constraint.substitutions)
        else:
            self.substitutions.update({k: k.value \
                for k in constraint.vks if "value" in k.descr})
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

    @property
    def varkeys(self):
        "The NomialData's varkeys, created when necessary for a substitution."
        if self._varkeys is None:
            self._varkeys = KeySet(self.vks)
        return self._varkeys

    def constrained_varkeys(self):
        "Return all varkeys in non-ConstraintSet constraints"
        return self.vks - self.unique_varkeys

    flat = flatiter

    def as_hmapslt1(self, subs):
        "Yields hmaps<=1 from self.flat()"
        yield from chain(*(c.as_hmapslt1(subs)
                           for c in flatiter(self,
                                             yield_if_hasattr="as_hmapslt1")))

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
        evalfn_vars = {v.veckey or v for v in self.unique_varkeys
                       if v.evalfn and v not in result["variables"]}
        for v in evalfn_vars:
            val = v.evalfn(result["variables"])
            result["variables"][v] = result["freevariables"][v] = val

    def __repr__(self):
        "Returns namespaced string."
        if not self:
            return "<gpkit.%s object>" % self.__class__.__name__
        return ("<gpkit.%s object containing %i top-level constraint(s)"
                " and %i variable(s)>" % (self.__class__.__name__,
                                          len(self), len(self.varkeys)))

    def set_necessarylineage(self, clear=False):  # pylint: disable=too-many-branches
        "Returns the set of contained varkeys whose names are not unique"
        if self._name_collision_varkeys is None:
            self._name_collision_varkeys = {}
            name_collisions = defaultdict(set)
            for key in self.varkeys:
                if hasattr(key, "key"):
                    if key.veckey and all(k.veckey == key.veckey
                                          for k in self.varkeys[key.name]):
                        self._name_collision_varkeys[key] = 0
                        self._name_collision_varkeys[key.veckey] = 0
                    elif len(self.varkeys[key.name]) == 1:
                        self._name_collision_varkeys[key] = 0
                    else:
                        shortname = key.str_without(["lineage", "vec"])
                        if len(self.varkeys[shortname]) > 1:
                            name_collisions[shortname].add(key)
            for varkeys in name_collisions.values():
                min_namespaced = defaultdict(set)
                for vk in varkeys:
                    *_, mineage = vk.lineagestr().split(".")
                    min_namespaced[(mineage, 1)].add(vk)
                while any(len(vks) > 1 for vks in min_namespaced.values()):
                    for key, vks in list(min_namespaced.items()):
                        if len(vks) <= 1:
                            continue
                        del min_namespaced[key]
                        mineage, idx = key
                        idx += 1
                        for vk in vks:
                            lineages = vk.lineagestr().split(".")
                            submineage = lineages[-idx] + "." + mineage
                            min_namespaced[(submineage, idx)].add(vk)
                for (_, idx), vks in min_namespaced.items():
                    vk, = vks
                    self._name_collision_varkeys[vk] = idx
        if clear:
            self._lineageset = False
            for vk in self._name_collision_varkeys:
                del vk.descr["necessarylineage"]
        else:
            self._lineageset = True
            for vk, idx in self._name_collision_varkeys.items():
                vk.descr["necessarylineage"] = idx

    def lines_without(self, excluded):
        "Lines representation of a ConstraintSet."
        excluded = frozenset(excluded)
        root, rootlines = "root" not in excluded, []
        if root:
            excluded = {"root"}.union(excluded)
            self.set_necessarylineage()
            if hasattr(self, "_rootlines"):
                rootlines = self._rootlines(excluded)  # pylint: disable=no-member
        lines = recursively_line(self, excluded)
        indent = " " if root or getattr(self, "lineage", None) else ""
        if root:
            self.set_necessarylineage(clear=True)
        return rootlines + [(indent+line).rstrip() for line in lines]

    def str_without(self, excluded=("units",)):
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
        keys, iterable = sort_constraints_dict(iterable)
        named_constraints = dict(enumerate(keys))
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



def badelement(cns, i, constraint, cause=""):
    "Identify the bad element and raise a ValueError"
    cause = cause if not isinstance(constraint, bool) else (
        " Did the constraint list contain an accidental equality?")
    if len(cns) == 1:
        loc = "the only constraint"
    elif i == 0:
        loc = "at the start, before %s" % cns[i+1]
    elif i == len(cns) - 1:
        loc = "at the end, after %s" % cns[i-1]
    else:
        loc = "between %s and %s" % (cns[i-1], cns[i+1])
    return ValueError("Invalid ConstraintSet element '%s' %s was %s.%s"
                      % (repr(constraint), type(constraint), loc, cause))

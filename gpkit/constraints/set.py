"Implements ConstraintSet"
from __future__ import unicode_literals
from collections import defaultdict, OrderedDict
import numpy as np
from ..small_classes import Numbers
from ..keydict import KeySet, KeyDict
from ..small_scripts import try_str_without
from ..repr_conventions import GPkitObject
from .single_equation import SingleEquationConstraint


def add_meq_bounds(bounded, meq_bounded):
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


# pylint: disable=too-many-instance-attributes
class ConstraintSet(list, GPkitObject):
    "Recursive container for ConstraintSets and Inequalities"
    varkeys = None
    unique_varkeys = frozenset()
    # idxlookup holds the names of the top-level constraintsets
    idxlookup = None
    _name_collision_varkeys = None

    def __init__(self, constraints, substitutions=None):  # pylint: disable=too-many-branches
        if isinstance(constraints, ConstraintSet):
            constraints = [constraints]  # put it one level down
        elif isinstance(constraints, dict):
            if isinstance(constraints, OrderedDict):
                items = constraints.items()
            else:
                items = sorted(list(constraints.items()), key=_sort_constrs)
            self.idxlookup = {k: i for i, (k, _) in enumerate(items)}
            constraints = list(zip(*items))[1]
        list.__init__(self, constraints)

        # initializations for attributes used elsewhere
        self.posymap = []
        self.relax_sensitivity = 0
        self.numpy_bools = False

        # get substitutions and convert all members to ConstraintSets
        self.substitutions = KeyDict()
        self.bounded = set()
        self.meq_bounded = defaultdict(set)
        for i, constraint in enumerate(self):
            if not isinstance(constraint, ConstraintSet):
                if hasattr(constraint, "__iter__"):
                    list.__setitem__(self, i, ConstraintSet(constraint))
                elif not hasattr(constraint, "varkeys"):
                    if not isinstance(constraint, np.bool_):
                        raise_badelement(self, i, constraint)
                    else:
                        # allow NomialArray equalities (arr == "a", etc.)
                        self.numpy_bools = True  # but mark them
                        # so we can catch them later (next line)
            elif not hasattr(constraint, "numpy_bools"):
                raise ValueError("a ConstraintSet of type %s was included in"
                                 " another ConstraintSet before being"
                                 " initialized." % type(constraint))
            elif constraint.numpy_bools:
                raise_elementhasnumpybools(constraint)
            for attr in ["substitutions", "bounded"]:
                if hasattr(self[i], attr):
                    getattr(self, attr).update(getattr(self[i], attr))
            if hasattr(self[i], "meq_bounded"):
                for bound, solutionset in self[i].meq_bounded.items():
                    self.meq_bounded[bound].update(solutionset)
        self.reset_varkeys()
        self.substitutions.update({k: k.descr["value"]
                                   for k in self.unique_varkeys
                                   if "value" in k.descr})
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
                if key.value is not None and not key.constant:
                    del key.descr["value"]
                    if key.veckey and key.veckey.value is not None:
                        del key.veckey.descr["value"]
                for direction in ("upper", "lower"):
                    self.bounded.add((key, direction))
        add_meq_bounds(self.bounded, self.meq_bounded)

    def __getitem__(self, key):
        if self.idxlookup and key in self.idxlookup:
            key = self.idxlookup[key]
        if isinstance(key, int):
            return list.__getitem__(self, key)
        return self._choosevar(key, self.variables_byname(key))

    def _choosevar(self, key, variables):
        if not variables:
            raise KeyError(key)
        if variables[0].key.veckey:  # maybe it's all one vector variable?
            from ..nomials import NomialArray
            vk = variables[0].key.veckey
            arr = NomialArray(np.full(vk.shape, np.nan, dtype="object"))
            arr.key = vk
            for variable in variables:
                if variable.key.veckey == vk:
                    arr[variable.key.idx] = variable
                else:
                    arr = None
                    break
            if arr is not None:
                return arr
        elif len(variables) == 1:
            return variables[0]
        raise ValueError("multiple variables are called '%s'; use"
                         " variables_byname('%s') to see all of them"
                         % (key, key))

    def variables_byname(self, key):
        "Get all variables with a given name"
        from ..nomials import Variable
        variables = [Variable(key) for key in self.varkeys[key]]
        variables.sort(key=_sort_by_name_and_idx)
        return variables

    def constrained_varkeys(self):
        "Return all varkeys in non-ConstraintSet constraints"
        constrained_varkeys = set()
        for constraint in self.flat():
            constrained_varkeys.update(constraint.varkeys)
        return constrained_varkeys

    def __setitem__(self, key, value):
        if self.idxlookup and key in self.idxlookup:
            key = self.idxlookup[key]
        self.substitutions.update(value.substitutions)
        list.__setitem__(self, key, value)
        self.reset_varkeys()

    def flat(self, constraintsets=False):
        "Yields contained constraints, optionally including constraintsets."
        for constraint in self:
            if not isinstance(constraint, ConstraintSet):
                yield constraint
            else:
                if constraintsets:
                    yield constraint
                subgenerator = constraint.flat(constraintsets)
                for yielded_constraint in subgenerator:
                    yield yielded_constraint

    def reset_varkeys(self):
        "Goes through constraints and collects their varkeys."
        self.varkeys = KeySet(self.unique_varkeys)
        for constraint in self:
            if hasattr(constraint, "varkeys"):
                self.varkeys.update(constraint.varkeys)
        if hasattr(self.substitutions, "varkeys"):
            self.substitutions.varkeys = self.varkeys
        self._name_collision_varkeys = None

    def as_posyslt1(self, substitutions=None):
        "Returns list of posynomials which must be kept <= 1"
        posylist, self.posymap = [], []
        for i, constraint in enumerate(self):
            if not hasattr(constraint, "as_posyslt1"):
                raise_badelement(self, i, constraint)
            posys = constraint.as_posyslt1(substitutions)
            self.posymap.append(len(posys))
            posylist.extend(posys)
        return posylist

    def sens_from_dual(self, las, nus, result):
        """Computes constraint and variable sensitivities from dual solution

        Arguments
        ---------
        las : list
            Sensitivity of each posynomial returned by `self.as_posyslt1`

        nus: list of lists
             Each posynomial's monomial sensitivities


        Returns
        -------
        constraint_sens : dict
            The interesting and computable sensitivities of this constraint

        var_senss : dict
            The variable sensitivities of this constraint
        """
        var_senss = {}
        offset = 0
        self.relax_sensitivity = 0
        for i, constr in enumerate(self):
            n_posys = self.posymap[i]
            la = las[offset:offset+n_posys]
            nu = nus[offset:offset+n_posys]
            constr.v_ss = constr.sens_from_dual(la, nu, result)
            self.relax_sensitivity += constr.relax_sensitivity
            for key, value in constr.v_ss.items():
                var_senss[key] = value + var_senss.get(key, 0)
            offset += n_posys
        return var_senss

    def as_gpconstr(self, x0):
        """Returns GPConstraint approximating this constraint at x0

        When x0 is none, may return a default guess."""
        return ConstraintSet([constr.as_gpconstr(x0) for constr in self])

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
        for constraint in self:
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
        indent = " "*2 if (len(self) > 1
                           or getattr(self, "lineage", None)) else ""
        if root:
            excluded += ("root",)
            if "unnecessary lineage" in excluded:
                for key in self.name_collision_varkeys():
                    key.descr["necessarylineage"] = True
            if hasattr(self, "rootconstr_str"):
                rootlines = self.rootconstr_str(excluded)  # pylint: disable=no-member
        if self.idxlookup:
            named_constraints = {v: k for k, v in self.idxlookup.items()}
        for i, constraint in enumerate(self):
            clines = try_str_without(constraint, excluded).split("\n")
            if (getattr(constraint, "lineage", None)
                    and isinstance(constraint, ConstraintSet)):
                name, num = constraint.lineage[-1]
                if not any(clines):
                    clines = [indent + "(no constraints)"]
                if lines:
                    lines.append("")
                lines.append(name if not num else name + str(num))
            elif ("constraint names" not in excluded
                  and self.idxlookup and i in named_constraints):
                lines.append("\"%s\":" % named_constraints[i])
                for j, line in enumerate(clines):
                    if clines[j][:len(indent)] != indent:
                        clines[j] = indent + line  # must be indented
            lines.extend(clines)
        if root:
            indent = " "
            if "unnecessary lineage" in excluded:
                for key in self.name_collision_varkeys():
                    del key.descr["necessarylineage"]
        return rootlines + [indent+line for line in lines]

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
            if hasattr(self, "rootconstr_latex"):
                lines.append(self.rootconstr_latex(excluded))  # pylint: disable=no-member
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


class ConstraintSetView(object):
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
        if hasattr(self.constraintset, attr):
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
        else:
            raise AttributeError("the underlying ConstraintSet does not have"
                                 "attribute %s." % attr)


def raise_badelement(cns, i, constraint):
    "Identify the bad element and raise a ValueError"
    cause = "" if not isinstance(constraint, bool) else (
        " Did the constraint list contain"
        " an accidental equality?")
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


def raise_elementhasnumpybools(constraint):
    "Identify the bad subconstraint array and raise a ValueError"
    cause = ("An ArrayConstraint was created with elements of"
             " numpy.bool_")
    for side in [constraint.left, constraint.right]:
        if not (isinstance(side, Numbers)
                or hasattr(side, "hmap")
                or hasattr(side, "__iter__")):
            cause += (", because "
                      "NomialArray comparison with %.10s %s"
                      " does not return a valid constraint."
                      % (repr(side), type(side)))
    raise ValueError("%s\nFull constraint: %s"
                     % (cause, constraint))

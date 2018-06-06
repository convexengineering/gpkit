"Implements ConstraintSet"
from collections import defaultdict
import numpy as np

from ..small_classes import HashVector, Numbers
from ..keydict import KeySet, KeyDict
from ..small_scripts import try_str_without
from ..repr_conventions import _str, _repr, _repr_latex_


def add_meq_bounds(bounded, meq_bounded):
    "Iterates through meq_bounds until convergence"
    still_alive = True
    while still_alive:
        still_alive = False  # if no changes are made, the loop exits
        for bound, conditions in meq_bounded.items():
            if bound in bounded:
                del meq_bounded[bound]
                continue
            meq_bounded[bound] = set(conditions)
            for condition in conditions:
                if condition.issubset(bounded):
                    del meq_bounded[bound]
                    bounded.add(bound)
                    still_alive = True
                    break


def _sort_by_name_and_idx(var):
    "return tuplef for Variable sorting"
    return (var.key.str_without(["units", "idx"]), var.key.idx)


# pylint: disable=too-many-instance-attributes
class ConstraintSet(list):
    "Recursive container for ConstraintSets and Inequalities"
    varkeys = None
    unique_varkeys = frozenset()

    def __init__(self, constraints, substitutions=None):  # pylint: disable=too-many-branches
        if isinstance(constraints, ConstraintSet):
            # stick it in a list to maintain hierarchy
            constraints = [constraints]
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
            elif constraint.numpy_bools:  # not initialized??
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
        for key in self.varkeys:
            if key in self.substitutions:
                if key.value is not None and not key.constant:
                    del key.descr["value"]
                    if key.veckey and key.veckey.value is not None:
                        del key.veckey.descr["value"]
                for direction in ("upper", "lower"):
                    self.bounded.add((key, direction))
        add_meq_bounds(self.bounded, self.meq_bounded)

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        return self._choosevar(key, self.variables_byname(key))

    def _choosevar(self, key, variables):
        if not variables:
            raise KeyError(key)
        if variables[0].key.veckey:
            # maybe it's all one vector variable!
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
        for constraint in self.flat(constraintsets=False):
            constrained_varkeys.update(constraint.varkeys)
        return constrained_varkeys

    def __setitem__(self, key, value):
        self.substitutions.update(value.substitutions)
        list.__setitem__(self, key, value)
        self.reset_varkeys()

    def append(self, value):
        if hasattr(value, "__iter__") and not isinstance(value, ConstraintSet):
            value = ConstraintSet(value)
        self.substitutions.update(value.substitutions)
        list.append(self, value)
        self.reset_varkeys()

    __str__ = _str
    __repr__ = _repr
    _repr_latex_ = _repr_latex_

    def str_without(self, excluded=None):
        "String representation of a ConstraintSet."
        if not excluded:
            excluded = ["units"]
        lines = []
        if "root" not in excluded:
            excluded.append("root")
            lines.append("")
            root_str = self.rootconstr_str(excluded)
            if root_str:
                lines.append(root_str)
        for constraint in self:
            cstr = constraint.subconstr_str(excluded)
            if cstr is None:
                cstr = try_str_without(constraint, excluded)
            if cstr[:8] != "        ":  # require indentation
                cstr = "        " + cstr
            lines.append(cstr)
        return "\n".join(lines)

    def latex(self, excluded=None):
        "LaTeX representation of a ConstraintSet."
        if not excluded:
            excluded = ["units"]
        lines = []
        root = "root" not in excluded
        if root:
            excluded.append("root")
            lines.append("\\begin{array}{ll} \\text{}")
            root_latex = self.rootconstr_latex(excluded)
            if root_latex:
                lines.append(root_latex)
        for constraint in self:
            cstr = constraint.subconstr_latex(excluded)
            if cstr is None:
                cstr = constraint.latex(excluded)
            if cstr[:6] != "    & ":  # require indentation
                cstr = "    & " + cstr + " \\\\"
            lines.append(cstr)
        if root:
            lines.append("\\end{array}")
        return "\n".join(lines)

    def rootconstr_str(self, excluded=None):
        "The appearance of a ConstraintSet in addition to its contents"
        pass

    def rootconstr_latex(self, excluded=None):
        "The appearance of a ConstraintSet in addition to its contents"
        pass

    def subconstr_str(self, excluded=None):
        "The collapsed appearance of a ConstraintSet"
        pass

    def subconstr_latex(self, excluded=None):
        "The collapsed appearance of a ConstraintSet"
        pass

    def flat(self, constraintsets=True):
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

    def subinplace(self, subs):
        """Substitutes in place, updating self.substitutions accordingly.

        Keys substituted with `subinplace` are no longer present, so if such a
        key is also in self.substitutions that substitution is now orphaned. If
        `subs[key]` describes some key in the ConstraintSet (i.e. one key has
        been substituted for another), then a substitution is added, mapping
        the orphaned value to this new key; otherwise, an error is raised.
        """
        subs = {getattr(k, "key", k): getattr(v, "key", v)
                for k, v in subs.items()}
        subkeys = frozenset(subs)
        for constraint in self:
            if isinstance(constraint.varkeys, set):
                constraint.varkeys = KeySet(constraint.varkeys)
            csubs = {k: v for k, v in subs.items() if k in constraint.varkeys}
            if csubs:
                constraint.subinplace(csubs)
        if subkeys.intersection(self.substitutions):
            for key, value in subs.items():
                if key in self.substitutions:
                    valkey, _ = self.substitutions.parse_and_index(value)
                    self.substitutions[valkey] = self.substitutions[key]
                    del self.substitutions[key]
        self.unique_varkeys = frozenset(subs[vk] if vk in subs else vk
                                        for vk in self.unique_varkeys)
        self.reset_varkeys()

    def reset_varkeys(self):
        "Goes through constraints and collects their varkeys."
        varkeys = set(self.unique_varkeys)
        for constraint in self:
            if hasattr(constraint, "varkeys"):
                varkeys.update(constraint.varkeys)
        self.varkeys = KeySet(varkeys)
        if hasattr(self.substitutions, "varkeys"):
            self.substitutions.varkeys = self.varkeys

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
        var_senss = HashVector()
        offset = 0
        self.relax_sensitivity = 0
        for i, constr in enumerate(self):
            n_posys = self.posymap[i]
            la = las[offset:offset+n_posys]
            nu = nus[offset:offset+n_posys]
            v_ss = constr.sens_from_dual(la, nu, result)
            constr.v_ss = v_ss
            self.relax_sensitivity += constr.relax_sensitivity
            # not using HashVector addition because we want to preseve zeros
            for key, value in v_ss.items():
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
            result["freevariables"][v] = val
            result["variables"][v] = val


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

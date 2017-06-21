"Implements ConstraintSet"
import numpy as np

from ..small_classes import HashVector, Numbers
from ..keydict import KeySet, KeyDict
from ..small_scripts import try_str_without
from ..repr_conventions import _str, _repr, _repr_latex_


def _sort_by_name_and_idx(var):
    "return tuplef for Variable sorting"
    return (var.key.str_without(["units", "idx"]), var.key.idx)


class ConstraintSet(list):
    "Recursive container for ConstraintSets and Inequalities"
    varkeys = None
    unique_varkeys = frozenset()

    def __init__(self, constraints, substitutions=None):
        if isinstance(constraints, ConstraintSet):
            # stick it in a list to maintain hierarchy
            constraints = [constraints]
        list.__init__(self, constraints)

        # initializations for attributes used elsewhere
        self.posymap = []
        self.numpy_bools = False

        # get substitutions and convert all members to ConstraintSets
        self.substitutions = KeyDict()
        for i, constraint in enumerate(self):
            if getattr(constraint, "numpy_bools", None):
                raise_elementhasnumpybools(constraint)
            elif not isinstance(constraint, ConstraintSet):
                if hasattr(constraint, "__iter__"):
                    list.__setitem__(self, i, ConstraintSet(constraint))
                elif not hasattr(constraint, "varkeys"):
                    if not isinstance(constraint, np.bool_):
                        raise_badelement(self, i, constraint)
                    else:
                        # allow NomialArray equalities (arr == "a", etc.)
                        self.numpy_bools = True  # but mark them
                        # so we can catch them (see above) in ConstraintSets
            if hasattr(self[i], "substitutions"):
                self.substitutions.update(self[i].substitutions)
        self.reset_varkeys()
        if substitutions:
            self.substitutions.update(substitutions)

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
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

    def topvar(self, key):
        "If a variable by a given name exists in the top model, return it"
        # TODO: could this be done with unique_varkeys?
        if not self.naming:  # pylint: disable=no-member
            raise TypeError("constraintsets must be named to have top-level"
                            " named variables.")
        topvars = [var for var in self.variables_byname(key)
                   if var.key.naming == self.naming]  # pylint: disable=no-member
        return self._choosevar(key, topvars)

    def variables_byname(self, key):
        "Get all variables with a given name"
        from ..nomials import Variable
        variables = [Variable(newvariable=False, **key.descr)
                     for key in self.varkeys[key]]
        variables.sort(key=_sort_by_name_and_idx)
        return variables

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
        "Substitutes in place."
        subs = {k.key: getattr(v, "key", v) for k, v in subs.items()}
        for constraint in self:
            constraint.subinplace(subs)
        for key in subs:
            if key not in self.substitutions:
                continue
            if hasattr(subs[key], "key"):
                self.substitutions[subs[key]] = self.substitutions[key]
                del self.substitutions[key]
            else:
                raise ValueError("the substitution {%s: %s} is invalidated"
                                 " by the subinplace {%s: %s}, because"
                                 " %s does not have a `key` attribute"
                                 % (key, self.substitutions[key],
                                    key, subs[key], subs[key]))
        self.unique_varkeys = frozenset(subs[vk] if vk in subs else vk
                                        for vk in self.unique_varkeys)
        self.reset_varkeys()

    def reset_varkeys(self):
        "Goes through constraints and collects their varkeys."
        self.varkeys = KeySet(self.unique_varkeys)
        for constraint in self:
            if hasattr(constraint, "varkeys"):
                self.varkeys.update(constraint.varkeys)
        self.substitutions.varkeys = self.varkeys

    def as_posyslt1(self, substitutions=None):
        "Returns list of posynomials which must be kept <= 1"
        posylist, self.posymap = [], []
        for constraint in self:
            posys = constraint.as_posyslt1(substitutions)
            self.posymap.append(len(posys))
            posylist.extend(posys)
        return posylist

    def sens_from_dual(self, las, nus):
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
        for i, constr in enumerate(self):
            n_posys = self.posymap[i]
            la = las[offset:offset+n_posys]
            nu = nus[offset:offset+n_posys]
            v_ss = constr.sens_from_dual(la, nu)
            var_senss += v_ss
            offset += n_posys
        return var_senss

    def as_gpconstr(self, x0, substitutions=None):
        """Returns GPConstraint approximating this constraint at x0

        When x0 is none, may return a default guess."""
        gpconstrs = [constr.as_gpconstr(x0, substitutions) for constr in self]
        return ConstraintSet(gpconstrs, substitutions)

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
                or hasattr(side, "exps")
                or hasattr(side, "__iter__")):
            cause += (", because "
                      "NomialArray comparison with %.10s %s"
                      " does not return a valid constraint."
                      % (repr(side), type(side)))
    raise ValueError("%s\nFull constraint: %s"
                     % (cause, constraint))

"Implements ConstraintSet"
import numpy as np

from ..small_classes import HashVector
from ..keydict import KeySet, KeyDict
from ..small_scripts import try_str_without
from ..repr_conventions import _str, _repr, _repr_latex_


def _sort_by_name_and_idx(var):
    "return tuplef for Variable sorting"
    return (var.key.str_without(["units", "idx"]), var.key.idx)


class ConstraintSet(list):
    "Recursive container for ConstraintSets and Inequalities"
    def __init__(self, constraints, substitutions=None, recursesubs=True):
        if isinstance(constraints, ConstraintSet):
            constraints = [constraints]
        list.__init__(self, constraints)
        subs = dict(substitutions) if substitutions else {}
        self.unused_variables = None
        if not isinstance(constraints, ConstraintSet):
            # constraintsetify everything
            for i, constraint in enumerate(self):
                if (hasattr(constraint, "__iter__") and
                        not isinstance(constraint, ConstraintSet)):
                    self[i] = ConstraintSet(constraint)
                elif isinstance(constraint, bool):
                    # TODO refactor this depending on the outcome of issue #824
                    if len(self) == 1:
                        adjacent = "as the only constraint"
                    elif i == 0:
                        adjacent = "at the start, before %s" % self[i+1]
                    elif i == len(self) - 1:
                        adjacent = "at the end, after %s" % self[i-1]
                    else:
                        adjacent = "between %s and %s" % (self[i-1], self[i+1])
                    raise ValueError("boolean found %s."
                                     " Did the constraint list contain an"
                                     " accidental equality?" % adjacent)
        else:
            # grab the substitutions dict from the top constraintset
            subs.update(constraints.substitutions)  # pylint: disable=no-member
        if recursesubs:
            self.reset_varkeys()
            self.substitutions = KeyDict.with_keys(self.varkeys,
                                                   self._iter_subs(subs))
        else:
            self.substitutions = subs
        # initializations for attributes used elsewhere
        self.posymap = []

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
            from ..nomials import Variable
            variables = [Variable(**k.descr) for k in self.varkeys[key]]
            if len(variables) == 1:
                return variables[0]
            elif variables[0].key.idx and variables[0].key.shape:
                # maybe it's all one vector variable!
                vector = np.full(variables[0].key.shape, np.nan, dtype="object")
                goaldict = dict(variables[0].key.descr)
                del goaldict["idx"]
                for variable in variables:
                    testdict = dict(variable.key.descr)
                    del testdict["idx"]
                    if testdict == goaldict:
                        vector[variable.key.idx] = variable
                    else:
                        vector = None
                        break
                if vector is not None:
                    return vector
            raise ValueError("multiple variables are called '%s'; use"
                             " variable_byname('%s') to see all of them"
                             % (key, key))

    def variables_byname(self, key):
        "Get all variables with a given name"
        from ..nomials import Variable
        variables = [Variable(**key.descr) for key in self.varkeys[key]]
        variables.sort(key=_sort_by_name_and_idx)
        return variables

    def __setitem__(self, key, value):
        list.__setitem__(self, key, value)
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
        for constraint in self:
            constraint.subinplace(subs)
        if self.unused_variables is not None:
            unused_vars = []
            for var in self.unused_variables:
                if var.key in subs:
                    unused_vars.append(subs[var.key])
                else:
                    unused_vars.append(var.key)
            self.unused_variables = unused_vars
        self.reset_varkeys()

    def reset_varkeys(self, init_dict=None):
        "Goes through constraints and collects their varkeys."
        varkeys = KeySet()
        if init_dict is not None:
            varkeys.update(init_dict)
        for constraint in self:
            if hasattr(constraint, "varkeys"):
                varkeys.update(constraint.varkeys)
        if self.unused_variables is not None:
            varkeys.update(self.unused_variables)
        self.varkeys = varkeys

    def as_posyslt1(self):
        "Returns list of posynomials which must be kept <= 1"
        posylist, self.posymap = [], []
        for constraint in self:
            constraint.substitutions = self.substitutions
            posys = constraint.as_posyslt1()
            self.posymap.append(len(posys))
            posylist.extend(posys)
        return posylist

    def sens_from_dual(self, las, nus):
        """Computes constraint and variable sensitivities from dual solution

        Arguments
        ---------
        las : list
            Sensitivity of each posynomial returned by `self.as_posyslt1()`

        nus: list of lists
             Each posynomial's monomial sensitivities


        Returns
        -------
        constraint_sens : dict
            The interesting and computable sensitivities of this constraint

        var_senss : dict
            The variable sensitivities of this constraint
        """
        # constr_sens = {}
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

    def as_gpconstr(self, x0):
        """Returns GPConstraint approximating this constraint at x0

        When x0 is none, may return a default guess."""
        gpconstrs = [constr.as_gpconstr(x0) for constr in self]
        return ConstraintSet(gpconstrs,
                             self.substitutions, recursesubs=False)

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

    def _iter_subs(self, substitutions):
        for constraint in self.flat():
            if hasattr(constraint, "substitutions"):
                subs = constraint.substitutions
                yield subs
        yield substitutions

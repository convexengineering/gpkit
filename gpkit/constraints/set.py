"Implements ConstraintSet"
from ..small_classes import HashVector
from ..keydict import KeySet, KeyDict
from ..small_scripts import try_str_without
from ..repr_conventions import _str, _repr, _repr_latex_


class ConstraintSet(list):
    "Recursive container for ConstraintSets and Inequalities"
    def __init__(self, constraints, substitutions=None, recursesubs=True):
        if isinstance(constraints, ConstraintSet):
            constraints = [constraints]
        list.__init__(self, constraints)
        subs = substitutions if substitutions else {}
        if not isinstance(constraints, ConstraintSet):
            # constraintsetify everything
            for i, constraint in enumerate(self):
                if (hasattr(constraint, "__iter__") and
                        not isinstance(constraint, ConstraintSet)):
                    self[i] = ConstraintSet(constraint)
        else:
            # grab the substitutions dict from the top constraintset
            subs.update(constraints.substitutions)  # pylint: disable=no-member
        if recursesubs:
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
            variables = [Variable(**key.descr) for key in self.varkeys[key]]
            if len(variables) == 1:
                return variables[0]
            else:
                return variables

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

    def subinplace(self, subs, value=None):
        "Substitutes in place."
        for constraint in self:
            constraint.subinplace(subs, value)

    @property
    def varkeys(self):
        "return all Varkeys present in this ConstraintSet"
        return self._varkeys()

    def _varkeys(self, init_dict=None):
        "return all Varkeys present in this ConstraintSet"
        init_dict = {} if init_dict is None else init_dict
        out = KeySet(init_dict)
        for constraint in self:
            if hasattr(constraint, "varkeys"):
                out.update(constraint.varkeys)
        return out

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
        constr_sens = {}
        var_senss = HashVector()
        offset = 0
        for i, constr in enumerate(self):
            n_posys = self.posymap[i]
            la = las[offset:offset+n_posys]
            nu = nus[offset:offset+n_posys]
            constr_sens[str(constr)], v_ss = constr.sens_from_dual(la, nu)
            var_senss += v_ss
            offset += n_posys
        return constr_sens, var_senss

    def as_gpconstr(self, x0):
        """Returns GPConstraint approximating this constraint at x0

        When x0 is none, may return a default guess."""
        gpconstrs = [constr.as_gpconstr(x0) for constr in self]
        return ConstraintSet(gpconstrs,
                             self.substitutions, recursesubs=False)

    def sens_from_gpconstr(self, gpapprox, gp_sens, var_senss):
        """Computes sensitivities from GPConstraint approximation

        Arguments
        ---------
        gpapprox : GPConstraint
            The GPConstraint returned by `self.as_gpconstr()`

        gpconstr_sens :
            Sensitivities created by `gpconstr.sens_from_dual`

        var_senss : dict
            Variable sensitivities from last GP solve.


        Returns
        -------
        constraint_sens : dict
            The interesting and computable sensitivities of this constraint
        """
        constr_sens = {}
        for i, c in enumerate(self):
            gpa = gpapprox[i]
            gp_s = gp_sens[str(gpa)]
            constr_sens[str(c)] = c.sens_from_gpconstr(gpa, gp_s, var_senss)
        return constr_sens

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

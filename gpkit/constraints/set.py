import numpy as np
from ..nomials.array import NomialArray
from ..small_classes import Numbers, HashVector, KeySet, KeyDict
from ..small_scripts import try_str_without


def constraintset_iterables(obj):
    if hasattr(obj, "__iter__"):
        return ConstraintSet(obj)
    else:
        return obj


class ConstraintSet(list):
    substitutions = None

    def __init__(self, constraints, substitutions=None):
        list.__init__(self, constraints)
        self.recurse(constraintset_iterables)
        self.substitutions = KeyDict.subs_from_constr(self, substitutions)

    def str_without(self, excluded=[]):
        return "[" + ", ".join([try_str_without(el, excluded)
                                for el in self]) + "]"

    def __str__(self):
        "Returns list-like string, but with str(el) instead of repr(el)."
        return self.str_without()

    def __repr__(self):
        return "gpkit.%s(%s)" % (self.__class__.__name__, self)

    def latex(self, matwrap=True):
        return ("\\begin{bmatrix}" +
                " \\\\\n".join(el.latex(matwrap=False) for el in self) +
                "\\end{bmatrix}")

    def _repr_latex_(self):
        return "$$"+self.latex()+"$$"

    @property
    def flat(self):
        for constraint in self:
            if isinstance(constraint, ConstraintSet):
                try:
                    yield constraint.flat.next()
                except StopIteration:
                    pass
            else:
                yield constraint

    def recurse(self, function, *args, **kwargs):
        "Apply a function to each terminal constraint"
        for i, constraint in enumerate(self):
            if isinstance(constraint, ConstraintSet):
                constraint.recurse(function, *args, **kwargs)
            else:
                self[i] = function(constraint, *args, **kwargs)

    @property
    def varkeys(self):
        "Varkeys present in the constraints"
        return KeySet.from_constraintset(self)

    def as_posyslt1(self):
        "Returns list of posynomials which must be kept <= 1"
        posylist, self.posymap = [], []
        for constraint in self:
            constraint.substitutions.update(self.substitutions)
            posys = constraint.as_posyslt1()
            self.posymap.append(len(posys))
            posylist.extend(posys)
        return posylist

    def sens_from_dual(self, p_senss, m_sensss):
        """Computes constraint and variable sensitivities from dual solution

        Arguments
        ---------
        p_senss : list
            Sensitivity of each posynomial returned by `self.as_posyslt1()`

        m_sensss: list of lists
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
            p_ss = p_senss[offset:offset+n_posys]
            m_sss = m_sensss[offset:offset+n_posys]
            constr_sens[str(constr)], v_ss = constr.sens_from_dual(p_ss, m_sss)
            var_senss += v_ss
            offset += n_posys
        return constr_sens, var_senss

    def as_gpconstr(self, x0):
        """Returns GPConstraint approximating this constraint at x0

        When x0 is none, may return a default guess."""
        cs = ConstraintSet([constr.as_gpconstr(x0) for constr in self])
        cs.substitutions.update(self.substitutions)
        return cs

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
        processed = {}
        for constraint in self:
            if hasattr(constraint, "process_result"):
                p = constraint.process_result(result)
                if p:
                    processed.update(p)
        return processed

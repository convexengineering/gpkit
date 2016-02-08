import numpy as np
from ..nomials.array import NomialArray
from ..small_classes import Numbers, HashVector, KeySet, KeyDict


class ConstraintSet(NomialArray):
    substitutions = None
    def __new__(cls, constraints, substitutions=None):
        "Constructor. Required for objects inheriting from np.ndarray."
        # Input array is an already formed ndarray instance
        # cast to be our class type
        obj = np.asarray(constraints).view(cls)
        obj.substitutions = KeyDict.from_constraintset_subs(obj, substitutions)
        return obj

    def __array_finalize__(self, obj):
        "Finalizer. Required for objects inheriting from np.ndarray."
        if obj is None:
            return
        self.substitutions = getattr(obj, 'substitutions', {})

    @property
    def varkeys(self):
        "Varkeys present in the constraints"
        return KeySet.from_constraintset(self)

    def as_posyslt1(self):
        "Returns list of posynomials which must be kept <= 1"
        posylist, posymap = [], []
        for constraint in self:
            constraint.substitutions.update(self.substitutions)
            posys, _ = constraint.as_posyslt1()
            posymap.append(len(posys))
            posylist.extend(posys)
        return posylist, posymap

    def sens_from_dual(self, posymap, p_senss, m_sensss):
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
            n_posys = posymap[i]
            p_ss = p_senss[offset:offset+n_posys]
            m_sss = m_sensss[offset:offset+n_posys]
            constr_sens[str(constr)], v_ss = constr.sens_from_dual(posymap[i],
                                                                   p_ss, m_sss)
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
        for constraint in self.iter_flat():
            if hasattr(constraint, "process_result"):
                p = constraint.process_result(result)
                if p:
                    processed.update(p)
        return processed

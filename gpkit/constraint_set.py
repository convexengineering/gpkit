import numpy as np
from .nomialarray import NomialArray
from .small_classes import Numbers, HashVector, KeySet, KeyDict
from .constraint_single_equation import SingleEquationConstraint


class ConstraintSet(NomialArray):
    substitutions = None

    def __new__(cls, input_array, substitutions=None):
        "Constructor. Required for objects inheriting from np.ndarray."
        # Input array is an already formed ndarray instance
        # cast to be our class type
        obj = np.asarray(input_array).view(cls)
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

    def parse_constraints(self):
        self.onlyposyconstrs, self.localposyconstrs = [], []
        self.all_have_posy_rep = True
        self.allposyconstrs = []
        for constr in self.iter():
            constr.substitutions.update(self.substitutions)
            localposy = False
            if hasattr(constr, "as_gpconstr"):
                localposy = constr.as_gpconstr(None)
                if localposy:
                    self.localposyconstrs.append(constr)
            if hasattr(constr, "as_posyslt1"):
                self.allposyconstrs.append(constr)
                if not localposy:
                    self.onlyposyconstrs.append(constr)
            elif localposy:
                self.all_have_posy_rep = False
            else:
                raise ValueError("constraints must have either an"
                                 "`as_gpconstr` method or an"
                                 "`as_posyslt1` method, but %s has neither"
                                 % constr)

    def as_posyslt1(self):
        "Returns list of posynomials which must be kept <= 1"
        posylist, self.posymap = [], []
        for constraint in self.iter():
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
        for i, constr in enumerate(self.iter()):
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
        as_gpconstr = lambda c: c.as_gpconstr(x0)
        cs = ConstraintSet(self.recurse(as_gpconstr))
        cs.substitutions.update(self.substitutions)
        return cs

    def sens_from_gpconstr(self, posyapprox, posy_approx_sens, var_senss):
        """Computes sensitivities from GPConstraint approximation

        Arguments
        ---------
        gpconstr : GPConstraint
            Sensitivity of the GPConstraint returned by `self.as_gpconstr()`

        gpconstr_sens :
            Sensitivities created by `gpconstr.sens_from_dual`

        var_senss : dict
            Variable sensitivities from last GP solve.


        Returns
        -------
        constraint_sens : dict
            The interesting and computable sensitivities of this constraint
        """
        pass
        # constr_sens = {}
        # for i, lpc in enumerate(self):
        #     pa = posyapprox[i]
        #     p_a_s = posy_approx_sens[str(pa)]
        #     constr_sens[str(lpc)] = lpc.sens_from_gpconstr(pa, p_a_s, var_senss)
        # return constr_sens

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
        for constraint in self.iter():
            if hasattr(constraint, "process_result"):
                p = constraint.process_result(result)
                if p:
                    processed.update(p)
        return processed


class ArrayConstraint(SingleEquationConstraint, ConstraintSet):
    left = None
    oper = None
    right = None
    substitutions = None

    def __new__(cls, input_array, left, oper, right):
        "Constructor. Required for objects inheriting from np.ndarray."
        obj = ConstraintSet.__new__(cls, input_array, {})
        obj.left = left
        obj.oper = oper
        obj.right = right
        return obj

    def __array_finalize__(self, obj):
        "Finalizer. Required for objects inheriting from np.ndarray."
        if obj is None:
            return
        ConstraintSet.__array_finalize__(self, obj)

    def str_without(self, excluded=["units"]):
        if self.oper:
            return SingleEquationConstraint.str_without(self, excluded)
        else:
            return NomialArray.str_without(self, excluded)

    def latex(self, *args, **kwargs):
        if self.oper:
            return SingleEquationConstraint.latex(self)
        else:
            return NomialArray.latex(self, *args, **kwargs)

import numpy as np
from .small_classes import Numbers, HashVector, KeySet, KeyDict
from .constraint_meta import LocallyApproximableConstraint, GPConstraint


class ConstraintSet(LocallyApproximableConstraint, GPConstraint):
    substitutions = None

    def __init__(self, constraints, substitutions=None,
                 latex=None, string=None):
        self.constraints = constraints
        cs = self.flatconstraints()
        vks = self.make_varkeys(cs)
        self.substitutions = KeyDict.from_constraints(vks, cs, substitutions)

    @property
    def varkeys(self):
        return self.make_varkeys()

    def make_varkeys(self, constraints=None):
        varkeys = KeySet()
        constraints = constraints if constraints else self.flatconstraints()
        for constr in constraints:
            varkeys.update(constr.varkeys)
        return varkeys

    def flatconstraints(self):
        constraints = self.constraints
        if hasattr(constraints, "flatten"):
            constraints = constraints.flatten()
            isnt_numpy_bool = lambda c: c and type(c) is not np.bool_
            constraints = filter(isnt_numpy_bool, constraints)
        return constraints

    def parse_constraints(self):
        self.onlyposyconstrs, self.localposyconstrs = [], []
        self.all_have_posy_rep = True
        self.allposyconstrs = []
        for constr in self.flatconstraints():
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

    def __len__(self):
        return len(self.constraints)

    def __getattr__(self, attr):
        return getattr(self.constraints, attr)

    def __getitem__(self, idx):
        return self.constraints[idx]

    def __setitem__(self, idx, value):
        self.constraints[idx] = value

    def as_posyslt1(self):
        self.parse_constraints()
        if self.all_have_posy_rep:
            posyss, self.posymap = [], []
            for c in self.allposyconstrs:
                posys = c.as_posyslt1()
                self.posymap.append(len(posys))
                posyss.extend(posys)
            return posyss
        else:
            return [None]

    def as_gpconstr(self, x0):
        self.parse_constraints()
        if not self.localposyconstrs:
            return None
        self.posymap = "sp"
        localposyconstrs = [c.as_gpconstr(x0)
                            for c in self.localposyconstrs]
        localposyconstrs.extend(self.onlyposyconstrs)
        return ConstraintSet(localposyconstrs, self.substitutions)

    def __repr__(self):
        return "gpkit.%s(%s)" % (self.__class__.__name__, str(self))

    def __str__(self):
        return str(self.constraints)

    def latex(self):
        return self.constraints.latex()

    def sub(self, subs, value=None):
        return self  # TODO

    def sens_from_dual(self, p_senss, m_sensss):
        assert self.all_have_posy_rep
        constr_sens = {}
        var_senss = HashVector()
        offset = 0
        for i, n_posys in enumerate(self.posymap):
            constr = self.allposyconstrs[i]
            p_ss = p_senss[offset:offset+n_posys]
            m_sss = m_sensss[offset:offset+n_posys]
            constr_sens[str(constr)], v_ss = constr.sens_from_dual(p_ss, m_sss)
            var_senss += v_ss
            offset += n_posys

        return constr_sens, var_senss

    def sens_from_gpconstr(self, posyapprox, posy_approx_sens, var_senss):
        constr_sens = {}
        for i, lpc in enumerate(self.localposyconstrs):
            pa = posyapprox[i]
            p_a_s = posy_approx_sens[str(pa)]
            constr_sens[str(lpc)] = lpc.sens_from_gpconstr(pa, p_a_s, var_senss)
        return constr_sens

    def process_result(self, result):
        processed = {}
        for constraint in self.constraints:
            if hasattr(constraint, "process_result"):
                p = constraint.process_result(result)
                if p:
                    processed.update(p)
        return processed


class ArrayConstraint(ConstraintSet):
    def __str__(self):
        return "%s %s %s" % (self.left, self.oper, self.right)

    def __repr__(self):
        return "gpkit.%s(%s)" % (self.__class__.__name__, self)

    def latex(self):
        latex_oper = self.latex_opers[self.oper]
        units = bool(self.units)
        return ("%s %s %s" % (self.left.latex(showunits=units), latex_oper,
                              self.right.latex(showunits=units)))

# -*coding: utf-8 -*-
"""Module for creating PosyArray instances.

    Example
    -------
    >>> x = gpkit.Monomial('x')
    >>> px = gpkit.PosyArray([1, x, x**2])

"""

import numpy as np
from .small_classes import Numbers, KeyVector, KeySet, KeyDict

from . import units as ureg
from . import DimensionalityError
Quantity = ureg.Quantity


class PosyArray(np.ndarray):
    """A Numpy array with elementwise inequalities and substitutions.

    Arguments
    ---------
    input_array : array-like

    Example
    -------
    >>> px = gpkit.PosyArray([1, x, x**2])
    """

    def __str__(self):
        "Returns list-like string, but with str(el) instead of repr(el)."
        if self.shape:
            return "[" + ", ".join(str(p) for p in self) + "]"
        else:
            return str(self.flatten()[0])

    def __repr__(self):
        "Returns str(self) tagged with gpkit information."
        if self.shape:
            return "gpkit.%s(%s)" % (self.__class__.__name__, str(self))
        else:
            return str(self.flatten()[0])

    def __hash__(self):
        return hash(self.tostring())

    def __new__(cls, input_array):
        "Constructor. Required for objects inheriting from np.ndarray."
        # Input array is an already formed ndarray instance
        # cast to be our class type
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        "Finalizer. Required for objects inheriting from np.ndarray."
        pass

    def __array_wrap__(self, out_arr, context=None):
        """Called by numpy ufuncs.
        Special case to avoid creation of 0-dimensional arrays
        See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html"""
        if out_arr.ndim:
            return np.ndarray.__array_wrap__(self, out_arr, context)
        try:
            val = out_arr.item()
            return np.float(val) if isinstance(val, np.generic) else val
        except:
            print("Something went wrong. I'd like to raise a RuntimeWarning,"
                  " but you wouldn't see it because numpy seems to catch all"
                  " Exceptions coming from __array_wrap__.")
            raise

    def latex(self, unused=None, matwrap=True):
        "Returns 1D latex list of contents."
        if len(self.shape) == 0:
            return self.flatten()[0].latex()
        if len(self.shape) == 1:
            return (("\\begin{bmatrix}" if matwrap else "") +
                    " & ".join(el.latex() for el in self) +
                    ("\\end{bmatrix}" if matwrap else ""))
        elif len(self.shape) == 2:
            return ("\\begin{bmatrix}" +
                    " \\\\\n".join(el.latex(matwrap=False) for el in self) +
                    "\\end{bmatrix}")
        else:
            return None

    def _repr_latex_(self):
        return "$$"+self.latex()+"$$"

    def __nonzero__(self):
        "Allows the use of PosyArrays as truth elements."
        return all(p.__nonzero__() for p in self)

    def __bool__(self):
        "Allows the use of PosyArrays as truth elements in python3."
        return all(p.__bool__() for p in self)

    @property
    def c(self):
        try:
            floatarray = np.array(self, dtype='float')
            if not floatarray.shape:
                return floatarray.flatten()[0]
            else:
                return floatarray
        except TypeError:
            raise ValueError("only a posyarray of numbers has a 'c'")

    _eq = np.vectorize(lambda a, b: a == b)

    def __eq__(self, other):
        "Applies == in a vectorized fashion."
        if isinstance(other, Quantity):
            if isinstance(other.magnitude, np.ndarray):
                l = []
                for i, e in enumerate(self):
                    l.append(e == other[i])
                return VectorConstraint(l)
            else:
                return VectorConstraint([e == other for e in self])
        return VectorConstraint(self._eq(self, other))

    def __ne__(self, other):
        "Does type checking, then applies 'not ==' in a vectorized fashion."
        return (not isinstance(other, self.__class__)
                or not all(self._eq(self, other)))

    # inequality constraints
    _leq = np.vectorize(lambda a, b: a <= b)

    def __le__(self, other):
        "Applies '<=' in a vectorized fashion."
        if isinstance(other, Quantity):
            if isinstance(other.magnitude, np.ndarray):
                l = []
                for i, e in enumerate(self):
                    l.append(e <= other[i])
                return VectorConstraint(l)
            else:
                return VectorConstraint([e <= other for e in self])
        return VectorConstraint(self._leq(self, other))

    _geq = np.vectorize(lambda a, b: a >= b)

    def __ge__(self, other):
        "Applies '>=' in a vectorized fashion."
        if isinstance(other, Quantity):
            if isinstance(other.magnitude, np.ndarray):
                l = []
                for i, e in enumerate(self):
                    l.append(e >= other[i])
                return VectorConstraint(l)
            else:
                return VectorConstraint([e >= other for e in self])
        return VectorConstraint(self._geq(self, other))

    def outer(self, other):
        "Returns the array and argument's outer product."
        return PosyArray(np.outer(self, other))

    def sub(self, subs, val=None, require_positive=True):
        "Substitutes into the array"
        return PosyArray([p.sub(subs, val, require_positive) for p in self])

    @property
    def units(self):
        units = None
        for el in self:  # does this need to be done with np.iter?
            if not isinstance(el, Numbers) or el != 0 and not np.isnan(el):
                if units:
                    try:
                        (units/el.units).to("dimensionless")
                    except DimensionalityError:
                        raise ValueError("all elements of a PosyArray must"
                                         " have the same units.")
                else:
                    units = el.units
        return units

    def padleft(self, padding):
        "Returns ({padding}, self[0], self[1] ... self[N])"
        if self.ndim != 1:
            raise NotImplementedError("not implemented for ndim = %s" %
                                      self.ndim)
        padded = PosyArray(np.hstack((padding, self)))
        padded.units  # check that the units are consistent
        return padded

    def padright(self, padding):
        "Returns (self[0], self[1] ... self[N], {padding})"
        if self.ndim != 1:
            raise NotImplementedError("not implemented for ndim = %s" %
                                      self.ndim)
        padded = PosyArray(np.hstack((self, padding)))
        padded.units  # check that the units are consistent
        return padded

    @property
    def left(self):
        "Returns (0, self[0], self[1] ... self[N-1])"
        return self.padleft(0)[:-1]

    @property
    def right(self):
        "Returns (self[1], self[2] ... self[N], 0)"
        return self.padright(0)[1:]


class ConstraintSet(object):

    def __init__(self, constraints, substitutions=None,
                 latex=None, string=None):
        self.constraints = constraints
        cs = self.flatconstraints()
        vks = self.make_varkeys(cs)
        self.substitutions = KeyDict.from_constraints(vks, cs, substitutions)
        if latex:
            self.latex_rep = latex  # TODO
        if string:
            self.string = string  # TODO

    @property
    def varkeys(self):
        return self.make_varkeys()

    def make_varkeys(self, constraints=None):
        self._varkeys = KeySet()
        constraints = constraints if constraints else self.flatconstraints()
        for constr in constraints:
            self._varkeys.update(constr.varkeys)
        return self._varkeys

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
            if hasattr(constr, "as_localposyconstr"):
                localposy = constr.as_localposyconstr(None)
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
                                 "`as_localposyconstr` method or an"
                                 "`as_posyslt1` method, but %s has neither"
                                 % constr)

    def __len__(self):
        return len(self.constraints)

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

    def as_localposyconstr(self, x0):
        self.parse_constraints()
        if not self.localposyconstrs:
            return None
        self.posymap = "sp"
        localposyconstrs = [c.as_localposyconstr(x0)
                            for c in self.localposyconstrs]
        localposyconstrs.extend(self.onlyposyconstrs)
        return ConstraintSet(localposyconstrs, self.substitutions)

    def latex(self):
        return "hihi"  # TODO

    def sub(self, subs, value=None):
        return self  # TODO

    def sensitivities(self, p_senss, m_sensss):
        assert self.all_have_posy_rep
        constr_sens = {}
        var_senss = KeyVector()
        offset = 0
        for i, n_posys in enumerate(self.posymap):
            constr = self.allposyconstrs[i]
            p_ss = p_senss[offset:offset+n_posys]
            m_sss = m_sensss[offset:offset+n_posys]
            constr_sens[str(constr)], v_ss = constr.sensitivities(p_ss, m_sss)
            var_senss += v_ss

        return constr_sens, var_senss

    def sp_sensitivities(self, posyapprox, posy_approx_sens, var_senss):
        constr_sens = {}
        for i, lpc in enumerate(self.localposyconstrs):
            pa = posyapprox[i]
            p_a_s = posy_approx_sens[str(pa)]
            constr_sens[str(lpc)] = lpc.sp_sensitivities(pa, p_a_s, var_senss)
        return constr_sens

    def process_result(self, result):
        processed = {}
        for constraint in self.constraints:
            if hasattr(constraint, "process_result"):
                p = constraint.process_result(result)
                if p:
                    processed.update(p)
        return processed


class VectorConstraint(ConstraintSet):
    pass

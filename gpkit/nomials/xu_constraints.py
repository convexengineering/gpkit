import numpy as np
from .data import simplify_exps_and_cs
from .array import NomialArray
from .nomial_core import Nomial, fast_monomial_str
from .substitution import substitution, parse_subs
from ..constraints import SingleEquationConstraint
from ..small_classes import Strings, Numbers, Quantity
from ..small_classes import HashVector
from ..keydict import KeySet
from ..varkey import VarKey
from ..small_scripts import mag
from .. import units as ureg
from .. import DimensionalityError
from .nomial_math import ScalarSingleEquationConstraint, Signomial

class XuObjective(Signomial):
    """ The objective function defined as y + sum of WiSi where Wi 
        are the weights (defined apriori) and Si are the added 
        slack variables """
    def __init__(self, signomial, y, M, alpha):
        #Signomial.__init__(self, exps, cs, require_positive=True, simplify=True)
        #from .. import SIGNOMIALS_ENABLED
        #if not SIGNOMIALS_ENABLED:
        #    raise TypeError("Cannot initialize XuObjective"
        #                    " outside of a SignomialsEnabled environment.")
        self.signomial = signomial
        self.y = y
        self.M = M
        self.alpha = alpha
        
    def as_gpobj(self, W, S):
        cost = self.y + self.M 
        # W and S are lists of the weights and slack variables 
        # passed to the method as_gpobj
        for i in range(len(W)):
            cost = cost + W[i]*S[i]
        
        return cost
    
    def on_failedsolve(self):
        self.w = self.alpha*self.w
        return self
        
    def on_successfulsolve(self):
        self.w = self.alpha*self.w
        return self
    
class XuConstraint_Obj(Signomial):
    """ The added constraint arising from redefinition of the objective function.
    This constraint has the form (posy + M) / (negy + y) <= 1"""
    
    def __init__(self, signomial, M):
        #Signomial.__init__(self, self.exps, self.cs, require_positive=True, simplify=True)
        #from .. import SIGNOMIALS_ENABLED
        #if not SIGNOMIALS_ENABLED:
        #    raise TypeError("Cannot initialize XuConstraint_Obj"
        #                    " outside of a SignomialsEnabled environment.")
        self.signomial = signomial
        self.M = M
        
    def as_gpconstr(self):
        "Returns the GP constraint according to Xu's algorithm"
        y = Variable('y')
        posy, negy = self.posy_negy()
        oc = PosynomialInequality((posy+M)/(negy+y), "<=", 1)
        oc.substitutions = self.substitutions
        return oc
                
    def on_failedsolve(self):
        return self
        
class XuConstraint_K11(ScalarSingleEquationConstraint):
    """A constraint of the general form signomial <= 1 where the negy + 1
    is a monomial
    
    Stored internally (exps, cs) as a single Signomial (0 >= self)
    Usually initialized via operator overloading, e.g. cc = (y**2 >= 1 + x - y)
    Additionally retains input format (lhs vs rhs) in self.left and self.right
    Form is self.left >= self.right.
    """
    
    def __init__(self, left, oper, right):
        ScalarSingleEquationConstraint.__init__(self, left, oper, right)
        from .. import SIGNOMIALS_ENABLED
        if not SIGNOMIALS_ENABLED:
            raise TypeError("Cannot initialize SignomialInequality"
                            " outside of a SignomialsEnabled environment.")
        if self.oper == "<=":
            plt, pgt = self.left, self.right
        elif self.oper == ">=":
            pgt, plt = self.left, self.right
        else:
            raise ValueError("operator %s is not supported by "
                             "XuConstraint_K11." % self.oper)
        self.nomials = [self.left, self.right]
        self._unsubbed = plt - pgt
        self.nomials.append(self._unsubbed)
        self.substitutions = dict(self.left.values)
        self.substitutions.update(self.right.values)        
		
    def as_posyslt1(self):
        "Returns the posys <= 1 representation of this constraint."
        s = self._unsubbed.sub(self.substitutions, require_positive=False)
        posy, negy = s.posy_negy()
        if posy is 0:
            raise ValueError("XuConstraint_K11 %s became the tautological"
                             " constraint %s %s %s after substitution." %
                             (self, posy, "<=", negy))
        elif negy is 0:
            raise ValueError("XuConstraint_K11 %s became the infeasible"
                             " constraint %s %s %s after substitution." %
                             (self, posy, "<=", negy))
        elif not hasattr(negy, "cs") or len(negy.cs) == 1:
            self.__class__ = PosynomialInequality
            self.__init__(posy, "<=", negy)
            return self._unsubbed   # pylint: disable=no-member

        else:
            raise TypeError("XuConstraint_K11 could not simplify to"
                            " a PosynomialInequality")    
    
	def as_gpconstr(self, x0):
		"Returns the GP constraint according to Xu's algorithm"
        posy, negy = self._unsubbed.posy_negy()
        k11c = PosynomialInequality(posy/(negy+1), "<=", 1)       
        k11c.substitutions = self.substitutions
        return k11c
        
	def on_failedsolve(self):
		return self
        
        
class XuConstraint_K12(ScalarSingleEquationConstraint):
    """An equality constraint of the general form signomial <= 1 where 
    the negy + 1 is not a monomial
    
    Stored internally (exps, cs) as a single Signomial (0 >= self)
    Usually initialized via operator overloading, e.g. cc = (y**2 >= 1 + x - y)
    Additionally retains input format (lhs vs rhs) in self.left and self.right
    Form is self.left >= self.right.
    """
    def __init__(self, left, oper, right):
        ScalarSingleEquationConstraint.__init__(self, left, oper, right)
        from .. import SIGNOMIALS_ENABLED
        if not SIGNOMIALS_ENABLED:
            raise TypeError("Cannot initialize XuConstraint_K12"
                            " outside of a SignomialsEnabled environment.")
        if self.oper == "<=":
            plt, pgt = self.left, self.right
        elif self.oper == ">=":
            pgt, plt = self.left, self.right
        else:
            raise ValueError("operator %s is not supported by "
                             "XuConstraint_K12." % self.oper)
        self.nomials = [self.left, self.right]
        self._unsubbed = plt - pgt
        self.nomials.append(self._unsubbed)
        self.substitutions = dict(self.left.values)
        self.substitutions.update(self.right.values)        
    
    def as_posyslt1(self):
        "Returns the posys <= 1 representation of this constraint."
        s = self._unsubbed.sub(self.substitutions, require_positive=False)
        posy, negy = s.posy_negy()
        if posy is 0:
            raise ValueError("XuConstraint_K12 %s became the tautological"
                             " constraint %s %s %s after substitution." %
                             (self, posy, "<=", negy))
        elif negy is 0:
            raise ValueError("XuConstraint_K12 %s became the infeasible"
                             " constraint %s %s %s after substitution." %
                             (self, posy, "<=", negy))
        elif not hasattr(negy, "cs") or len(negy.cs) == 1:
            self.__class__ = PosynomialInequality
            self.__init__(posy, "<=", negy)
            return self._unsubbed   # pylint: disable=no-member

        else:
            raise TypeError("XuConstraint_K12 could not simplify to"
                            " a PosynomialInequality")    
    
    def as_gpconstr(self, x0):
        "Returns the GP constraint according to Xu's algorithm"
        posy, negy = self._unsubbed.posy_negy()
        
        #on the first solve if x0 not specified in localsolve args, x0 is None
        if x0 is None:
            x0 = {vk: vk.descr["sp_init"] for vk in negy.varlocs if "sp_init" in vk.descr}
        x0.update({var: 1 for var in negy.varlocs if var not in x0})
        x0.update(self.substitutions)
        
        k12c = PosynomialInequality(posy/negy.mono_lower_bound(x0), "<=", 1)
        k12c.substitutions = self.substitutions
     
        return k12c
        
	def on_failedsolve(self):
		return self
        

class XuConstraint_K21(ScalarSingleEquationConstraint):
    """An equality constraint of the general form monomial = monomial
    where posy and negy+1 are both monomials"""

    def __init__(self, left, right):
        ScalarSingleEquationConstraint.__init__(self, left, "=", right)
        from .. import SIGNOMIALS_ENABLED
        if not SIGNOMIALS_ENABLED:
            raise TypeError("Cannot initialize XuConstraint_K21"
                            " outside of a SignomialsEnabled environment.")
        if self.oper is not "=":
            raise ValueError("operator %s is not supported by"
                             " XuConstraint_K21." % self.oper)
        self.nomials = [self.left, self.right]
        self._unsubbed = self._gen_unsubbed()
        self.nomials.append(self._unsubbed)
        self.substitutions = dict(self.left.values)
        self.substitutions.update(self.right.values)        
    
    def _gen_unsubbed(self):
        "Returns the unsubstituted signomial = 1."
        return self.left - self.right
        
    def as_gpconstr(self, x0):
        "Returns the GP constraint according to Xu's algorithm"
        posy, negy = self._unsubbed.posy_negy()
        k21c = PosynomialInequality(posy/(negy+1), "<=", 1)
        k21c.substitutions = self.substitutions
        return k12c
        
    def on_failedsolve(self):
        return self
    

class XuConstraint_K22(ScalarSingleEquationConstraint):
    """An equality constraint of the general form posynomial = monomial
    where posy is a posynomial and negy+1 is a monomial"""
    
    def __init__(self, left, right, s):
        ScalarSingleEquationConstraint.__init__(self, left, "=", right)
        from .. import SIGNOMIALS_ENABLED
        if not SIGNOMIALS_ENABLED:
            raise TypeError("Cannot initialize XuConstraint_K21"
                            " outside of a SignomialsEnabled environment.")
        if self.oper is not "=":
            raise ValueError("operator %s is not supported by"
                             " XuConstraint_K21." % self.oper)
        self.nomials = [self.left, self.right]
        self.s = s
        self._unsubbed = self._gen_unsubbed()
        self.nomials.append(self._unsubbed)
        self.substitutions = dict(self.left.values)
        self.substitutions.update(self.right.values)    

    def _gen_unsubbed(self):
        "Returns the unsubstituted signomial = 1."
        return self.left - self.right
        
    def as_gpconstr(self, x0):
        "Returns the 3 GP constraints according to Xu's algorithm"
        posy, negy = self._unsubbed.posy_negy()
        k22c1 = PosynomialInequality(posy/(negy+1), "<=", 1)
        k22c2 = PosynomialInequality((negy+1)/(self.s*posy.mono_lower_bound(x0)), "<=", 1)
        k22c3 = PosynomialInequality(self.s, ">=", 1)
        return [k22c1, k22c2, k22c3]
    
    def on_failedsolve(self):
        #self.s = self.s + 1

        return self
    
        
class XuConstraint_K23(ScalarSingleEquationConstraint):
    """An equality constraint of the general form monomial = posynomial
    where the posy is a monomial and negy+1 is a posynomial"""

    def __init__(self, left, right, s):
        ScalarSingleEquationConstraint.__init__(self, left, "=", right)
        from .. import SIGNOMIALS_ENABLED
        if not SIGNOMIALS_ENABLED:
            raise TypeError("Cannot initialize XuConstraint_K21"
                            " outside of a SignomialsEnabled environment.")
        if self.oper is not "=":
            raise ValueError("operator %s is not supported by"
                             " XuConstraint_K21." % self.oper)
        self.nomials = [self.left, self.right]
        self.s = s
        self._unsubbed = self._gen_unsubbed()
        self.nomials.append(self._unsubbed)
        self.substitutions = dict(self.left.values)
        self.substitutions.update(self.right.values)    
		
    def _gen_unsubbed(self):
        "Returns the unsubstituted signomial = 1."
        return self.right - self.left        
        
    def as_gpconstr(self, x0):
        "Returns the 3 GP constraints according to Xu's algorithm"
        posy, negy = self._unsubbed.posy_negy()
        k22c1 = PosynomialInequality((negy+1)/posy, "<=", 1)
        k22c2 = PosynomialInequality(posy/(self.s*negy.mono_lower_bound(x0)), "<=", 1)
        k22c3 = PosynomialInequality(self.s, ">=", 1)
        return [k22c1, k22c2, k22c3]
    
    def on_failedsolve(self):
        #self.s = self.s + 1
        
        return self

class XuConstraint_K24(ScalarSingleEquationConstraint):
    """ A equality contraint which does not lie in K21, K22 
    or K23"""
    
    def __init__(self, left, right, s):
        ScalarSingleEquationConstraint.__init__(self, left, "=", right)
        from .. import SIGNOMIALS_ENABLED
        if not SIGNOMIALS_ENABLED:
            raise TypeError("Cannot initialize XuConstraint_K21"
                            " outside of a SignomialsEnabled environment.")
        if self.oper is not "=":
            raise ValueError("operator %s is not supported by"
                             " XuConstraint_K21." % self.oper)
        self.nomials = [self.left, self.right]
        self.s = s
        self._unsubbed = self._gen_unsubbed()
        self.nomials.append(self._unsubbed)
        self.substitutions = dict(self.left.values)
        self.substitutions.update(self.right.values)    
		
    def _gen_unsubbed(self):
        "Returns the unsubstituted signomial = 1."
        return self.left - self.right
        
    def as_gpconstr(self, x0):
        "Returns the 3 GP constraints according to Xu's algorithm"
        posy, negy = self._unsubbed.posy_negy()
        k22c1 = PosynomialInequality(posy/negy.mono_lower_bound(x0), "<=", 1)
        k22c2 = PosynomialInequality((negy+1)/(self.s*posy.mono_lower_bound(x0)), "<=", 1)
        k22c3 = PosynomialInequality(self.s, ">=", 1)
        return [k22c1, k22c2, k22c3]
    
    def on_failedsolve(self):
        #self.s = self.s + 1
        
        return self
from abc import ABCMeta, abstractmethod, abstractproperty


class Constraint(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def varkeys(self):
        pass

    @abstractproperty
    def substitutions(self):
        pass
    
    @substitutions.setter
    def substitutions(self, newvalue):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def latex(self):
        pass

    def _repr_latex_(self):
        return "$$" + self.latex() + "$$"

    @abstractmethod
    def sub(self, subs, value=None):
        pass

    @abstractmethod
    def sub(self, subs, value=None):
        pass

    @abstractmethod
    def process_result(self, result):
        pass


class GlobalConstraint(Constraint):
    __metaclass__ = ABCMeta

    @abstractmethod
    def as_posyslt1(self):
        pass

    @abstractmethod
    def sensitivities_from_dual(self, p_senss, m_sensss):
        pass


class LocalConstraint(Constraint):
    __metaclass__ = ABCMeta

    @abstractmethod
    def as_localposyconstr(self, x0):
        pass

    @abstractmethod
    def sensitivities_from_approx(self, posyapprox, pa_sens, var_senss):
        pass

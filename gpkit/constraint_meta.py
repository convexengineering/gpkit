from abc import ABCMeta, abstractmethod, abstractproperty


class Constraint(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def varkeys(self):
        "Varkeys present in the Constraint"
        pass

    @abstractproperty
    def substitutions(self):
        "Substitutions to apply before returning a posylt1 or approximation."
        pass

    @substitutions.setter
    def substitutions(self, newvalue):
        ".substitutions must be settable"
        pass

    @abstractmethod
    def __str__(self):
        "string representation of the constraint, used for printing programs"
        pass

    @abstractmethod
    def latex(self):
        "latex representation of the constraint, used for printing programs"
        pass

    def _repr_latex_(self):
        "Classes that inherit from Constraint will show as latex in IPython"
        return "$$" + self.latex() + "$$"

    @abstractmethod
    def sub(self, subs, value=None):
        "Returns new object without any of the varkeys in subs.keys()"
        pass

    @abstractmethod
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
        pass


class GPConstraint(Constraint):
    "Interface for constraints representable as posynomials <= 1"
    __metaclass__ = ABCMeta

    @abstractmethod
    def as_posyslt1(self):
        "Returns list of posynomials which must be kept <= 1"
        pass

    @abstractmethod
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
        pass


class LocallyApproximableConstraint(Constraint):
    "Interface for constraints locally approximable as posynomials <= 1"
    __metaclass__ = ABCMeta

    @abstractmethod
    def as_gpconstr(self, x0):
        """Returns GPConstraint approximating this constraint at x0

        When x0 is none, may return a default guess."""
        pass

    @abstractmethod
    def sens_from_gpconstr(self, gpconstr, gpconstr_sens, var_senss):
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

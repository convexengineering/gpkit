"Implements Model"
from collections import defaultdict
from .costed import CostedConstraintSet
from ..varkey import VarKey
from ..nomials import Monomial
from .prog_factories import _progify_fctry, _solve_fctry
from ..geometric_program import GeometricProgram
from .signomial_program import SignomialProgram
from .linked import LinkedConstraintSet
from ..keydict import KeyDict
from .. import SignomialsEnabled


class Model(CostedConstraintSet):
    """Symbolic representation of an optimization problem.

    The Model class is used both directly to create models with constants and
    sweeps, and indirectly inherited to create custom model classes.

    Arguments
    ---------
    cost : Posynomial (optional)
        Defaults to `Monomial(1)`.

    constraints : ConstraintSet or list of constraints (optional)
        Defaults to an empty list.

    substitutions : dict (optional)
        This dictionary will be substituted into the problem before solving,
        and also allows the declaration of sweeps and linked sweeps.

    name : str (optional)
        Allows "naming" a model in a way similar to inherited instances,
        and overrides the inherited name if there is one.

    Attributes with side effects
    ----------------------------
    `program` is set during a solve
    `solution` is set at the end of a solve
    """
    _nums = defaultdict(int)
    name = None
    num = None
    program = None
    solution = None

    def __init__(self, cost=None, constraints=None,
                 substitutions=None, name=None):
        if hasattr(self, "setup"):
            # temporarily fail gracefully for backwards compatibility
            raise RuntimeWarning(
                "setup methods are no longer used in GPkit. "
                "To initialize a model, rename your setup method as "
                "__init__(self, **kwargs) and have it call "
                "Model.__init__(self, cost, constraints, **kwargs) at the end.")
        cost = cost if cost else Monomial(1)
        constraints = constraints if constraints else []
        CostedConstraintSet.__init__(self, cost, constraints, substitutions)
        if name is None:
            name = self.__class__.__name__
        if name and name != "Model":
            self.name = name
            self.num = Model._nums[name]
            Model._nums[name] += 1
            self._add_modelname_tovars(self.name, self.num)

    gp = _progify_fctry(GeometricProgram)
    sp = _progify_fctry(SignomialProgram)
    solve = _solve_fctry(_progify_fctry(GeometricProgram, "solve"))
    localsolve = _solve_fctry(_progify_fctry(SignomialProgram, "localsolve"))

    def link(self, other, include_only=None, exclude=None):
        "Connects this model with a set of constraints"
        lc = LinkedConstraintSet([self, other], include_only, exclude)
        cost = self.cost.sub(lc.linked)
        return Model(cost, lc, lc.substitutions)

    def zero_lower_unbounded_variables(self):
        "Recursively substitutes 0 for variables that lack a lower bound"
        zeros = True
        while zeros:
            # pylint: disable=no-member
            bounds = self.gp(verbosity=0).missingbounds
            zeros = {var: 0 for var, bound in bounds.items()
                     if bound == "lower"}
            self.substitutions.update(zeros)

    def _add_modelname_tovars(self, name, num):
        add_model_subs = KeyDict()
        for vk in self.varkeys:
            descr = dict(vk.descr)
            descr["models"] = descr.pop("models", []) + [name]
            descr["modelnums"] = descr.pop("modelnums", []) + [num]
            newvk = VarKey(**descr)
            add_model_subs[vk] = newvk
            if vk in self.substitutions:
                self.substitutions[newvk] = self.substitutions[vk]
                del self.substitutions[vk]
        with SignomialsEnabled():  # since we're just substituting varkeys.
            self.subinplace(add_model_subs)

    def subconstr_str(self, excluded=None):
        "The collapsed appearance of a ConstraintBase"
        if self.name:
            return "%s_%s" % (self.name, self.num)

    def subconstr_latex(self, excluded=None):
        "The collapsed appearance of a ConstraintBase"
        if self.name:
            return "%s_{%s}" % (self.name, self.num)

    def debug(self):
        "Attempts to diagnose infeasible models."
        from .relax import RelaxConstants, RelaxConstraints
        from .bounded import Bounded

        relaxedconstraints = False
        if self.substitutions:
            feas = RelaxConstants(Bounded(self))
            feas.cost *= self.cost**0.01
        else:
            feas = Model(self.cost, Bounded(self))  # pylint: disable=redefined-variable-type
        try:
            print "Debugging..."
            sol = feas.solve(verbosity=0)

            for constant, sub in self.substitutions.items():
                if sol(constant)/sub >= 1.01:
                    if not relaxedconstraints:
                        print
                        print "RELAXED VARIABLES"
                        relaxedconstraints = True
                    print ("  %s: relaxed from %-.4g to %-.4g"
                           % (constant, sub, sol(constant)))
        except (ValueError, RuntimeWarning):
            print ("Model does not solve with bounded variables"
                   " or relaxed constants.")
        try:
            feas = RelaxConstraints(self)
            feas.cost *= self.cost**0.01
            sol = feas.solve(verbosity=0)

            relaxvals = sol(feas.relaxvars)
            if relaxvals.prod() >= 1.01:
                if relaxedconstraints:
                    print "Alternately, the model solves with:"
                print
                print "RELAXED CONSTRAINTS"
            iterator = enumerate(zip(relaxvals, feas[0][0]))
            for i, (relaxval, constraint) in iterator:
                if relaxval >= 1.01:
                    relax_percent = "%i%%" % ((relaxval-1)*100)
                    print ("  %i: %4s relaxed  (as posynomial: %s <= 1)"
                           % (i, relax_percent, constraint.right))
        except (ValueError, RuntimeWarning):
            print ("Model does not solve with relaxed constraints.")
        # NOTE: If the cost has a very strong relationship to feasibility,
        #       the self.cost**0.01 component above could be the problem.

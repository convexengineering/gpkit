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
from ..small_scripts import mag


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

    # pylint: disable=too-many-locals
    def debug(self, verbosity=1, **solveargs):
        "Attempts to diagnose infeasible models."
        from .relax import ConstantsRelaxed, ConstraintsRelaxed
        from .bounded import Bounded

        sol = None
        relaxed = False
        if self.substitutions:
            feas = ConstantsRelaxed(Bounded(self))
            feas.cost = feas.cost**30 * self.cost
            # NOTE: It hasn't yet been seen but might be possible that
            #       the self.cost component above could cause infeasibility
        else:
            feas = Model(self.cost, Bounded(self))  # pylint: disable=redefined-variable-type
        try:
            print "Debugging..."
            print "_____________________"
            sol = feas.solve(verbosity=verbosity, **solveargs)

            for relax, orig in zip(feas.relaxvars, feas.origvars):
                if sol(relax) >= 1.01:
                    if not relaxed:
                        if sol["boundedness"]:
                            print "and these constants relaxed:"
                        else:
                            print
                            print "Feasible with these constants relaxed:"
                        relaxed = True
                    print ("  %s: relaxed from %-.4g to %-.4g"
                           % (orig, mag(self.substitutions[orig]),
                              mag(sol(orig))))
        except (ValueError, RuntimeWarning):
            print
            print ("Model does not solve with variables bounded"
                   " and constants relaxed.")
        print "_____________________"

        try:
            feas = ConstraintsRelaxed(self)
            feas.cost = feas.cost**30 * self.cost
            sol_constraints = feas.solve(verbosity=verbosity, **solveargs)

            relaxvals = sol_constraints(feas.relaxvars)
            if any(rv >= 1.01 for rv in relaxvals):
                print
                if not relaxed:
                    # then this is the only solution we have to return
                    sol = sol_constraints
                    print "Feasible with relaxed constraints:"
                else:
                    print "Also feasible with these constraints relaxed:"
            iterator = enumerate(zip(relaxvals, feas[0][0]))
            for i, (relaxval, constraint) in iterator:
                if relaxval >= 1.01:
                    relax_percent = "%i%%" % (0.5+(relaxval-1)*100)
                    print ("  %i: %4s relaxed  Canonical form: %s <= %.2f)"
                           % (i, relax_percent, constraint.right, relaxval))

        except (ValueError, RuntimeWarning):
            print
            print ("Model does not solve with relaxed constraints.")

        print "_____________________"
        return sol

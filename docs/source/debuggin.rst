Common Errors and Warnings
*********************

A number of errors and warnings can be returned when attempting to solve a model. Many errors and warnings are indicative of particular bugs.


``AttributeError: 'bool' object has no attribute 'subinplace’`` - indicative of a ``==`` being used where an inequality is required, such as between a monorail and posynomial 
 
``TypeError: unhashable type: ‘xxxx’`` - normally caused by passing something other than a constraint set as the second argument of a ``Model`` constructor
 
``TypeError: SignomialInequality could not simplify to a PosynomialInequality`` - this error occurs when there is a signomial constraint in a problem not solved with ``localsolve``. Either signomials must be enabled and ``localsolve`` used, or the signomial constraint must be turned into a GP constraint.
 
``AttributeError: 'str' object has no attribute 'sub’`` - this is caused by attempting a quantity in the solution dict with an invalid varkey
 
``TypeError: unsupported operand type(s) for ** or pow(): 'int' and 'ParserHelper’`` - this normally indicates a ``Variable`` has been created with invalid units (i.e. units not supported by ``pint``)
 
``RuntimeWarning: Primal solution violates constraint: 1.0000149786 is greater than 1`` - this warning is typically seen in dual infeasible models, see *Dual Infeasible Models* below for more tips on debugging a dual infeasible model.

``RuntimeWarning: Dual cost nan does not match primal cost 1.00122315152`` - this warning is typically seen in dual infeasible models, see *Dual Infeasible Models* below for more tips on debugging a dual infeasible model.

``RuntimeWarning: final status of solver 'cvxopt' was 'unknown', not 'optimal’`` or - this is the most difficult warning to debug. It can be thrown when attempting to solve a dual infeasible model or a primal infeasible model. See *Dual Infeasible Models* and *Primal Infeasible Models* below for more information.

``RuntimeWarning: final status of solver 'mosek' was 'DUAL_INFEAS_CER', not 'optimal’`` - this error is thrown when attempting to solve a dual infeasible model with MOSEK,  see *Dual Infeasible Models* below for more tips on debugging a dual infeasible model.


Dual Infeasible Models
=============

A dual infeasible error typically means that more than one unique solution meets the objective and satisfies the constraints. Usually this means that one or more variables are not sufficiently constrained.  When solving with ``mosek``, the error message will tell you that the solution is dual-infeasible.  When solving with ``cvxopt`` the error for a dual-feasible solution will usually display as a ``Rank`` error or ``cvxopt`` will return ``unknown``.  An example of a dual-feasible model is shown below. This model is dual-infeasible because there are multiple values of ``x`` and ``y`` that satisfy the constraint set even though the obvious value of the objective should be 1.
 
 .. code-block:: python
 
     from gpkit import Variable, Model
     x = Variable("x")
     y = Variable("y")
     m = Model(x*y, [x*y >= 1])
     m.solve()
 
Note: When solving with ``mosek`` this model will actually solve because ``mosek`` is robust enough to handle some dual-feasible problems. The following example is a model that ``mosek`` labels as dual-feasible and will not solve. 
 
 .. code-block:: python
 
     from gpkit import Variable, Model
     x = Variable("x")
     y = Variable("y")
     m = Model(x**0.01 * y, [x*y >= 1])
     m.solve()
 
While this model is very similar to the previous model, ``mosek`` is unable to solve this model and labels it as dual-feasible.

Another common cause of dual-infeasability is a model’s objective applying pressure on a variable in an unexpected direction. If the variable is not bounded in this direction, its value will be pushed to either zero or infinity. This usually results in the solver returning a final status of dual-infeasible. A simple example is given below. ``x`` has no lower bound, and the objective is to minimize ``x``, so the solver pushes ``x`` towards zero and returns dual infeasible.

 .. code-block:: python
 
     from gpkit import Variable, Model
     x = Variable("x")
     m = Model(x, [x <= 1])
     m.solve()

Debugging large, dual infeasible, models can be difficult. The recommended procedure is to use a ``BoundedConstraintSet``, found in ``gpkit.tools``. ``BoundedConstraintSet`` adds additional constraints to the model that bounds each variable to be greater than or equal to a variables ``eps`` and less than or equal to ``1/eps``. The default value for ``eps`` is 1e-30. This prevents variables from being truly unbounded and allows most dual infeasible models to solve. By inspecting the solution, or by also making use of the a ``Tight Constraint Set``, it is easy to determine which variables are unbounded and modify constraints as necessary. Below, a BoundedConstraintSet is used to make the previous model solvable.

  .. code-block:: python
 
     from gpkit import Variable, Model
     from gpkit.tools import BoundedConstraintSet
     x = Variable("x")
     m = Model(x, BoundedConstraintSet([x <= 1]))
     m.solve()

With the formulation above, ``x`` has a lower bound at 1e-30 so the solver returns a solution with cost is 1e-30.


Primal Infeasible Models
=============


Common Errors and Warnings
*********************

A number of errors and warnings can be raised when attempting to solve a model.

- ``TypeError: unhashable type: ‘xxxx’``
    - can be caused by passing something other than a constraint set as the second argument of a ``Model`` constructor

- ``InvalidGPConstraint: SignomialInequality could not simplify to a PosynomialInequality``
    - this error occurs when there is a signomial constraint in a problem not solved with ``localsolve``. Either signomials must be enabled and ``localsolve`` used, or the signomial constraint must be turned into a GP constraint.

- ``AttributeError: 'str' object has no attribute 'sub’``
    - this is caused by attempting to access an item in the solution dict with an invalid varkey

- ``TypeError: unsupported operand type(s) for ** or pow(): 'int' and 'ParserHelper’``
    - this normally indicates a ``Variable`` has been created with invalid units (i.e. units not supported by ``pint``)

- ``RuntimeWarning: Primal solution violates constraint: 1.0000149786 is greater than 1``
    - this warning may be seen in dual infeasible models, see *Dual Infeasible Models* below for more tips on debugging a dual infeasible model.

- ``RuntimeWarning: Dual cost nan does not match primal cost 1.00122315152``
    - this warning may be seen in dual infeasible models, see *Dual Infeasible Models* below for more tips on debugging a dual infeasible model.

- ``RuntimeWarning: final status of solver 'cvxopt' was 'unknown', not 'optimal’`` or ``RuntimeWarning: final status of solver 'mosek' was ‘UNKNOWN’, not 'optimal’.``
    - this is the most difficult warning to debug. It can be thrown when attempting to solve a dual infeasible model or a primal infeasible model. See *Dual Infeasible Models* and *Primal Infeasible Models* below for more information.

- ``RuntimeWarning: final status of solver 'mosek' was 'DUAL_INFEAS_CER', not 'optimal’``
    - this error is thrown when attempting to solve a dual infeasible model with MOSEK,  see *Dual Infeasible Models* below for more tips on debugging a dual infeasible model.


Dual Feasible and Infeasible Models
===================================

Debugging
---------

In some cases a model will not solve because its variables are pushing to 0 or infinity. If the solver catches such behaviour it will return ``dual infeasible`` (or equivalent), but sometimes solvers do not catch it and return ``unknown``.

``gpkit.constraints.bounded.BoundedConstraintSet`` is a
simple tool that attempts to detect unbounded variables and get unbounded models to solve by adding extremely large upper bounds and extremely small lower bounds to all variables in a ConstraintSet.

When a model with an BoundedConstraintSet is solved, it checks whether any variables slid off to the bounds, notes this in the solution dictionary and prints a warning (if verbosity is greater than 0).

For example, Mosek returns ``DUAL_INFEAS_CER`` when attempting to solve the following model:

.. literalinclude:: examples/unbounded.py

Upon viewing the printed output,

.. literalinclude:: examples/unbounded_output.txt

it becomes clear that the problem is, unsurprisingly, that ``x`` is unbounded below in the original model.

Details
-------

A dual feasible solution typically means that more than one unique solution meets the objective.   An example of a dual-feasible model is shown below. This model is dual-infeasible because there are multiple values of ``x`` and ``y`` that satisfy the constraint set and yield the globally optimum cost of 0.5.

.. literalinclude:: examples/subinplace.py

``cvxopt`` and ``Mosek`` both solve the above model and output a cost of 0.5, however, the values of ``x`` and ``y`` will be different, illustrating how the model is dual feasible.

The following is an example of a dual-infeasible problem. While the difference is slight, this cannot be solved by either ``mosek`` or ``cvxopt``.  ``cvxopt`` will again give a ``Rank`` error.  ``Mosek`` can identify deal-infeasible models and the error message will label it as such. Typically, this type of error means that one or more variables are not sufficiently bounded.

.. literalinclude:: examples/dual_infeasible_ex2.py

Another common cause of dual-infeasability occurs when a constrain applies pressure on a variable in an unexpected direction and pushs its value to either zero or infinity. When this occurs, Mosek usually returns a final status of dual-infeasible while cvxopt will return a final solver status of unknown. A simple example is given below. ``x`` has no upper bound, and the objective is to minimize ``1/x``, so the solver pushes ``x`` towards infinitiy and returns dual infeasible.

.. literalinclude:: examples/dual_infeasible_ex.py

Debugging large, dual infeasible, models can be difficult. The recommended procedure is to use a ``BoundedConstraintSet``, found in ``gpkit.tools``. ``BoundedConstraintSet`` adds additional constraints to the model that bounds each variable to be greater than or equal to ``eps``, and less than or equal to ``1/eps``. The default value for ``eps`` is 1e-30. This prevents variables from being truly unbounded and allows most dual infeasible models to solve. By inspecting the solution, or by also making use of the a ``Tight Constraint Set``, it is easy to determine which variables are unbounded and modify constraints as necessary. Below, a BoundedConstraintSet is used to make the previous model solvable.

.. literalinclude:: examples/BoundedConstraintSet_ex.py

With the formulation above, ``x`` has a lower bound at 1e-30, so the solver returns a solution with cost 1e-30.


Primal Infeasible Models
========================

A model is primal infeasible when it has no feasible region. This means there is no point which simultaneously satisfies all of the model’s constraints. A simple example is presented below.

.. literalinclude:: examples/primal_infeasible_ex1.py

It is not possible for ``x*y`` to be less than 1.5 while ``x`` is greater than 1 and ``y`` is greater than 2.

A common bug in large models that use ``substitutions`` is to substitute overly constraining values in for variables that make the model primal infeasible. An example of this is given below.

.. literalinclude:: examples/primal_infeasible_ex2.py

Since ``y`` is now set to 2 and ``x`` can be no less than 1, it is again impossible for ``x*y`` to be less than 1.5 and the model is primal infeasible. If ``y`` was instead set to 1, the model would be feasible and the cost would be 1.

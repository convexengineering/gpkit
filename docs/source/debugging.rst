Debugging Models
****************

A number of errors and warnings may be raised when attempting to solve a model. 
A model may be primal infeasible: there is no possible solution that satisfies all constraints. A model may be dual infeasible: the optimal value of one or more variables is 0 or infinity (negative and positive infinity in logspace).

For a GP model that does not solve, solvers may be able to prove its primal or dual infeasibility, or may return an unknown status.

GPkit contains several tools for diagnosing which constraints and variables might be causing infeasibility. The first thing to do with a model ``m`` that won't solve is to run ``m.debug()``, which will search for changes that would make the model feasible:

.. literalinclude:: examples/debug.py

.. literalinclude:: examples/debug_output.txt

Note that certain modeling errors (such as omitting or forgetting a constraint) may be difficult to diagnose from this output.


Potential errors and warnings
=============================

- ``RuntimeWarning: final status of solver 'mosek' was 'DUAL_INFEAS_CER', not 'optimal’``
    - The solver found a certificate of dual infeasibility: the optimal value of one or more variables is 0 or infinity. See *Dual Infeasibility* below for debugging advice.

- ``RuntimeWarning: final status of solver 'mosek' was 'PRIM_INFEAS_CER', not 'optimal’``
    - The solver found a certificate of primal infeasibility: no possible solution satisfies all constraints. See *Primal Infeasibility* below for debugging advice.

- ``RuntimeWarning: final status of solver 'cvxopt' was 'unknown', not 'optimal’`` or ``RuntimeWarning: final status of solver 'mosek' was ‘UNKNOWN’, not 'optimal’.``
    - The solver could not solve the model or find a certificate of infeasibility. This may indicate a dual infeasible model, a primal infeasible model, or other numerical issues. Try debugging with the techniques in *Dual* and *Primal Infeasibility* below.

- ``RuntimeWarning: Primal solution violates constraint: 1.0000149786 is greater than 1``
    - this warning indicates that the solver-returned solution violates a constraint of the model, likely because the solver's tolerance for a final solution exceeds GPkit's tolerance during solution checking. This is sometimes seen in dual infeasible models, see *Dual Infeasibility* below. If you run into this, please note on `this GitHub issue <https://github.com/convexengineering/gpkit/issues/753>`_ your solver and operating system.

- ``RuntimeWarning: Dual cost nan does not match primal cost 1.00122315152``
    - Similarly to the above, this warning may be seen in dual infeasible models, see *Dual Infeasibility* below.

.. 
    note: remove the above when we match solver tolerance in GPkit (issue #753)


Dual Infeasibility
==================

In some cases a model will not solve because the optimal value of one or more variables is 0 or infinity (negative or positive infinity in logspace). Such a problem is `dual infeasible` because the GP's dual problem, which determines the optimal values of the sensitivites, does not have any feasible solution. If the solver can prove that the dual is infeasible, it will return a dual infeasibility certificate. Otherwise, it may finish with a solution status of ``unknown``.

``gpkit.constraints.bounded.Bounded`` is a
simple tool that can be used to detect unbounded variables and get dual infeasible models to solve by adding extremely large upper bounds and extremely small lower bounds to all variables in a ConstraintSet.

When a model with a Bounded ConstraintSet is solved, it checks whether any variables slid off to the bounds, notes this in the solution dictionary and prints a warning (if verbosity is greater than 0).

For example, Mosek returns ``DUAL_INFEAS_CER`` when attempting to solve the following model:

.. literalinclude:: examples/unbounded.py

Upon viewing the printed output,

.. literalinclude:: examples/unbounded_output.txt

The problem, unsurprisingly, is that the cost ``1/x`` has no lower bound because ``x`` has no upper bound.

For details read the `Bounded <autodoc/gpkit.constraints.html#module-gpkit.constraints.bounded>`__ docstring.



Primal Infeasibility
====================

A model is primal infeasible when there is no possible solution that satisfies all constraints. A simple example is presented below.

.. literalinclude:: examples/primal_infeasible_ex1.py

It is not possible for ``x*y`` to be less than 1.5 while ``x`` is greater than 1 and ``y`` is greater than 2.

A common bug in large models that use ``substitutions`` is to substitute overly constraining values in for variables that make the model primal infeasible. An example of this is given below.

.. literalinclude:: examples/primal_infeasible_ex2.py

Since ``y`` is now set to 2 and ``x`` can be no less than 1, it is again impossible for ``x*y`` to be less than 1.5 and the model is primal infeasible. If ``y`` was instead set to 1, the model would be feasible and the cost would be 1.

Relaxation
----------

If you suspect your model is primal infeasible, you can find the nearest primal feasible version of it by relaxing constraints: either relaxing all constraints by the smallest number possible (that is, dividing the less-than side of every constraint by the same number), relaxing each constraint by its own number and minimizing the product of those numbers, or changing each constant by the smallest total percentage possible.

.. literalinclude:: examples/relaxation.py

.. literalinclude:: examples/relaxation_output.txt

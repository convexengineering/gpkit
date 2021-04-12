Advanced Commands
*****************

.. _migp:

Choice Variables
================
If MOSEK 9 is installed, GPkit supports discretized free variables with the ``mosek_conif`` solver.
Choice variables are free in the sense of having multiple possible choices, but discretized in the
sense of having a limited set of possible solutions.

.. literalinclude:: examples/migp.py

If solved with the mosek_conif solver, the script above will print::

    Optimal Cost
    ------------
     [ 1.5       2.15      2.8       3.22     ... ]

    ~~~~~~~~
    WARNINGS
    ~~~~~~~~
    No Dual Solution
    ----------------
    This model has the discretized choice variables [x] and hence no dual
    solution. You can fix those variables to their optimal value and get
    sensitivities to the resulting continuous problem by updating your model's
    substitions with `sol["choicevariables"]`.
    ~~~~~~~~

    Swept Variables
    ---------------
    numerator : [ 0.5
                  1.15
                  1.8
                  2.45
                  3.1
                  3.75
                  4.4
                  5.05
                  5.7
                  6.35
                  7         ]

    Free Variables
    --------------
    x : [ 1
          1
          1
          2
          2
          2
          2
          2
          2
          3
          3         ]

Note that the optimal values for ``x`` are discretized, clicking from 1
to 2 to 3 during the sweep, and that the solution has no dual variables.


Derived Variables
=================

Evaluated Fixed Variables
-------------------------

Some fixed variables may be derived from the values of other fixed variables.
For example, air density, viscosity, and temperature are functions of altitude.
These can be represented by a substitution or value that is a one-argument function
accepting ``model.substitutions`` (for details, see `Substitutions`_ below).

.. literalinclude:: examples/evaluated_fixed_variables.py

These functions are automatically differentiated with the `ad <https://pypi.org/project/ad/>`_ package to provide more accurate sensitivities. In some cases may require using functions from the ``ad.admath`` instead of their python or numpy equivalents; the `ad documentation <https://pypi.org/project/ad/>`_ contains details on how to do this.


Evaluated Free Variables
------------------------

Some free variables may be evaluated from the values of other (non-evaluated) free variables
after the optimization is performed. For example, if the efficiency :math:`\nu` of a motor is not a GP-compatible
variable, but :math:`(1-\nu)` is a valid GP variable, then :math:`\nu` can be calculated after solving.
These evaluated free variables can be represented by a ``Variable`` with ``evalfn`` metadata.
When constructing an ``evalfn``, remember that square-bracket access to variables
pulls out magnitudes: use round-bracket access (i.e. ``v(var)``) to ensure unit correctness.

.. literalinclude:: examples/evaluated_free_variables.py

Note that this variable should not be used in constructing your model!
For evaluated variables that can be used during a solution, see :ref:`sgp`.


.. _Sweeps:

Sweeps
======

Sweeps are useful for analyzing tradeoff surfaces. A sweep “value” is an Iterable of numbers, e.g. ``[1, 2, 3]``. The simplest way to sweep a model is to call ``model.sweep({sweepvar: sweepvalues})``, which will return a solution array but not change the model's substitutions dictionary. If multiple ``sweepvars`` are given, the method will run them all as independent one-dimensional sweeps and return a list of one solution per sweep. The method ``model.autosweep({sweepvar: (start, end)}, tol=0.01)`` behaves very similarly, except that only the bounds of the sweep need be specified and the region in betwen will be swept to a maximum possible error of tol in the log of the cost. For details see `1D Autosweeps`_ below.


Sweep Substitutions
-------------------
Alternatively, or to sweep a higher-dimensional grid, Variables can swept with a substitution value takes the form ``('sweep', Iterable)``, such as ``('sweep', np.linspace(1e6, 1e7, 100))``. During variable declaration, giving an Iterable value for a Variable is assumed to be giving it a sweep value: for example, ``x = Variable("x", [1, 2, 3])`` will sweep ``x`` over three values.

Vector variables may also be substituted for: ``{y: ("sweep" ,[[1, 2], [1, 2], [1, 2]])}`` will sweep :math:`y\ \forall~y_i\in\left\{1,2\right\}`. These sweeps cannot be specified during Variable creation.

A Model with sweep substitutions will solve for all possible combinations: e.g., if there’s a variable ``x`` with value ``('sweep', [1, 3])`` and a variable ``y`` with value ``('sweep', [14, 17])`` then the gp will be solved four times, for :math:`(x,y)\in\left\{(1, 14),\ (1, 17),\ (3, 14),\ (3, 17)\right\}`. The returned solutions will be a one-dimensional array (or 2-D for vector variables), accessed in the usual way.

1D Autosweeps
-------------
If you're only sweeping over a single variable, autosweeping lets you specify a
tolerance for cost error instead of a number of exact positions to solve at.
GPkit will then search the sweep segment for a locally optimal number of sweeps
that can guarantee a max absolute error on the log of the cost.

Accessing variable and cost values from an autosweep is slightly different, as
can be seen in this example:

.. literalinclude:: examples/autosweep.py

If you need access to the raw solutions arrays, the smallest simplex tree containing
any given point can be gotten with ``min_bst = bst.min_bst(val)``, the extents of that tree with ``bst.bounds`` and solutions of that tree with ``bst.sols``. More information is in ``help(bst)``.


Tight ConstraintSets
====================

Tight ConstraintSets will warn if any inequalities they contain are not
tight (that is, the right side does not equal the left side) after solving. This
is useful when you know that a constraint *should* be tight for a given model,
but representing it as an equality would be non-convex.

.. literalinclude:: examples/tight_constraintsets.py

Loose ConstraintSets
====================

Loose ConstraintSets will warn if any GP-compatible constraints they contain are
not loose (that is, their sensitivity is above some threshold after solving). This
is useful when you want a constraint to be inactive for a given model because
it represents an important model assumption (such as a fit only valid over a particular interval).

.. literalinclude:: examples/loose_constraintsets.py

Substitutions
=============

Substitutions are a general-purpose way to change every instance of one variable into either a number or another variable.

Substituting into Posynomials, NomialArrays, and GPs
-----------------------------------------------------

The examples below all use Posynomials and NomialArrays, but the syntax is identical for GPs (except when it comes to sweep variables).

.. literalinclude:: examples/substitutions.py

Here the variable ``x`` is being replaced with ``3`` in three ways: first by substituting for ``x`` directly, then by substituting for the ``VarKey("x")``, then by substituting the string "x". In all cases the substitution is understood as being with the VarKey: when a variable is passed in the VarKey is pulled out of it, and when a string is passed in it is used as an argument to the Posynomial's ``varkeys`` dictionary.

Substituting multiple values
----------------------------

.. literalinclude:: examples/sub_multi_values.py

To substitute in multiple variables, pass them in as a dictionary where the keys are what will be replaced and values are what it will be replaced with. Note that you can also substitute for VectorVariables by their name or by their NomialArray.

Substituting with nonnumeric values
-----------------------------------

You can also substitute in sweep variables (see Sweeps_), strings, and monomials:

.. literalinclude:: examples/substituting_nonnumeric_values.py

Note that units are preserved, and that the value can be either a string (in which case it just renames the variable), a varkey (in which case it changes its description, including the name) or a Monomial (in which case it substitutes for the variable with a new monomial).

Updating ConstraintSet substitutions
------------------------------------
ConstraintSets have a ``.substitutions`` KeyDict attribute which will be substituted before solving.
This KeyDict accepts variable names, VarKeys, and Variable objects as keys, and can be updated (or deleted from)
like a regular Python dictionary to change the substitutions that will be used at solve-time. If a ConstraintSet itself
contains ConstraintSets, it and all its elements share pointers to the same substitutions dictionary object,
so that updating any one of them will update all of them.


Fixed Variables
---------------

When a Model is created, any fixed Variables are used to form a dictionary: ``{var: var.descr["value"] for var in self.varlocs if "value" in var.descr}``. This dictionary in then substituted into the Model's cost and constraints before the ``substitutions`` argument is (and hence values are supplanted by any later substitutions).

``solution.subinto(p)`` will substitute the solution(s) for variables into the posynomial ``p``, returning a NomialArray. For a non-swept solution, this is equivalent to ``p.sub(solution["variables"])``.

You can also substitute by just calling the solution, i.e. ``solution(p)``. This returns a numpy array of just the coefficients (``c``) of the posynomial after substitution, and will raise a` ``ValueError``` if some of the variables in ``p`` were not found in ``solution``.

Freeing Fixed Variables
-----------------------

After creating a Model, it may be useful to "free" a fixed variable and resolve.  This can be done using the command ``del m.substitutions["x"]``, where ``m`` is a Model.  An example of how to do this is shown below.

.. literalinclude:: examples/freeing_fixed_variables.py

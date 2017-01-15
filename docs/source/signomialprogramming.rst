.. _signomialprogramming:

Signomial Programming
*********************

Signomial programming finds a local solution to a problem of the form:


.. math:: \begin{array}{lll}\text{}
    \text{minimize} & g_0(x) & \\
    \text{subject to} & f_i(x) = 1, & i = 1,....,m \\
                      & g_i(x) - h_i(x) \leq 1, & i = 1,....,n
                      \end{array}

where each :math:`f` is monomial while each :math:`g` and :math:`h` is a posynomial.

This requires multiple solutions of geometric programs, and so will take longer to solve than an equivalent geometric programming formulation.

In general, when given the choice of which variables to include in the positive-posynomial / :math:`g` side of the constraint, the modeler should:

    #. maximize the number of variables in :math:`g`,
    #. prioritize variables that are in the objective,
    #. then prioritize variables that are present in other constraints.

The ``.localsolve`` syntax was chosen to emphasize that signomial programming returns a local optimum. For the same reason, calling ``.solve`` on an SP will raise an error.

By default, signomial programs are first solved conservatively (by assuming each :math:`h` is equal only to its constant portion) and then become less conservative on each iteration.

Example Usage
=============

.. literalinclude:: examples/simple_sp.py

When using the ``localsolve`` method, the ``reltol`` argument specifies the relative tolerance of the solver: that is, by what percent does the solution have to improve between iterations? If any iteration improves less than that amount, the solver stops and returns its value.

If you wish to start the local optimization at a particular point :math:`x_k`, however, you may do so by putting that position (a dictionary formatted as you would a substitution) as the ``xk`` argument.

.. _sgp:

Sequential Geometric Programs
=============================

The method of solving local GP approximations of a non-GP compatible model can be generalized, at the cost of the general smoothness and lack of a need for trust regions that SPs guarantee.

For some applications, it is useful to call external codes which may not be GP compatible.  Imagine we wished to solve the following optimization problem:

.. math:: \begin{array}{lll}\text{}
    \text{minimize} & y & \\
    \text{subject to} & y \geq \sin(x) \\
                      & \frac{\pi}{4} \leq x \leq \frac{\pi}{2}
                      \end{array}

This problem is not GP compatible due to the sin(x) constraint.  One approach might be to take the first term of the Taylor expansion of sin(x) and attempt to solve:

.. literalinclude:: examples/sin_approx_example.py

.. literalinclude:: examples/sin_approx_example_output.txt

We can do better, however, by utilizing some built in functionality of GPkit.
For simple cases with a single Variable, GPkit looks for ``externalfn`` metadata:

.. literalinclude:: examples/external_sp2.py

.. literalinclude:: examples/external_sp2_output.txt

However, for external functions not intrinsically tied to a single variable it's best to
use the full ConstraintSet API, as follows:

Assume we have some external code which is capable of evaluating our incompatible function:

.. literalinclude:: examples/external_function.py

Now, we can create a ConstraintSet that allows GPkit to treat the incompatible constraint as though it were a signomial programming constraint:

.. literalinclude:: examples/external_constraint.py

and replace the incompatible constraint in our GP:

.. literalinclude:: examples/external_sp.py

.. literalinclude:: examples/external_sp_output.txt

which is the expected result.  This method has been generalized to larger problems, such as calling XFOIL and AVL.

If you wish to start the local optimization at a particular point :math:`x_0`, however, you may do so by putting that position (a dictionary formatted as you would a substitution) as the ``x0`` argument

.. Mention sp_init varkey arg. Should that be x0 instead for consistency?

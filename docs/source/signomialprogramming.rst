Signomial Programming
*********************

Signomial programming finds a local solution to a problem of the form:


.. math:: \begin{array}[lll]\text{}
    \text{minimize} & g_0(x) & \\
    \text{subject to} & f_i(x) = 1, & i = 1,....,m \\
                      & g_i(x) - h_i(x) \leq 1, & i = 1,....,n
                      \end{array}

where each :math:`f` is monomial while each :math:`g` and :math:`h` is a posynomial.

This requires multiple solutions of geometric programs, and so will take longer to solve than an equivalent geometric programming formulation.

The specification of the signomial problem affects its solve time in a nuanced way: ``gpkit.SP(x, [x >= 0.1, x+y >= 1, y <= 0.1]).localsolve()`` takes about four times as many iterations to solve as ``gpkit.SP(x, [x >= 1-y, y <= 0.1]).localsolve()``, despite the two formulations being arithmetically equivalent.

In general, when given the choice of which variables to include in the positive-posynomial / :math:`g` side of the constraint, the modeler should:

    #. maximize the number of variables in :math:`g`,
    #. prioritize variables that are in the objective,
    #. then prioritize variables that are present in other constraints.

The syntax ``SP.localsolve`` is chosen to emphasize that signomial programming returns a local optimum. For the same reason, calling ``SP.solve`` will raise an error.

By default, signomial programs are first solved conservatively (by assuming each :math:`h` is equal only to its constant portion) and then become less conservative on each iteration.

Example Usage
=============

.. literalinclude:: examples/simple_sp.py

When using the ``localsolve`` method, the ``reltol`` argument specifies the relative tolerance of the solver: that is, by what percent does the solution have to improve between iterations? If any iteration improves less than that amount, the solver stops and returns its value.

If you wish to start the local optimization at a particular point :math:`x_k`, however, you may do so by putting that position (a dictionary formatted as you would a substitution) as the ``xk`` argument.

Calling to External Codes
=========================
For some applications, it is useful to be able to call external codes which may not be GP compatible.  Imagine we wished to solve the following optimization problem:

.. math:: \begin{array}[lll]\text{}
    \text{minimize} & y & \\
    \text{subject to} & y \geq \sin(x) \\
                      & \frac{\pi}{4} \leq x \leq \frac{\pi}{2}
                      \end{array}

This problem is not GP compatible due to the sin(x) constraint.  One approach might be to take the first term of the Taylor expansion of sin(x) and attempt to solve:

.. literalinclude:: examples/sin_approx_example.py

.. literalinclude:: examples/sin_approx_example_out.txt

However, we can do better by utilizing some built in functionality of GPkit.  Assume we have some external code which is capable of evaluating our incompatible function:

.. literalinclude:: examples/external_function.py

Now, we can write a class that will in essence replace the incompatible constraint:

.. literalinclude:: examples/external_class.py

and replace the incompatible constraint in our GP:

.. literalinclude:: examples/external_GP.py

.. literalinclude:: examples/external_GP_out.txt

which is the expected result.  This method has been generalized to much larger scale problems, including calling XFOIL and AVL.




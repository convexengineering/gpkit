Geometric Programming 101
*************************

What is a GP?
=============

A Geometric Program (GP) is a powerful type of constrained non-linear optimization problem.

A GP is made up of special types of functions called *monomials* and *posynomials*. In the context of a GP, a monomial is defined as:

.. math::

   f(x) = c x_1^{a_1} x_2^{a_2} ... x_n^{a_n}

where :math:`c` is a positive constant, :math:`x_{1..n}` are the decision variables, and :math:`a_{1..n}` are real exponents.

Building on this, a posynomial is defined as a sum of monomials:

.. math::

   g(x) = \sum_{k=1}^K c_k x_1^{a_1k} x_2^{a_2k} ... x_n^{a_nk}


Using these definitions, we can now write a GP in Standard Form:

.. math:: \begin{array}[lll]\text{}
    \text{minimize} & f_0(x) & \\
    \text{subject to} & f_i(x) = 1, & i = 1,....,m \\
                      & g_i(x) \leq 1, & i = 1,....,n
                      \end{array}


Why are GPs special?
====================

Geometric programs have several really nice properties:

    #. Unlike most non-linear optimization problems, large GPs can be **solved extremely quickly**
    #. If there exists an optimal solution to a GP, it is guaranteed to be **globally optimal**
    #. Many **practical problems** can be written as GPs, either in exact form or to a close approximation

These properties arise because a GP can easily be transformed into a *convex optimization problem* using a log transformation. 

Where can I learn more?
=======================

To learn more about GPs, take a look at the following resources:

    * `A Tutorial on Geometric Programming <http://stanford.edu/~boyd/papers/pdf/gp_tutorial.pdf>`_,  Boyd et al.
    * `Geometric Programming for Aircraft Design Optimization <http://www.cs.berkeley.edu/~pabbeel/papers/2012_gp_design.pdf>`_, Hoburg, Abbeel 2012

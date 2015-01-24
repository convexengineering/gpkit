What is a GP?
*************

A geometric program (GP) is a special type of constrained non-linear optimization problem.

Geometric programs have several really nice properties:

    #. Unlike most optimization problems, large GPs can be solved extremely quickly
    #. If there exists a solution to a GP, it is guaranteed to be globally optimal
    #. Many practical problems turn out to be GPs, or well-approximated by GPs

.. todo: describe solution methods and benefits

Monomials and Posynomials
=========================
Monomials and posynomials are the building blocks of the GP formulation.  Monomial functions transform in logspace to affine functions and posynomials transform to log-sum-exp functions, which produces a convex optimization problem.  In the context of a GP, a monomial function is defined as:

.. math::

   f(x) = c x_1^{a_1} x_2^{a_2} ... x_n^{a_n}

where :math:`c` is a positive constant, :math:`x_{1..n}` are the decision variables, and :math:`a_{1..n}` are real exponents.  Building on this, a posynomial is defined as a sum of monomials:

.. math::

   g(x) = \sum_{k=1}^K c_k x_1^{a_1k} x_2^{a_2k} ... x_n^{a_nk}

With these terms defined, we can now define a GP in Standard Form:

.. math:: \begin{array}[lll]\text{}
    \text{minimize} & f_0(x) & \\
    \text{subject to} & f_i(x) = 1, & i = 1,....,m \\
                      & g_i(x) \leq 1, & i = 1,....,p
                      \end{array}


To learn more about GPs, take a look at the following resources:

    * `A Tutorial on Geometric Programming <http://stanford.edu/~boyd/papers/pdf/gp_tutorial.pdf>`_,  Boyd et al.
    * `Geometric Programming for Aircraft Design Optimization <http://www.cs.berkeley.edu/~pabbeel/papers/2012_gp_design.pdf>`_, Hoburg, Abbeel 2012

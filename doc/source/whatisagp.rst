What is a GP?
*************

A geometric program (GP) is a special type of constrained optimization problem.

Geometric programs are useful because: 

#. unlike most optimization problems,
   large GPs can be solved extremely efficiently, and
#. many practical problems turn out to be GPs, or well-approximated by GPs.



.. todo: describe solution methods and benefits

Monomials and Posynomials
=========================
Monomials and posynomials are the building blocks of the GP formulation.  Monomial functions transform in logspace to affine functions and posynomials transform to log-sum-exp functions, which produces a convex optimization problem.  In the context of a GP, a monomial function is defined as:

.. math::

   f(x) = c x_1^{a_1} x_2^{a_2} ... x_n^{a_n}

where :math:`c` is a positive constant, :math:`x_{1..n}` are the decision variables, and :math:`a_{1..n}` are real exponents.  Building on this, a posynomial is defined as a sum of monomials:

.. math::
   
   g(x) = \sum_{k=1}^K c_k x_1^{a_1k} x_2^{a_2k} ... x_n^{a_nk}

where :math:`K` is the number of terms in the posynomial. With these terms defined, we can now define a GP in Standard Form:

.. math::

   minimize    f_0(x) \\
   subject to  f_i(x) = 1, i = 1,....,m \\
               g_i(x) \leq 1, i = 1,....,p


To learn more about GPs, take a look at the following resources:
* `GP Tutorial <http://stanford.edu/~boyd/papers/pdf/gp_tutorial.pdf>`_
.. * Citation of Hoburg 2012, Boyd tutorial

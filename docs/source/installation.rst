Installation Instructions
*************************

If you encounter any bugs during installation, please email gpkit@mit.edu.

Mac OS X
========

1. Install Python and build dependencies
++++++++++++++++++++++++++++++++++++++++
  - Install the Python 2.7 version of `Anaconda <http://continuum.io/downloads>`_.
  - If you don't want to install Anaconda, you'll need gcc and pip, and may find sympy, scipy, and iPython Notebook useful.
  - If ``which gcc`` does not return anything, install the `Apple Command Line Tools <https://developer.apple.com/downloads/index.action?=command%20line%20tools>`_.
  - Run ``pip install ctypesgen`` in the Terminal.


2. Install either the MOSEK or CVXOPT GP solvers
++++++++++++++++++++++++++++++++++++++++++++++++

  - Download `CVXOPT <http://cvxopt.org/download/index.html>`_, then:
      - Read the `official instructions and requirements <http://cvxopt.org/install/index.html#standard-installation>`_
      - In the Terminal, navigate to the ``cvxopt`` folder
      - Run ``python setup.py install``

  - Download `MOSEK <http://mosek.com/resources/downloads>`_, then:
      - Move the ``mosek`` folder to your home directory
      - Follow `these steps for Mac <http://docs.mosek.com/7.0/toolsinstall/Mac_OS_X_installation.html>`_.
      - Request an `academic license file <http://license.mosek.com/academic>`_ and put it in ``~/mosek/``


3. Install GPkit
++++++++++++++++
  - Run ``pip install gpkit`` at the command line.
  - Run ``python -c "import gpkit.tests; gpkit.tests.run()"``
  - If you want units support, install pint with ``pip install pint``.



Linux
=====

1. Install either the MOSEK or CVXOPT GP solvers
++++++++++++++++++++++++++++++++++++++++++++++++

  - Download `CVXOPT <http://cvxopt.org/download/index.html>`_, then:
      - Read the `official instructions and requirements`_
      - In a terminal, navigate to the ``cvxopt`` folder
      - Run ``python setup.py install``

  - Download `MOSEK <http://mosek.com/resources/downloads>`_, then:
      - Move the ``mosek`` folder to your home directory
      - Follow `these steps for Linux <http://docs.mosek.com/7.0/toolsinstall/Linux_UNIX_installation_instructions.html>`_.
      - Request an `academic license file <http://license.mosek.com/academic>`_ and put it in ``~/mosek/``


2. Install GPkit
++++++++++++++++
  - Run ``pip install ctypesgen`` at the command line.
  - Run ``pip install gpkit`` at the command line.
  - Run ``python -c "import gpkit.tests; gpkit.tests.run()"``
  - If you want units support, install pint with ``pip install pint``.
  - You may find sympy, scipy, and iPython Notebook to be useful additional packages as well.



Windows
=======


1. Install Python dependencies
++++++++++++++++++++++++++++++
  - Install the Python 2.7 version of `Anaconda <http://continuum.io/downloads>`_.
  - If you don't want to install Anaconda, you'll need gcc and pip, and may find sympy, scipy, and iPython Notebook useful.
  - Run ``pip install ctypesgen`` at an Anaconda Command Prompt.


2. Install either the MOSEK or CVXOPT GP solvers
++++++++++++++++++++++++++++++++++++++++++++++++

  - Download `CVXOPT <http://cvxopt.org/download/index.html>`_, then follow `these steps <http://cvxopt.org/install/index.html#building-cvxopt-for-windows>`_ to install a linear algebra library

  - Download `MOSEK <http://mosek.com/resources/downloads>`_, then:
      - Follow `these steps for Windows <http://docs.mosek.com/7.0/toolsinstall/Windows_installation.html>`_.
      - Request an `academic license file <http://license.mosek.com/academic>`_ and put it in ``~/mosek/``


3. Install GPkit
++++++++++++++++
  - Run ``pip install gpkit`` at an Anaconda Command Prompt.
  - Run ``python -c "import gpkit.tests; gpkit.tests.run()"``
  - If you want units support, install pint with ``pip install pint``.

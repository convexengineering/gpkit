Installation Instructions
*************************

If you encounter any bugs during installation, please email gpkit@mit.edu.

Mac OS X
========

1. Install Python and build dependencies
++++++++++++++++++++++++++++++++++++++++
  - Install the Python 2.7 version of `Anaconda <http://continuum.io/downloads>`_.
    - Check that Anaconda is installed: in a Terminal window, run ``python`` and check that the version string it prints while starting includes "Anaconda".
      - If it does not, check that the Anaconda location in ``.profile`` in your home directory (you can run ``vim ~/.profile`` to read it) corresponds to the location of your Anaconda folder; if it doesn't, move the Anaconda folder there, and check again in the ``python`` startup header.
  - If you don't want to install Anaconda, you'll need gcc, pip, numpy, and scipy, and may find iPython Notebook useful as a modeling environment.
  - If ``which gcc`` does not return anything, install the `Apple Command Line Tools <https://developer.apple.com/downloads/index.action?=command%20line%20tools>`_.
  - *Optional:* to install gpkit into an isolated python environment you can create a new conda virtual environment with ``conda create -n gpkit anaconda`` and activate it with ``source activate gpkit``.
  - Run ``pip install ctypesgen --pre`` in the Terminal if you want to use a MOSEK solver.


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
  - _Optional:_ to install gpkit into an isolated python environment, install virtualenv, run ``virtualenv $DESTINATION_DIR`` then activate it with ``source activate $DESTINATION_DIR/bin``.
  - Run ``pip install ctypesgen --pre`` at the command line if you want to use a MOSEK solver.
  - Run ``pip install gpkit`` at the command line.
  - Run ``python -c "import gpkit.tests; gpkit.tests.run()"``
  - If you want units support, install pint with ``pip install pint``.
  - You may find iPython Notebook to be useful modeling environment.



Windows
=======


1. Install Python dependencies
++++++++++++++++++++++++++++++
  - Install the Python 2.7 version of `Anaconda <http://continuum.io/downloads>`_.
  - If you don't want to install Anaconda, you'll need gcc, pip, numpy, and scipy, and may find iPython Notebook useful as a modeling environment.
  - *Optional:* to install gpkit into an isolated python environment you can create a new conda virtual environment with ``conda create -n gpkit anaconda`` and activate it with ``source activate gpkit``.
  - Run ``pip install ctypesgen --pre`` at an Anaconda Command Prompt if you want to use a MOSEK solver.


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

.. _installation:

Installation Instructions
*********************

If you encounter bugs during installation, please email ``gpkit@mit.edu``
or `raise a GitHub issue <http://github.com/hoburg/gpkit/issues/new>`_.


Installation dependencies
====================
To install GPkit, you'll need to have the following python packages already installed on your system:

- ``pip``
- ``numpy`` version 1.8.1 or newer
- ``scipy``
- ``pint``

and at least one solver, which we'll choose and install in a later step.

There are many ways to install these dependencies, but here's our suggestion:

Get ``pip``
-----------

Mac OS X
    Run ``easy_install pip`` at a terminal window.
Linux
    Use your package manager to install ``pip``
        Ubuntu: ``sudo apt-get install python-pip``
Windows
    Install the Python 2.7 64-bit version of `Anaconda <http://www.continuum.io/downloads#_windows>`_.

Get python packages
-------------------

Mac OS X
    Run the following commands:
      - ``pip install pip --upgrade``
      - ``pip install numpy``
      - ``pip install scipy``
      - ``pip install pint``

Linux
    Use your package manager to install ``numpy`` and ``scipy``
        Ubuntu: ``sudo apt-get install python-numpy python-scipy``
    Run ``pip install pint`` (for system python installs, use ``sudo pip``)

Windows
    Do nothing at this step; Anaconda already has the needed packages.


Install a GP solver
===================
GPkit interfaces with two off the shelf solvers: cvxopt, and mosek.
Cvxopt is open source; mosek requires a commercial licence or (free)
academic license.

At least one solver is required.

Installing cvxopt
-----------------

Mac OSX
    Run ``pip install cvxopt``

Linux
    Run ``sudo apt-get install libblas-dev liblapack-dev libsuitesparse-dev`` or otherwise install those libraries

    Run ``pip install cvxopt`` (for system python installs, use ``sudo pip``)

    If experiencing issues with wheel in Ubuntu 16.04, try the `official installer. <http://cvxopt.org/install/index.html>`_

Windows
    Run ``conda install -c omnia cvxopt`` in an Anaconda Command Prompt.

Installing mosek
----------------

Dependency note: GPkit uses the python package ctypesgen to interface with the MOSEK C bindings.

Licensing note: if you do not have a paid license,
you will need an academic or trial license to proceed.

Mac OS X
  - If ``which gcc`` does not return anything, install ``XCode`` and the `Apple Command Line Tools <https://developer.apple.com/downloads/index.action?=command%20line%20tools>`_.
  - Install ctypesgen with ``pip install ctypesgen --pre``.
  - Download `MOSEK <http://mosek.com/resources/downloads>`_, then:
      - Move the ``mosek`` folder to your home directory
      - Follow `these steps for Mac <http://docs.mosek.com/7.0/toolsinstall/Mac_OS_X_installation.html>`_.
      - Request an `academic license file <http://license.mosek.com/academic>`_ and put it in ``~/mosek/``

Linux
  - Install ctypesgen with ``pip install ctypesgen --pre`` (for system python installs, use ``sudo pip``)
  - Download `MOSEK <http://mosek.com/resources/downloads>`_, then:
      - Move the ``mosek`` folder to your home directory
      - Follow `these steps for Linux <http://docs.mosek.com/7.0/toolsinstall/Linux_UNIX_installation_instructions.html>`_.
      - Request an `academic license file <http://license.mosek.com/academic>`_ and put it in ``~/mosek/``

Windows
    - Install ctypesgen by running ``pip install ctypesgen --pre`` in an Anaconda Command Prompt .
    - Download `MOSEK <http://mosek.com/resources/downloads>`_, then:
        - Follow `these steps for Windows <http://docs.mosek.com/7.0/toolsinstall/Windows_installation.html>`_.
        - Request an `academic license file <http://license.mosek.com/academic>`_ and put it in ``C:\Users\(your_username)\mosek\``
        - Make sure ``gcc`` is on your system path.
            - To do this, type ``gcc`` into a command prompt.
            - If you get ``executable not found``, then install the 64-bit version (x86_64 installer architecture dropdown option) of `mingw <http://sourceforge.net/projects/mingw-w64/>`_.
            - Make sure the ``mingw`` bin directory is on your system path (you may have to add it manually).


Install GPkit
=============
  - Run ``pip install gpkit`` at the command line (for system python installs, use ``sudo pip``)
  - Run ``pip install jupyter`` to install jupyter notebook (recommended)
  - Run ``jupyter nbextension enable --py widgetsnbextension`` for interactive control of models in jupyter (recommended)
  - Run ``python -c "import gpkit.tests; gpkit.tests.run()"`` to run the tests; if any tests do not pass, please email ``gpkit@mit.edu`` or `raise a GitHub issue <http://github.com/hoburg/gpkit/issues/new>`_.
  - Join our `mailing list <https://mailman.mit.edu/mailman/listinfo/gpkit-users/>`_ and/or `chatroom <https://gitter.im/gpkit-users/Lobby>`_ for support and examples.


Debugging installation
======================

You may need to rebuild GPkit if any of the following occur:
  - You install a new solver (mosek or cvxopt) after installing GPkit
  - You delete the ``.gpkit`` folder from your home directory
  - You see ``Could not load settings file.`` when importing GPkit, or
  - ``Could not load MOSEK library: ImportError('$HOME/.gpkit/expopt.so not found.')``
To rebuild GPkit, first try running ``python -c "from gpkit.build import rebuild; rebuild()"``. If that doesn't work then try the following:
  - Run ``pip uninstall gpkit``
  - Run ``pip install --no-cache-dir --no-deps gpkit``
  - Run ``python -c "import gpkit.tests; gpkit.tests.run()"``
  - If any tests fail, please email ``gpkit@mit.edu`` or `raise a GitHub issue <http://github.com/hoburg/gpkit/issues/new>`_.


Bleeding-edge / developer installations
=======================================

Active developers may wish to install the `latest GPkit <http://github.com/hoburg/gpkit>`_ directly from the source code on Github. To do so,

  1. Run ``pip uninstall gpkit`` to uninstall your existing GPkit.
  2. Run ``git clone https://github.com/hoburg/gpkit.git`` to clone the GPkit repository.
  3. Run ``pip install -e gpkit`` to install that directory as your environment-wide GPkit.
  4. Run ``cd ..; python -c "import gpkit.tests; gpkit.tests.run()"`` to test your installation from a non-local directory.
